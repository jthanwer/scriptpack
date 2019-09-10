#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:12:19 2018

@author: jthanwer
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.interpolate import interp1d
import pandas as pd
from scriptpack.util.useful import *
from scriptpack.util.constants import *
from scriptpack.util.grid import *
from scriptpack.util.altitude import *
from scriptpack.util.convunits import *


def area_weighted_average(data_array, hemis=None):
    """
    Compute the average of a grid considering the real area of each model box.

    :param data_array: input 2D DataArray
    :param hemis: choose the hemisphere on which to compute the total mass

    :return: Data Array if time dimension
    """

    data_array = clean_data(data_array)
    time_dim = detect_dimensions(data_array)[0]
    areas = get_areas_lmdz()
    weights_areas = areas / areas.sum()
    weights_areas = np.broadcast_to(weights_areas.values, data_array.shape)
    weights_areas = data_array.copy(data=weights_areas)
    data_meang = data_array * weights_areas
    data_meang = data_meang.sum(dim=['lat', 'lon'])

    return data_meang


def fluxes_weighted_average(data_array, fluxes, reg=None, nreg=None, mean_time=False):
    """
    Compute the fluxes-weighted average of a DataArray.

    :param data_array: input 2D DataArray
    :param fluxes: fluxes, must have the same dimensions as data_array
    :param reg: a DataArray with the regions mask
    :param nreg: the region number in which the area has to be computed
    :param mean_time : compute the mean also over time

    :return: Data Array
    """

    data_array = clean_data(data_array)
    fluxes = clean_data(fluxes)
    areas = get_areas_lmdz(reg, nreg)
    fluxes_area = fluxes * areas
    fluxes_tot = fluxes_area.sum(dim=['lat', 'lon'])
    if mean_time:
        fluxes_tot = fluxes_area.sum(dim=['time', 'lat', 'lon'])
    weights = fluxes_area / fluxes_tot
    data_meang = data_array * weights

    if mean_time:
        return data_meang.sum(dim=['time', 'lat', 'lon'])
    else:
        return data_meang.sum(dim=['lat', 'lon'])


def mass_weighted_column_aver(data_array, pbeg=0, pend=39, climato=False,
                              tropo=False, strato=False, glob=True, month=0):
    """
    Get the statistics over the column by using mass weighted cells. Only works for monthly data DataArray.
    The Data has to be VMR to be physically consistent.

    :param data_array: input Data_array
    :param pbeg: starting pressure index for slicing
    :param pend: ending pressure index for slicing
    :param climato: if climato needed
    :param tropo: if tropospheric mean needed
    :param strato: if stratospheric mean needed
    :param glob: if mean over the whole grid needed
    :param month: if no time dimension and a month has to be given
    :return: Data Array
    """

    # print("Dry air mass-weighted average...")
    nc_file_mass = xr.open_dataset('/home/satellites10/jthanwer/flux_mass/masse_phi_2017.nc')
    mass = nc_file_mass['masse']
    mass = clean_data(mass)
    data_array = clean_data(data_array)

    time_dim = detect_dimensions(data_array)[0]
    if time_dim:
        if climato:
            data_array = data2climato(data_array)
        mass = broadcast_time(data_array, mass)
        time = data_array.time.values
    else:
        print("No time dimension")
        mass = mass[month, ...]

    if tropo:
        data_array = apply_mask_tropo(data_array)
        mass = apply_mask_tropo(mass)

    elif strato:
        data_array = apply_mask_tropo(data_array, strato=True)
        mass = apply_mask_tropo(mass, strato=True)

    else:
        data_array = data_array[dict(pres=slice(pbeg, pend))]
        mass = mass[dict(pres=slice(pbeg, pend))]

    dims2sum = ['pres', 'lat', 'lon'] if glob else ['pres']

    weights = mass / mass.sum(dim=dims2sum)
    weighted_data = data_array * weights.values
    aver_data = weighted_data.sum(dim=dims2sum)
    return aver_data


def volume_weighted_column_aver(data_array, pbeg=0, pend=39, climato=False,
                                tropo=False, strato=False, glob=True, month=1):
    """
        Get the statistics over the column by using volume weighted cells. Only works for monthly data DataArray.
    The Data has to be in molecules/cm3 to be physically consistent.
    The volume has no time dimension.

    :param data_array: input Data_array
    :param pbeg: starting pressure for slicing
    :param pend: ending pressure for slicing
    :param climato: if climato needed
    :param tropo: if tropospheric mean needed
    :param strato: if stratospheric mean needed
    :param glob: if mean over the whole grid needed
    :param month: if no time dimension then a month has to be given
    :return: Data Array
    """

    # print("Volume-weighted average...")
    data_array = lon_adjust(data_array)
    data_array = change_dim_name(data_array)

    time_dim = detect_dimensions(data_array)[0]
    if time_dim:
        time = data_array.time.values
        print("Time dimension with {} values".format(time.shape[0]))
        if climato:
            data_array = data2climato(data_array)
        time = data_array.time.values
    # else:
        # print("No time dimension")

    volume = get_volume_flxmass()
    volume = np.broadcast_to(volume, data_array.shape)
    volume = data2data_array(data_array, volume)

    if tropo:
        data_array = apply_mask_tropo(data_array, month=month)
        volume = apply_mask_tropo(volume, month=month)

    elif strato:
        data_array = apply_mask_tropo(data_array, strato=True)
        volume = apply_mask_tropo(volume, strato=True)

    else:
        data_array = data_array[dict(pres=slice(pbeg, pend))]
        volume = volume[dict(pres=slice(pbeg, pend))]

    dims2sum = ['pres', 'lat', 'lon'] if glob else ['pres']

    weights = volume / volume.sum(dim=dims2sum)
    weighted_data = data_array * weights.values
    aver_data = weighted_data.sum(dim=dims2sum)
    return aver_data


def xconc_molec(data_array, glob=False):
    """
    Compute the integrated column in molecules/cm2
    :param data_array: input Data_array
    :param glob: if column over the whole grid needed
    :return: Data Array
    """
    print("Integration of the column in molecules/cm2...")
    # data_array = lon_adjust(data_array)
    data_array = change_dim_name(data_array)

    alt = get_alt_volume() * 100000  # cm
    alt = np.diff(alt, axis=0)

    weighted_data = data_array * alt
    sum_data = weighted_data.sum(dim='pres')

    if glob:
        return area_weighted_average(sum_data)
    else:
        return sum_data


def total_mass(data_array, spec, pbeg=0, pend=39, climato=False, tropo=False, glob=True, month=0, hemis=None):
    """
    Get the total mass. Only works for monthly data DataArray.

    :param data_array: input DataArray
    :param spec: spec studied 'CH4' 'Cl' or 'OH'
    :param pbeg: pressure starting index for slicing
    :param pend: pressure ending index for slicing
    :param climato: if climato needed
    :param tropo: if tropospheric mean needed
    :param glob: if mean over the whole grid or over a portion of atmosphere needed
    :param month: if no time dimension and a month has to be given
    :param hemis: choose the hemisphere on which to compute the total mass
    :return: Data Array if time dimension
    """

    print("Computing total mass of the species {} ...".format(spec))
    nc_file_mass = xr.open_dataset('/home/satellites10/jthanwer/flux_mass/masse_phi_2017.nc')
    mass = nc_file_mass['masse']
    mass = clean_data(mass)
    data_array = clean_data(data_array)

    time_dim = detect_dimensions(data_array)[0]
    if time_dim:
        if climato:
            data_array = data2climato(data_array)
        mass = broadcast_time(data_array, mass)
        time = data_array.time.values
    else:
        print("No time dimension")
        mass = mass[month, ...]

    if tropo:
        data_array = apply_mask_tropo(data_array)
        mass = apply_mask_tropo(mass)

    pres = data_array.pres.values
    lat, lon = data_array.lat.values, data_array.lon.values

    if hemis == 'N':
        lon_north_slice = {lon: slice(0, 48, 1)}
        mass = mass[lon_north_slice]
        data = data[lon_north_slice]
    elif hemis == 'S':
        lon_south_slice = {lon: slice(0, 48, 1)}
        mass = mass[lon_south_slice]
        data = data[lon_south_slice]

    spec_dic = {'CH4': M_CH4, 'OH': M_OH, 'Cl': M_Cl}
    M_spec = spec_dic[spec]

    mass = mass[dict(pres=slice(pbeg, pend))]
    dims2sum = ['pres', 'lat', 'lon'] if glob else ['pres']
    sum_data = (M_spec / M_AIR) * mass * data_array * 1000  # in g
    sum_data = sum_data.sum(dim=dims2sum)

    return sum_data


def total_mass_ratio(data_array, pbeg=0, pend=39, tropo=False, glob=True, month=0, spec='CH4', hemis=None):
    dataMass_N = total_mass(data_array, pbeg=pbeg, pend=pend,
                            tropo=tropo, glob=glob, month=month, spec=spec, hemis='N')
    dataMass_S = total_mass(data_array, pbeg=pbeg, pend=pend,
                            tropo=tropo, glob=glob, month=month, spec=spec, hemis='S')

    return dataMass_N / dataMass_S
