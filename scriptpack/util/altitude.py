#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 12:15:57 2018

Conversion of mean value from VMR to molecules.cm-3

@author: jthanwer
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scriptpack.util.useful import *
from scriptpack.util.constants import *

M_AIR = M_AIR / 1000.  # kg/mol
month_nums = create_month_list()


def get_alt_phi(year=2017, month='01', climato=False):
    """
    Compute the altitude using the variable "Phi" from mass fluxes files. The altitude is the one at middle-level.

    :param year: the altitude year needed
    :param month: the altitude month needed if climato=False
    :param climato: if climatology needed (one full year = 12 months)
    :return: DataArray (xarray) of altitude in km
    """
    if climato:
        file_pattern = '/home/satellites1/fcheval/LMDZ5/fluxstoke.an{}.m{}.nc'
        phi_tot = files2climato(file_pattern, 'phi')

    else:
        path_file = '/home/satellites1/fcheval/LMDZ5/fluxstoke.an{}.m{}.nc'.format(year, month)
        phi_tot = xr.open_dataset(path_file)['phi']
        phi_tot = change_dim_name(phi_tot)
        phi_tot = lon_adjust(phi_tot)

    altitude = phi_tot / (1000 * GRAV)

    return altitude


def get_alt_volume(year=2017, month='01', climato=False):
    """
    Compute the altitude using the volume (computed with temp, mass,etc...). The altitude is the one at inter-level.
    Compute also the pressure at mid-level by using ap and bp coefficients and pres_surf deduced from phi_surf.

    :param year: the altitude year needed
    :param month: the altitude month needed if climato=False
    :param climato: TODO : give a climato
    :return: altitude (numpy array) in km
    """
    df = pd.read_csv("/home/satellites10/jthanwer/flux_mass/ap_bp.txt", sep=' ')
    ap39 = df['ap'].values
    bp39 = df['bp'].values
    flux_path = '/home/comdata1/PYVARLMDZ_files_96x96x39/MASS_FLUXES/TD/fluxstoke.an{}.m{}.nc'.format(year, month)
    phy_path = '/home/comdata1/PYVARLMDZ_files_96x96x39/MASS_FLUXES/TD/phystoke.an{}.m{}.nc'.format(year, month)
    flux_file = xr.open_dataset(flux_path)

    aire = flux_file['aire']
    masse = flux_file['masse']
    phi_surf = flux_file['phis']
    temp = xr.open_dataset(phy_path)['t']

    variables = [aire, masse, phi_surf, temp]
    for i, var in enumerate(variables):
        var2 = change_dim_name(var)
        if var2.shape[-1] == 97:
            var2 = lon_adjust(var2)
        variables[i] = var2.values

    aire, masse, phi_surf, temp = variables

    ref = change_dim_name(flux_file['masse'])
    time = ref.time.values
    time = time[int(len(time) / 2)]
    pres = ref.pres.values
    lat, lon = ref.lat.values, ref.lon.values

    masse = masse.mean(0)
    temp = temp.mean(0)

    npres = masse.shape[0]
    nlat = masse.shape[1]
    nlon = masse.shape[2]

    sum_mass = np.sum(masse, axis=0)
    p_surf = sum_mass * GRAV / aire
    p_surf /= 100.
    z_surf = phi_surf / GRAV

    pres = np.zeros((npres, nlat, nlon))
    z_delta = np.zeros((npres, nlat, nlon))
    altitude = np.zeros((npres + 1, nlat, nlon))
    altitude[0, ...] = z_surf
    for p in range(npres):
        pres[p, ...] = 0.5 * (
                ap39[p] + bp39[p] * p_surf + ap39[p + 1] + bp39[p + 1] * p_surf) * 100.  # middle-levels
        z_delta[p, ...] = masse[p, ...] * R_GAZ * temp[p, ...] / (aire[:] * M_AIR * pres[p, ...])
        altitude[p + 1, ...] = z_delta[p, ...] + altitude[p, ...]

    altitude /= 1000 #altitude in km

    return altitude


def get_pressure_mid(year=2012, month='01', climato=False):
    """
        Compute the pressure by using ap and bp coefficients and pres_surf deduced from phi_surf.
        The pressure is the one at mid-levels.

        :param year: the altitude year needed
        :param month: the altitude month needed if climato=False
        :param climato: TODO : give a climato
        :return: daily pressure (numpy array
        """
    df = pd.read_csv("/home/satellites10/jthanwer/flux_mass/ap_bp.txt", sep=' ')
    ap39 = df['ap'].values
    bp39 = df['bp'].values
    flux_path = '/home/satellites1/fcheval/LMDZ5/fluxstoke.an{}.m{}.nc'.format(year, month)
    phy_path = '/home/satellites1/fcheval/LMDZ5/phystoke.an{}.m{}.nc'.format(year, month)
    flux_file = xr.open_dataset(flux_path)

    aire = flux_file['aire']
    masse = flux_file['masse']
    phi_surf = flux_file['phis']
    ref = lon_adjust(masse)
    ref = change_dim_name(ref)

    variables = [aire, masse, phi_surf]
    for i, var in enumerate(variables):
        var2 = change_dim_name(var)
        if var2.shape[-1] == 97:
            var2 = lon_adjust(var2)
        variables[i] = var2.values

    aire, masse, phi_surf = variables
    ntime, npres, nlat, nlon = masse.shape

    sum_mass = np.sum(masse, axis=1)
    p_surf = sum_mass * GRAV / aire
    p_surf /= 100.

    pres = np.zeros(masse.shape)
    for p in range(npres):
        pres[:, p, ...] = 0.5 * (
                ap39[p] + bp39[p] * p_surf + ap39[p + 1] + bp39[p + 1] * p_surf) * 100.  # middle-levels

    pres = data2data_array(ref, pres)

    return pres


def get_pressure_inter(year=2012, month='01', climato=False):
    """
        Compute the pressure by using ap and bp coefficients and pres_surf deduced from phi_surf.
        The pressure is the one at mid-levels.

        :param year: the altitude year needed
        :param month: the altitude month needed if climato=False
        :param climato: TODO : give a climato
        :return: daily pressure (numpy array
        """
    df = pd.read_csv("/home/satellites10/jthanwer/flux_mass/ap_bp.txt", sep=' ')
    ap39 = df['ap'].values
    bp39 = df['bp'].values
    flux_path = '/home/satellites1/fcheval/LMDZ5/fluxstoke.an{}.m{}.nc'.format(year, month)
    phy_path = '/home/satellites1/fcheval/LMDZ5/phystoke.an{}.m{}.nc'.format(year, month)
    flux_file = xr.open_dataset(flux_path)

    aire = flux_file['aire']
    masse = flux_file['masse']
    phi_surf = flux_file['phis']
    ref = lon_adjust(masse)
    ref = change_dim_name(ref)

    variables = [aire, masse, phi_surf]
    for i, var in enumerate(variables):
        var2 = change_dim_name(var)
        if var2.shape[-1] == 97:
            var2 = lon_adjust(var2)
        variables[i] = var2.values

    aire, masse, phi_surf = variables
    ntime, npres, nlat, nlon = masse.shape

    sum_mass = np.sum(masse, axis=1)
    p_surf = sum_mass * GRAV / aire
    p_surf /= 100.

    pres = np.zeros(masse.shape)
    for p in range(npres):
        pres[:, p, ...] = (ap39[p] + bp39[p] * p_surf) * 100.  # inter-levels

    pres = data2data_array(ref, pres)

    return pres


def get_presnivs():
    """
    Standard pressure at inter-levels
    """
    dir_apbp = '/home/satellites10/jthanwer/flux_mass/'
    df = pd.read_csv(dir_apbp + 'pres_alt.txt', sep=' ')
    presnivs = df['presnivs'].values

    return presnivs


def get_alt_presnivs():
    dir_apbp = '/home/satellites10/jthanwer/flux_mass/'
    df = pd.read_csv(dir_apbp + 'pres_alt.txt', sep=' ')
    altitude = df['altitude'].values

    return altitude


def get_volume_flxmass():
    """
    Compute the volume of each box by using the altitude given by get_alt_volume().

    :return: a Numpy Array with the volume of each box
    """
    altitude = get_alt_volume() * 1000
    flux_path = '/home/comdata1/PYVARLMDZ_files_96x96x39/MASS_FLUXES/TD/fluxstoke.an2017.m01.nc'
    flux_file = xr.open_dataset(flux_path)
    aire = flux_file['aire'].values

    aire = lon_adjust(aire)
    altitude_diff = np.diff(altitude, axis=0)
    volume = altitude_diff * aire

    return volume


def get_volume_kin():
    """
    Compute the volume of each box of the model by using variables from kinetic files (chemistry)
    :return: a Numpy Array with the volume of each box
    """
    nc_file_kin = xr.open_dataset('/home/satellites10/jthanwer/flux_mass/temp_pmid_2017.nc')
    nc_file_mass = xr.open_dataset('/home/satellites10/jthanwer/flux_mass/masse_phi_2017.nc')
    mass = nc_file_mass['masse'].values
    pmid = nc_file_kin['pmid'].values
    temp = nc_file_kin['temp'].values

    mass = lon_adjust(mass)
    mass *= 1000 # conversion into g

    volume = mass * temp * R_GAZ / (M_AIR * pmid)
    print(np.sum(volume.mean(0)) / (AIRE * 1000)) # top of the model

    return volume


def get_tropo_lvl(mean_tuple=False):
    """
    Return the levels of the tropopause associated with the presnivs levels
    :return: climatology of levels of tropopause
    """
    df2 = pd.read_csv("/home/satellites10/jthanwer/flux_mass/pres_alt.txt", sep=' ')
    pres = df2['presnivs'].values
    alt = df2['altitude2'].values

    nc_file_tropo = xr.open_dataset("/home/satellites10/jthanwer/flux_mass/tropo_96x96.nc")
    data_array = nc_file_tropo['ptp']

    if mean_tuple:
        data_array = data_array.mean(axis=mean_tuple)

    pres_tropo = data_array.values
    pres_tropo /= 100.

    lvl_tropo = np.zeros(pres_tropo.shape, dtype=int)
    for idx, value in np.ndenumerate(lvl_tropo):
        lvl_tropo[idx] = find_nearest(pres, pres_tropo[idx])

    lvl_tropo = data_array.copy(data=lvl_tropo)

    return lvl_tropo


def apply_mask_tropo(data_array, month=1, forced=False, strato=False, value=np.nan, freq='M'):
    """
    Apply a tropopause mask to the data_array.
    
    :param data_array: A climatology Data Array (xarray) - 12 months if time dimension.
    :param month: If no time dimension. Select the month needed.
    :param strato: If stratosphere needed rather than troposphere.
    :param forced: If we want to force the month
    :param value: value for rescaling the tropo if strato=True and the opposite if False
    :param freq: frequency of the data ('D' for daily values, 'M' for monthly values)
    :return: The data_array with mask (NaN values above tropopause).
    """
    if data_array.shape[-1] == 96:
        nc_file_tropo = xr.open_dataset("/home/satellites10/jthanwer/flux_mass/tropo_96x96.nc")
    else:
        nc_file_tropo = xr.open_dataset("/home/satellites10/jthanwer/flux_mass/tropo_97x96.nc")
    tropo_pres = nc_file_tropo['ptp']
    data_array = change_dim_name(data_array)
    time_dim = detect_dimensions(data_array)[0]

    pres_array = np.zeros(data_array.shape)
    df2 = pd.read_csv("/home/satellites10/jthanwer/flux_mass/pres_alt.txt", sep=' ')
    presnivs = df2['presnivs'].values

    for p in range(data_array.pres.shape[0]):
        if time_dim:
            pres_array[:, p, :, :] = presnivs[p] * 100
        else:
            pres_array[p, :, :] = presnivs[p] * 100

    if time_dim:
        data_masked_tropo = np.zeros(data_array.shape)
        data_masked_strato = np.zeros(data_array.shape)
        if freq == 'M':
            if forced:
                for m in range(data_array.shape[0]):
                    data_masked_tropo[m] = np.where(pres_array[m] >= tropo_pres.values[month-1], data_array[m],
                                                    data_array[m]*value)
                    data_masked_strato[m] = np.where(pres_array[m] < tropo_pres.values[month-1], data_array[m],
                                                     data_array[m]*value)
            else:
                for m in range(data_array.shape[0]):
                    data_masked_tropo[m] = np.where(pres_array[m] >= tropo_pres.values[m % 12], data_array[m],
                                                    data_array[m]*value)
                    data_masked_strato[m] = np.where(pres_array[m] < tropo_pres.values[m % 12], data_array[m],
                                                     data_array[m]*value)
        elif freq == 'D':
            for d in range(data_array.shape[0]):
                data_masked_tropo[d] = np.where(pres_array[d] >= tropo_pres.values[month-1], data_array[d],
                                                data_array[d] * value)
                data_masked_strato[d] = np.where(pres_array[d] < tropo_pres.values[month-1], data_array[d],
                                                 data_array[d] * value)
    else:
        data_masked_tropo = np.where(pres_array >= tropo_pres.values[month-1], data_array, data_array*value)
        data_masked_strato = np.where(pres_array < tropo_pres.values[month-1], data_array, data_array*value)

    data_masked_tropo = data_array.copy(data=data_masked_tropo)
    data_masked_strato = data_array.copy(data=data_masked_strato)

    if strato:
        return data_masked_strato
    else:
        return data_masked_tropo


if __name__ == '__main__':
    # nc_file1 = xr.open_dataset('/home/satellites10/jthanwer/simus/simu_flxGCP/simu_INCA_totCH4_Cl/2012-2017_Cl.nc')
    # CH4_1 = nc_file1['CH4'] * M_AIR / M_CH4
    # CH4_1 = lon_adjust(CH4_1)
    # CH4_1 = data2climato(CH4_1)
    # mt = apply_mask_tropo(CH4_1[0,...])
    pressure = get_alt_phi(2017, '01')
    print(pressure.values.mean((0,2,3)))
    # pressure.to_netcdf('/home/users/jthanwer/pressure.nc')
    # print(pressure.mean(axis=(0, 2, 3)))
    # print(pressure.std(axis=(0, 2, 3)))


