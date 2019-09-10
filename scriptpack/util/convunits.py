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
from scriptpack.util.dates import *


def convert_vmr2molec(data_array, climato=False, month=1, freq='M'):
    """
    Convert a field from VMR to molecules/cm3

    :param data_array: A DataArray (xarray) in units VMR
    :param climato: if climatology needed (12 values)
    :param month: in case there is no time dimension, need to give the month of the data
    :param freq: frequency of the data ('D' for daily values, 'M' for monthly values)
    :return: a Numpy Array in molecules/cm3
    """
    nc_file_kin = xr.open_dataset('/home/satellites10/jthanwer/flux_mass/temp_pmid_2017.nc')
    pmid = nc_file_kin['pmid']
    temp = nc_file_kin['temp']

    data_array = clean_data(data_array)
    pmid = clean_data(pmid)
    temp = clean_data(temp)

    time_dim = detect_dimensions(data_array)[0]
    if not time_dim:
        pmid = pmid[month-1, ...]
        temp = temp[month-1, ...]

    if climato:
        data_array = data2climato(data_array)

    if freq == 'D':
        data_values = data_array.values
        data_molec = np.copy(data_values)
        for t in range(data_array.time.shape[0]):
            data_molec[t, ...] = data_values[t, ...] * 1e-6 * \
                                          pmid.values[t // 31, ...] / \
                                          (KB * temp.values[t // 31, ...])
        data_molec = data_array.copy(data=data_molec)

    else:
        pmid = broadcast_time(data_array, pmid)
        temp = broadcast_time(data_array, temp)

        data_molec = data_array * 1e-6 * pmid.values / (KB * temp.values)

    return data_molec


def convert_vmr2mass(data_array, climato=False, month=1, spec='CH4'):
    """
    Convert a field from VMR to mass

    :param data_array: A DataArray (xarray) in units VMR
    :param climato: if climatology needed (12 values)
    :param month: in case there is no time dimension, need to give the month of the data
    :param spec: the studied species
    :return: a Numpy Array in molecules/cm3
    """
    nc_file_mass = xr.open_dataset('/home/satellites10/jthanwer/flux_mass/masse_phi_2017.nc')
    mass = nc_file_mass['masse']

    data_array = clean_data(data_array)
    mass = clean_data(mass)

    M_species = {'CH4': M_CH4, 'OH': M_OH, 'Cl': M_Cl}
    M_spec = M_species[spec]
    mass *= 1000  # conversion en g

    time_dim = detect_dimensions(data_array)[0]
    if not time_dim:
        mass = mass[month, ...]

    if climato:
        data_array = data2climato(data_array)

    mass = broadcast_time(data_array, mass)

    data_mass = (M_spec / M_AIR) * mass.values * data_array

    return data_mass


if __name__ == '__main__':
    nc_file1 = xr.open_dataset('/home/satellites10/jthanwer/champs/inca/inca.an2006-2011.nc')
    nc_file2 = xr.open_dataset('/home/satellites10/jthanwer/champs/transcom/Transcom.new.vmr_MM.nc')
    nc_file3 = xr.open_dataset('/home/satellites10/jthanwer/champs/invsat_Didier/PoleAsia_OXY3_MM_2010-2016.nc')
    OH_1 = nc_file1['OH']
    OH_2 = nc_file2['OH']
    OH_3 = nc_file3['OH']

    lats = OH_1.lat.values

    OH_1 = convert_vmr2molec(OH_1).values
    OH_2 = convert_vmr2molec(OH_2).values
    OH_3 = convert_vmr2molec(OH_3).values

    OH_1 = OH_1.mean((0, 3))
    OH_2 = OH_2.mean((0, 3))
    OH_3 = OH_3.mean((0, 3))
