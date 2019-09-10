#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:12:19 2018

@author: jthanwer
"""

import numpy as np
import xarray as xr


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def lon_adjust(data_array, mtype='lmdz96'):
    lon_shape = data_array.shape[-1]
    if mtype == 'lmdz96':
        return data_array[..., :-1] if lon_shape == 97 else data_array


def create_month_list():
    return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


def degrees_lat(x, pos=0):
    hemis = 'N' if x > 0 else 'S'
    x = np.abs(x)
    return 'EQ.' if x == 0 else '%1.0f%s' % (x, hemis)


def degrees_lon(x, pos=0):
    hemis = 'W' if x > 0 else 'E'
    x = np.abs(x)
    return '%1.0f%s' % (x, hemis)


def find_name_fluxes(file_name):
    idx_beg = file_name.find('.')
    idx_end = file_name.find('.', idx_beg + 1)
    return file_name[:idx_end]


def get_region_name(nreg):
    """
    Return the name of the TransCom region associated to nreg
    :param nreg: The TransCom region number (int)
    :return: The name of the region (str)
    """
    list_regions = {0: 'Oceans',
                    1: 'Boreal North America',
                    2: 'Temperate North America',
                    3: 'Tropical South America',
                    4: 'Temperate South America',
                    5: 'Africa',
                    6: 'Europe',
                    7: 'Russia',
                    8: 'Temperate Asia',
                    9: 'South East Asia',
                    10: 'Oceania'}

    return list_regions[nreg]


def change_dim_name(data_array):
    """
    Change the dimensions names to 'time', 'presnivs', 'lat', 'lon'

    :param data_array: A DataArray (xarray)
    :return: Same DataArray with modified dimensions names
    """
    dims = data_array.dims
    dims_new = ('time', 'pres', 'lat', 'lon')
    for i, dim in enumerate(dims):
        for dim_new in dims_new:
            if dim.lower().find(dim_new.lower()) > -1:
                data_array = data_array.rename({dim: dim_new})
        if dim.lower().find('lev') > -1:
            data_array = data_array.rename({dim: 'pres'})
        elif dim.lower().find('sig') > -1:
            data_array = data_array.rename({dim: 'pres'})
        elif dim.lower().find('alt') > -1:
            data_array = data_array.rename({dim: 'pres'})
    return data_array


def detect_dimensions(data_array):
    """
    Detect the dimensions of the DataArray (xarray)
    :param data_array:
    :return: a list with True if the dimension is detected, None if not. In the order [time, pres, lat, lon].
    """
    data_array = change_dim_name(data_array)
    dims = data_array.dims
    dims2detect = ['time', 'pres', 'lat', 'lon']
    detection = [dim in dims for dim in dims2detect]
    return detection


def clean_data(data_array):
    data_array = change_dim_name(data_array)
    data_array = lon_adjust(data_array)
    return data_array


def data2data_array(data_array, data, varname='data', dataset=False):
    """
    Convert a Numpy array into a DataArray (xarray). Retrieve the dimensions from
    the DataArray. The two arrays need to have the same shape.
    Assuming there is longitude and latitude dimensions.

    :param data_array: A DataArray (xarray) with dimensions needed
    :param data: Numpy array data
    :param varname: Name of the variable
    :param dataset: Return the DataSet instead of the DataArray
    :return: The DataArray with the data from the Numpy array.
    """
    data_array = change_dim_name(data_array)
    dims = detect_dimensions(data_array)
    lats, lons = data_array.lat.values, data_array.lon.values
    if dims[0]:
        time = data_array.time.values
        if dims[1]:
            pres = data_array.pres.values
            ds = xr.Dataset({varname: (['time', 'pres', 'lat', 'lon'], data)},
                            coords={'lat': (['lat'], lats),
                                    'pres': (['pres'], pres),
                                    'lon': (['lon'], lons),
                                    'time': (['time'], time)})
        else:
            ds = xr.Dataset({varname: (['time', 'lat', 'lon'], data)},
                            coords={'time': (['time'], time),
                                    'lat': (['lat'], lats),
                                    'lon': (['lon'], lons)})
    else:
        if dims[1]:
            pres = data_array.pres.values
            ds = xr.Dataset({varname: (['pres', 'lat', 'lon'], data)},
                            coords={'lat': (['lat'], lats),
                                    'pres': (['pres'], pres),
                                    'lon': (['lon'], lons)})
        else:
            ds = xr.Dataset({varname: (['lat', 'lon'], data)},
                            coords={'lat': (['lat'], lats),
                                    'lon': (['lon'], lons)})
    if dataset:
        return ds
    else:
        return ds['data']


def data2climato(data_array):
    """
    Create a climatology (12 months) whether there is a time dimension or not.

    :param data_array: a DataArray (xarray). If time dimension, has to be monthly
    :return: climatology DataArray
    """
    print("Creating a climatology...")
    istime = detect_dimensions(data_array)[0]
    is_other_dim = detect_dimensions(data_array)[1]
    data_array = change_dim_name(data_array)
    lons, lats = data_array.lon.values, data_array.lat.values

    if istime:
        time = data_array.time.values[:12]

        climato = np.array([data_array.values[t::12, ...] for t in range(12)])
        climato = climato.mean(1)

        if is_other_dim:
            pres = data_array.pres.values
            climato_ds = xr.Dataset({'climato': (['time', 'pres', 'lat', 'lon'], climato)},
                                    coords={'lat': (['lat'], lats),
                                            'pres': (['pres'], pres),
                                            'lon': (['lon'], lons),
                                            'time': (['time'], time)})
        else:
            climato_ds = xr.Dataset({'climato': (['time', 'lat', 'lon'], climato)},
                                    coords={'lat': (['lat'], lats),
                                            'lon': (['lon'], lons),
                                            'time': (['time'], time)})

        return climato_ds['climato']

    else:
        print("No time dimension. Creating a time dimension...")
        climato = data_array.values[np.newaxis, ...]
        new_shape = tuple([12] + list(climato.shape[1:]))
        climato = np.broadcast_to(climato, new_shape)
        # TODO : create a time dimension array to create a dataset and return a DataArray

        return climato


def files2climato(path_pattern, var, year='2017', period_in='monthly'):
    """
    Create a climatology out of files (multiple monthly files in a directory)
    :param path_pattern: Path pattern to get the file. Ex : 'file_y{}_m{}.nc'.format(month)
    :param var: The NetCDF variable
    :param year: The year needed
    :param period_in: The time period of the files (either monthly or daily)
    :return: A DataArray (xarray) with the climatology
    """
    month_nums = create_month_list()
    print('Creating a  climatology...')
    for month in month_nums:
        file_name = path_pattern.format(year, month)
        data = xr.open_dataset(file_name, drop_variables=['lat', 'lon'])[str(var)]
        data = change_dim_name(data)
        data = lon_adjust(data)
        print(month)
        print(path_pattern.format(year, month))
        if period_in == 'daily':
            data = data.mean(dim='time')
        if month == '01':
            climato = data
        else:
            climato = xr.concat([climato, data], dim='time')
    return climato


if __name__ == '__main__':
    pattern = 'aaa{}bbb'
    file_init = pattern.format(1)
    print(file_init)
