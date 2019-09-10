#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:48:06 2018

@author: jthanwer
"""
import xesmf as xe
from scriptpack.util.grid import *


def regrid(data_array, lon_out, lat_out, method='conservative'):
    """
    Regrid data. The DataArray given is regridded using rectilinear grid created out of lon_out
    and lat_out.
    The longitude mustn't do a loop. Examples for LMDz = [-180 -176.25 ... 176.25]

    :param data_array: DataArray (xarray) to regrid. Not a Dataset. It has to contain at least 'lat' and 'lon' dimensions.
                 If it is not under those names, the function will deal with it.
    :param lon_out: output centered longitude array
    :param lat_out: output centered latitude array
    :param method: type of interpolation ('bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch')
    :return: data_out: DataArray (xarray)
    """
    data_array = change_dim_name(data_array)
    lon, lat = data_array.lon.values, data_array.lat.values
    lon_b, lat_b = get_grid_corners(lon, lat)
    lon_out_b, lat_out_b = get_grid_corners(lon_out, lat_out)

    grid_in = {'lon': lon, 'lat': lat,
               'lon_b': lon_b, 'lat_b': lat_b}
    grid_out = {'lon': lon_out, 'lat': lat_out,
                'lon_b': lon_out_b, 'lat_b': lat_out_b}

    regridder = xe.Regridder(grid_in, grid_out, method, reuse_weights=True)
    data_out = regridder(data_array)

    return data_out
