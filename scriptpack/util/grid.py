#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:48:06 2018

@author: jthanwer
"""
import numpy as np
import xarray as xr
import pandas as pd
import os
from calendar import isleap
from scriptpack.util.useful import *
from scriptpack.util.constants import *
from scriptpack.util.dates import *


def get_grid_lmdz():
    """
    Return the LMDz 96*96 coordinates taken from a grid file used in PyVAR.
    LMDz grid has particularities :
    Number of boxes in latitude is 95 (2 half-boxes for poles and 94 full-boxes). The file gives us the centered
    coordinates and loop the longitude, resulting in a number of longitude centers equal to 97 instead of 96.
    This loop is useless.

    :return: zlon, zlat : centered coordinates
    """

    file_grid = "/home/users/aberchet/PYVAR/msaunois/pyvar/grid_LMDZ96_96.txt"

    with open(file_grid, 'r') as fgrid:
        # Loading geometry
        nlon = int(fgrid.readline())
        zlon = [float(fgrid.readline()) for _ in range(nlon)]

        nlat = int(fgrid.readline())
        zlat = [float(fgrid.readline()) for _ in range(nlat)]

        # Centered coordinates
        zlon, zlat = np.array(zlon[:-1]), np.array(zlat)  # delete last element of lon whose goal is to loop

    return zlon, zlat


def get_grid_corners(lon, lat):
    """
    Return the coordinates of the corners of a rectilinear grid. Must give the coordinates of the centers as input.

    :param lon: centered longitude coordinates
    :param lat: centered latitude coordinates
    :return: lon_b, lat_b : cornered coordinates
    """
    offset_lon = (lon[1] - lon[0]) / 2
    offset_lat = (lat[1] - lat[0]) / 2
    lon_b = np.concatenate((lon - offset_lon, [lon[-1] + offset_lon]), axis=0)
    lat_b = np.concatenate((lat - offset_lat, [lat[-1] + offset_lat]), axis=0)

    if abs(lat_b[0]) > 90:
        lat_b[0] *= 90 / abs(lat_b[0])
    if abs(lat_b[-1]) > 90:
        lat_b[-1] *= 90 / abs(lat_b[-1])

    return lon_b, lat_b


def lmdz2pyvar(data_array, name='flx_ch4'):
    """
    Convert a flux vector from LMDz grid to Pyvar grid, i.e. vector 9026 = 94 * 96 + 2
    :param data_array: a DataArray (xarray)
    :param name: name of the variable
    :return: flx : the DataArray (ntimes, 9026)
    """
    data_array = change_dim_name(data_array)
    lon, lat = data_array.lon.values, data_array.lat.values
    flx = data_array.values
    time = data_array.time.values
    ntimes = data_array.shape[0]

    if lat[0] < lat[-1]:
        flx = np.flip(flx, 1)

    values_north = flx[:, 0, :].mean(1)  # longitude average
    values_south = flx[:, -1, :].mean(1)
    values_north = values_north[:, np.newaxis]
    values_south = values_south[:, np.newaxis]

    flx = flx[:, 1:-1, :]
    flx = np.reshape(flx, (ntimes, 9024))
    flx = np.append(values_north, flx, axis=1)
    flx = np.append(flx, values_south, axis=1)

    vector = np.linspace(1, 9026, 9026)
    ds_out = xr.Dataset({name: (['time', 'vector'], flx)},
                        coords={'vector': (['vector'], vector),
                                'time': (['time'], time)})

    return ds_out[name]


def pyvar2lmdz(data_array):
    """
    Convert a flux vector from Pyvar grid to LMDz grid
    :param data_array: a DataArray (xarray)
    :return: fch4 : the DataArray (ntimes, 96, 96)
    """
    data_array = change_dim_name(data_array)
    flx = data_array.values  # get a (ntimes, 9026) vector
    ntimes = data_array.shape[0]
    time = data_array.time
    lons, lats = get_grid_lmdz()

    values_north = flx[:, 0]
    values_north = np.broadcast_to(values_north[:, np.newaxis], (ntimes, 96))
    values_north = values_north[:, np.newaxis, :]
    values_south = flx[:, -1]
    values_south = np.broadcast_to(values_south[:, np.newaxis], (ntimes, 96))
    values_south = values_south[:, np.newaxis, :]

    flx = flx[:, 1:-1]
    flx = np.reshape(flx, (ntimes, 94, 96))
    flx = np.append(values_north, flx, axis=1)
    flx = np.append(flx, values_south, axis=1)

    ds_out = xr.Dataset({'fch4': (['time', 'lat', 'lon'], flx)},
                        coords={'lat': (['lat'], lats),
                                'lon': (['lon'], lons),
                                'time': (['time'], time)})

    return ds_out['fch4']


def lmdz2pyvar_wt(data_array):
    """
    Convert an array without time dimension from LMDz grid to Pyvar grid, i.e. vector 9026 = 94 * 96 + 2
    :param data_array: a DataArray (xarray)
    :return: flx : the DataArray (9026)
    """
    data_array = change_dim_name(data_array)
    lon, lat = data_array.lon.values, data_array.lat.values
    data = data_array.values

    if lat[0] < lat[-1]:
        data = np.flip(data, 1)

    value_north = data[0, :].mean(0)  # longitude average
    value_south = data[-1, :].mean(0)

    data = data[1:-1, :]
    data = np.reshape(data, 9024)
    data = np.append([value_north], data, axis=0)
    data = np.append(data, [value_south], axis=0)

    vector = np.linspace(1, 9026, 9026)
    ds_out = xr.Dataset({'data': (['vector'], data)},
                        coords={'vector': (['vector'], vector)})

    return ds_out['data']


def integrate_field(data_array, reg=None, nreg=None, freq_out='Y'):
    """
        Integrate field over the global space, the lon and lat values must be centered.

        :param data_array: A monthly (freq='M') DataArray
        :param freq_out: A pandas frequency to retrieve monthly or annual emissions
        :param reg: a DataArray with the regions mask
        :param nreg: the region number in which the area has to be computed
        :return: value of integrated field
        """
    data_array = change_dim_name(data_array)

    lon, lat = data_array.lon.values, data_array.lat.values
    ntime = data_array.shape[0]

    nbdays_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    nbdays_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    nbdays = xr.DataArray(np.array(nbdays_noleap * (ntime // 12)),
                          coords=[data_array.time],
                          dims=['time'])

    for year in pd.DatetimeIndex(data_array.time.values).year:
        if isleap(year):
            nbdays.loc[str(year)] = np.array(nbdays_leap)

    areas = get_areas_lmdz(reg, nreg)
    areas = np.broadcast_to(areas.values, data_array.shape)
    areas = data_array.copy(data=areas)
    flux_intensity = data_array * areas * 86400 * nbdays / 1e9

    # Tg/month
    total = flux_intensity.sum(axis=(1, 2))

    if freq_out == 'Y':
        total = total.groupby('time.year').sum()\
            .rename({'year': 'time'})
        total['time'] = float2datetime(total.time.values)

    return total


def get_areas_lmdz(reg=None, nreg=False):
    """
    Compute the area and the total surface of the LMDz grid.
    A region can be given and in that case, the other areas are set to zero.

    :param reg: a DataArray with the regions mask
    :param nreg: the region number in which the area has to be computed
    :return: surface : The total area of a LMDz grid (total surface of the Earth)
             area : A DataArray with the area of each LMDz box.
    """
    lon, lat = get_grid_lmdz()
    lon_b, lat_b = get_grid_corners(lon, lat)
    nlon, nlat = lon.shape[0], lat.shape[0]

    dlon = 2 * PI / nlon

    areas = R_TERRE * R_TERRE * dlon * \
                   np.abs(np.sin(lat_b[:-1] * PI / 180) - np.sin(lat_b[1:] * PI / 180))
    areas = np.broadcast_to(areas[:, np.newaxis], (nlat, nlon))

    areas = xr.DataArray(areas, coords=[lat, lon], dims=['lat', 'lon'])

    if np.array(reg).any() and isinstance(nreg, int):
        mask = xr.DataArray(np.where(reg.values != nreg, 0, 1), coords=[lat, lon], dims=['lat', 'lon'])
        areas = areas * mask

    return areas


if __name__ == '__main__':
    pass
