from datetime import timedelta, datetime
from calendar import isleap
import numpy as np
import pandas as pd
import xarray as xr
from scriptpack.util.useful import *

def float2datetime(array):
    """
    Convert a float decimal date (2000.0412) to a Datetime date.
    :param array: a float decimal time array (for instance DataArray.time)
    :return: new_date : a Datetime array
    """
    new_array = np.empty(array.shape, dtype='datetime64[ns]')
    for i, date in np.ndenumerate(array):
        nbdays = (365 + isleap(int(date))) * (date - int(date))
        new_date = datetime(int(date), 1, 1) + timedelta(days=int(nbdays))
        new_array[i] = new_date
    return pd.DatetimeIndex(new_array)


def add_time_dim(data_array, time_coord):
    """
    Add a 'time' dimension to the DataArray coords

    :param data_array: A DataArray (xarray)
    :param time_coord: the time coord (usually a pd.date_range)
    :return: Same DataArray expanded with a new 'time' dimension
    """
    ones = np.ones((*time_coord.shape, *data_array.shape))

    coords_name = [coord for coord in data_array.coords]
    coords_array = [data_array.coords[coord].values for coord in data_array.coords]
    id_array = xr.DataArray(ones, dims=['time', *coords_name],
                            coords=[time_coord, *coords_array])
    new_da = data_array * id_array
    new_da = new_da.transpose('time', *data_array.dims)

    return new_da


def detect_freq(data_array):
    """
    Add a 'time' dimension to the DataArray coords

    :param data_array: A DataArray (xarray)
    :return: Time Frequency
    """
    time = pd.DatetimeIndex(data_array.time.values)
    time_diff = time[1:] - time[:-1]
    days = time_diff.days.values.mean()

    if days < 5:
        return 'D'

    elif days > 300:
        return 'Y'

    else:
        return 'M'


def broadcast_time(data_array_ref, data_array2adjust):
    """
    Refine time dimension of data_array2adjust considering the one of data_array_ref
    :param data_array_ref: Reference Data Array
    :param data_array2adjust: Data Array to adjust
    :return:
    """
    time_dim_ref = detect_dimensions(data_array_ref)[0]
    time_dim = detect_dimensions(data_array2adjust)[0]
    if time_dim and time_dim_ref:
        time_ref_shape = data_array_ref.shape[0]
        time_shape = data_array2adjust.shape[0]
        repeat_time = [time_ref_shape // time_shape]
        repeat_tuple = tuple(repeat_time + [1] * (len(data_array2adjust.shape) - 1))
        data_array_adjusted = np.tile(data_array2adjust.values, repeat_tuple)
        data_array_adjusted = data2data_array(data_array_ref, data_array_adjusted)
        return data_array_adjusted

    elif not (time_dim and time_dim_ref):
        print("No time dimension detected in one of the data_arrays.")
        return data_array2adjust
