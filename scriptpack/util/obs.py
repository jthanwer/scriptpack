import xarray as xr
import numpy as np
import pandas as pd


def band_mean_obs_ts(df_obs, lat1, lat2, cols=['obs', 'sim'], df_weights=None):
    """
    Compute the band-averaged time-series of observations using station locations.
    The DataFrame must include a 'station' column/

    :param df_obs: DataFrame of all the observations
    :param lat1: latitude of the southern border
    :param lat2: latitude of the northern border
    :param cols: DataFrame columns to extract

    :return:
    """

    mask = (df_obs['lat'] >= lat1) & (df_obs['lat'] < lat2)
    df_obs_lat = df_obs.loc[mask, ['station'] + cols]
    df_obs_lat = df_obs_lat.groupby(['station', pd.Grouper(freq="MS")]).mean().\
        unstack(level=0).groupby(axis=1, level=0).mean()

    return df_obs_lat


def global_mean_obs_ts(df_obs, lats, cols=['obs', 'sim'], df_weights=None):
    """
    Compute the global-averaged time-series of observations using first band-average.
    The DataFrame must include a 'station' column.

    :param df_obs: DataFrame of all the observations
    :param lats: list of latitudes to provide to band_mean_obs_ts
    :param cols: DataFrame columns to extract

    :return:
    """

    list_dfs = []
    for lat1, lat2 in zip(lats[:-1], lats[1:]):
        df_lat = band_mean_obs_ts(df_obs, lat1, lat2, cols=cols)
        list_dfs.append(df_lat)

    df_concat = pd.concat(list_dfs)
    df_mean = df_concat.groupby(df_concat.index).mean()

    return df_mean

