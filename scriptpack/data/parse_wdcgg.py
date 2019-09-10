import pandas as pd
import os
from scriptpack.data.wdcggClass import ObsD13C


def wdcgg_13ch4(dir_path):
    list_stat = os.listdir(dir_path)
    all_df_obs = []

    for stat in list_stat[:]:
        with open(dir_path + stat) as f:
            headerlines = int(f.readline().split()[-1])
            attributes = []
            for i in range(headerlines - 1):
                attributes.append(f.readline())

        names_header = attributes[-1].split()[1:]

        df = pd.read_csv(dir_path + stat, names=names_header,
                         sep=' ', skiprows=headerlines, engine='python', na_values=-999.999)

        df = df[['site_gaw_id', 'year', 'month', 'day', 'hour', 'minute', 'second',
                 'longitude', 'latitude', 'value', 'value_unc', 'altitude']]

        obs_d13c = ObsD13C(df['site_gaw_id'][0], df, attributes)

        all_df_obs.append(obs_d13c)

    return all_df_obs



