import pandas as pd
import os
import datetime as dt
from .aircoreClass import Aircore


def noaa2class():
    ac_dir = '/home/users/jthanwer/obs/aircores_total/Final_NOAA/'
    list_ac = os.listdir(ac_dir)
    all_aircores = []

    for ac in list_ac:
        df = pd.read_csv(ac_dir + ac)

        rename_columns = {'DateTime(yyyy-mm-dd HH:MM:SS)': 'DATE',
                          ' SecondsFromMidnight(s)': 'TIME',
                          ' Lon(E)': 'LON',
                          ' Lat(N)': 'LAT',
                          ' Alt(m)': 'ALT',
                          ' Pressure(mb)': 'PRES',
                          ' CH4(ppb)': 'CH4',
                          ' ch4_err': 'SD'}
        df.rename(columns=rename_columns, inplace=True)
        df['DATE'] = df['DATE'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df['TIME'] = df['DATE'].apply(lambda x: x.strftime('%H:%M:%S'))
        df['DATE'] = df['DATE'].apply(lambda x: x.strftime('%Y-%m-%d'))
        # df_rock['PRES'] = df_rock['PRES'] * 100

        aircore = Aircore(ac, df)

        aircore.metadata()
        aircore.data()
        aircore.provider = 'noaa'
        all_aircores.append(aircore)

    return all_aircores


def fr12class():
    ac_dir = '/home/users/jthanwer/obs/aircores_total/Final_France1/'
    list_ac = os.listdir(ac_dir)
    all_aircores = []

    for ac in list_ac:
        with open(ac_dir + ac, 'r') as f:
            nb_line = 1
            for line in f:
                if line.find('---------') >= 0:
                    break
                nb_line += 1
        df = pd.read_csv(ac_dir + ac,
                         sep='\s+', skipinitialspace=True, skiprows=nb_line + 1, header=[0,1],
                         na_values='99999.00', engine='python')

        rename_columns = {'Date': 'DATE',
                          'Time': 'TIME',
                          'Long.': 'LON',
                          'Lat.': 'LAT',
                          'Alt': 'ALT',
                          'P': 'PRES',
                          'CH4': 'CH4',
                          'sdCH4': 'SD'}
        df.rename(columns=rename_columns, inplace=True)
        df.columns = df.columns.droplevel(-1)
        df['DATE'] = df['DATE'].apply(lambda x: str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8])
        df['TIME'] = df['TIME'].apply(lambda x: "{:06d}".format(x)[0:2] + ':'
                                                + "{:06d}".format(x)[2:4] + ':'
                                                + "{:06d}".format(x)[4:6])
        df['ALT'] = df['ALT'] * 1000
        # df_rock['PRES'] = df_rock['PRES'] * 100

        aircore = Aircore(ac, df)

        aircore.metadata()
        aircore.data()
        aircore.provider = 'ipsl'
        all_aircores.append(aircore)

    return all_aircores


def fr22class():
    ac_dir = '/home/users/jthanwer/obs/aircores_total/Final_France2/'
    list_ac = os.listdir(ac_dir)
    all_aircores = []
    locations = [(21, 67, 'Esrange, Northern Sweden', 'KIR'),
                 (-83, 48, 'Timmins, Ontario, Canada', 'TMS')]
    for ac in list_ac:
        with open(ac_dir + ac, 'r') as f:
            nb_line = 1
            count_t = 0
            for line in f:
                if line.find('---------') >= 0:
                    if count_t > 0:
                        break
                    count_t += 1
                sec = line.find('seconds since')
                if sec > 0:
                    sec = line.find('20', sec)
                    timestamp = line[sec:sec+19]
                nb_line += 1

        df = pd.read_csv(ac_dir + ac,
                         sep='\s+', skipinitialspace=True, skiprows=nb_line, header=0,
                         na_values='99999.00', engine='python')

        date_ini = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        df['Time(s)'] = df['Time(s)'].apply(lambda x: dt.timedelta(seconds=x)) + date_ini
        df['DATE'] = pd.Series(df['Time(s)'].apply(lambda x: x.strftime('%Y-%m-%d')), index=df.index)
        df['TIME'] = pd.Series(df['Time(s)'].apply(lambda x: x.strftime('%H:%M:%S')), index=df.index)
        del df['Time(s)']
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df = df[cols]

        rename_columns = {'Date': 'DATE',
                          'lon(degE)': 'LON',
                          'lat(degN)': 'LAT',
                          'Alt(km)': 'ALT',
                          'P(hPa)': 'PRES',
                          'CH4(ppb)': 'CH4',
                          'sdCH4(ppb)': 'SD'}
        df.rename(columns=rename_columns, inplace=True)
        df['ALT'] = df['ALT'] * 1000
        # df_rock['PRES'] = df_rock['PRES'] * 100

        aircore = Aircore(ac, df)

        aircore.metadata()
        aircore.data()
        aircore.provider = 'ipsl'
        all_aircores.append(aircore)

    return all_aircores



