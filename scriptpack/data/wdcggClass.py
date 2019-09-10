import pandas as pd
import numpy as np


class ObsD13C:

    def __init__(self, name, df, header):
        self.name = name
        self.df = df
        self.header = header
        self.station = None
        self.provider = 'wdcgg'
        self.parameter = None
        self.lon = None
        self.lat = None
        self.values = None
        self.time = None
        self.tz_shift = 0
        self.duration = 1
        self.parse_header()
        self.metadata()
        self.data()
        self.std_df()

    def parse_header(self):
        """
        Parse the important attributes in the header.
        """
        for attribute in self.header:
            if 'dataset_parameter' in attribute and not self.parameter:
                self.parameter = attribute.split()[-1]
            elif 'time_zone' in attribute:
                utc_code = [w for w in attribute.split() if 'utc' in w.lower()][0]
                utc_num = utc_code[4:5]
                if len(utc_code) > 5:
                    utc_num += utc_code[5:6]
                if len(utc_code) > 3:
                    self.tz_shift = int(utc_num) * (-1 + 2 * (utc_code[3] == '+'))

            elif 'contributor_acronym' in attribute:
                self.provider = attribute.split()[-1].lower()

        if self.parameter == '13ch4':
            self.parameter = 'd13c'

    def metadata(self):
        """
        Get the metadata of the observation.
        """
        self.assign_time()
        self.assign_lonlat()

    def data(self):
        """
        Fill the values attribute with the NaN-free values of the DataFrame
        """
        self.df = self.df.dropna(subset=['value'])
        self.values = self.df['value']

    def assign_time(self):
        """
        Assign a DateTime array to the index.
        """
        years = self.df['year'].values
        months = self.df['month'].values
        days = self.df['day'].values
        hours = self.df['hour'].values
        minutes = self.df['minute'].values
        seconds = self.df['second'].values

        time = pd.DataFrame({'year': years,
                             'month': months,
                             'day': days,
                             'hour': hours,
                             'minute': minutes,
                             'second': seconds})

        self.df.index = pd.to_datetime(time)
        self.df.index += pd.tseries.offsets.DateOffset(hours=-self.tz_shift)
        self.df = self.df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'])

    def assign_lonlat(self):
        """
        Find the mean latitude and the mean longitude of the station.
        """
        self.lat = int(np.nanmean(self.df['latitude'].values))
        self.lon = int(np.nanmean(self.df['longitude'].values))

    def std_df(self):
        """
        Adapt the DataFrame to make it consistent with monitor files in PyCIF
        """
        self.df['network'] = self.provider
        self.change_cols()
        self.df['station'] = self.df['station'].str.lower()
        self.station = self.df['station'][0]
        self.df['parameter'] = self.parameter
        self.df['duration'] = self.duration
        self.df = self.df[['station', 'network', 'parameter', 'lon', 'lat', 'alt',
                          'i', 'j', 'level', 'obs', 'obserror', 'sim', 'sim_tl',
                          'tstep', 'tstep_glo', 'dtstep', 'duration']]
        self.df = self.df.sort_index()

    def change_cols(self):
        """
        Change the names of the initial columns and fill new ones with NaN
        """
        rename_dict = {'site_gaw_id': 'station',
                       'longitude': 'lon',
                       'latitude': 'lat',
                       'altitude': 'alt',
                       'value': 'obs',
                       'value_unc': 'obserror'}
        self.df = self.df.rename(columns=rename_dict)

        new_cols = ['i', 'j', 'level', 'sim', 'sim_tl', 'tstep',
                    'tstep_glo', 'dtstep', 'duration']
        for new_col in new_cols:
            self.df[new_col] = np.nan

    def get_station(self):
        """
        Return the station
        """
        return self.station





