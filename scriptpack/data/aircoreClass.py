import datetime as dt
import numpy as np


class Aircore:
    locations = [(-117, 34, 'Edwards AFB/Dryden, CA', 'EDR'),
                 (-103, 40, 'Boulder, CO', 'BOU'),
                 (-97, 36, 'Lamont, OK', 'LAM'),
                 (-90, 46, 'Park Falls, WI', 'PAF'),
                 (26, 67, 'Sodankyla, Finland', 'SOD'),
                 (169, -45, 'Lauder, NZ', 'LAU'),
                 (133, -23, 'Alice Springs, Australia', 'ASP'),
                 (1, 43, 'Aire sur l\'Adour, France', 'ASA'),
                 (2, 48, 'Trainou, France', 'TRN'),
                 (-83, 48, 'Timmins, Ontario, Canada', 'TMS'),
                 (21, 67, 'Esrange, Northern Sweden', 'KIR')]

    locs_abbr = [loc[3] for loc in locations]

    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.provider = None
        self.lon = None
        self.lat = None
        self.date = None
        self.loc = None
        self.loc_abbr = None
        self.season = None
        self.ch4 = None
        self.ch4_err = None
        self.pres = None
        self.alt = None

    def metadata(self):
        self.assign_lonlat()
        self.assign_date()
        self.assign_loc()
        self.assign_season()

    def dropna(self):
        self.df = self.df.dropna(subset=['CH4'])
        self.data()

    def data(self):
        self.ch4 = self.df['CH4']
        self.ch4_err = self.df['SD']
        self.pres = self.df['PRES']
        self.alt = self.df['ALT']

    def std_df(self):
        self.df = self.df[['DATE', 'TIME', 'LON', 'LAT', 'ALT', 'PRES', 'CH4', 'SD']]

    def assign_lonlat(self):
        self.lat = int(np.nanmean(self.df['LAT'].values))
        self.lon = int(np.nanmean(self.df['LON'].values))

    def assign_date(self):
        date = str(self.df['DATE'][0])
        time = str(self.df['TIME'][0])
        self.date = dt.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')

    def assign_loc(self):
        locations = self.locations

        for location in locations:
            if abs(location[0]) + 1 >= abs(self.lon) >= abs(location[0]) - 1 and \
                    abs(location[1]) + 1 >= abs(self.lat) >= abs(location[1]) - 1:
                loc_idx = locations.index(location)
                self.loc = locations[loc_idx][2]
                self.loc_abbr = locations[loc_idx][3]

    def assign_season(self):
        if 4 <= self.date.month <= 6:
            self.season = 'spring'

        elif 7 <= self.date.month <= 9:
            self.season = 'summer'

        elif 10 <= self.date.month <= 12:
            self.season = 'automn'

        else:
            self.season = 'winter'

    def get_year(self):
        return self.date.year

    def get_month(self):
        return self.date.month

    def get_season(self):
        return self.season

    @staticmethod
    def get_date(aircore):
        return aircore.date.year, aircore.date.month

    @classmethod
    def get_all_locs(cls):
        abbrs = [abbr[3] for abbr in cls.locations]
        return abbrs


