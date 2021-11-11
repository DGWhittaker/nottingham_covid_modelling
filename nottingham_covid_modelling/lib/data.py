import os

import cma
import matplotlib.pyplot as plt
import numpy as np
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib._command_line_args import POPULATION
from nottingham_covid_modelling.lib.equations import tanh_spline, step
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE


SPLINE_FITS = None


class DataLoader(object):
    def __init__(self, use_ons_data, parameters, country, data_dir=None):

        '''Data retrieval functions'''    

        self.p = parameters
        self.country_str = country
        self.country_display = country.replace(' ', '-')

        self.p.N = POPULATION[self.country_str]
        
        self.data_dir = data_dir if data_dir is not None else os.path.join(MODULE_DIR, '..', '..', 'data', 'archive', 'current')
            
        self.google_data = self._get_google_data()
        self.lgoog_data = len(self.google_data)
        
        if use_ons_data:
            self.p.day_1st_death_after_150220 = 22 # Feb 15th to March 8th
            self.p.N = 59.1e6  # Adjust population to England + Wales
            self.country_display = 'England+Wales'
            self.daily_deaths = self._get_ons_data()
        else:
            self.daily_deaths = self._get_global_data()
        fit_synthetic_data = False
        if fit_synthetic_data:
            self.daily_deaths = np.load('syn_data_phi0.001.npy')
        # self.daily_deaths = np.load('SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy')[self.p.day_1st_death_after_150220:-45, 4]
        self.google_data = self._smooth_truncate_google_data(self.google_data)

    def _get_global_data(self):

        '''Retrieves stored global data on daily deaths'''

        date, daily = [], []

        self.num_days_between_311219_150220 = 46

        with open(os.path.join(self.data_dir, 'owid-covid-data.csv'), encoding='utf8') as csvfile:
            header = csvfile.readline()
            assert header, "Got an empty file!"
            header = header.strip().split(',')           
            location_index = header.index('location')
            date_index = header.index('date')
            new_deaths_index = header.index('new_deaths')

            self.counter = 0
            self.flag = True
            row = csvfile.readline()
            while row:
                row = row.strip().split(',')
                if row[location_index] == self.country_str:
                    date.append(row[date_index])
                    new_deaths = row[new_deaths_index] if row[new_deaths_index] != '' else '0'
                    if float(new_deaths) < 0:
                        new_deaths = '0'
                    assert '.' not in new_deaths or new_deaths.endswith('.0'), \
                        "Expecting new_deaths to be a whole number"
                    if int(float(new_deaths)) > 0 and self.flag:
                        self.p.day_1st_death_after_150220 = self.counter - self.num_days_between_311219_150220
                        self.flag = False
                        print('Date of first death: ' + str(row[date_index]))
                    else:
                        self.counter += 1
                    daily.append(int(float(new_deaths)))
                row = csvfile.readline()

        # Align global data with Google mobility data
        date = date[self.num_days_between_311219_150220 + self.p.day_1st_death_after_150220:]
        daily = daily[self.num_days_between_311219_150220 + self.p.day_1st_death_after_150220:]

        # Compute remaining mismatch between dates
        self.mismatch = self._get_mismatch(len(date))

        # Truncate daily deaths to same length as Google travel data
        if self.mismatch > 0:
            daily = daily[:-self.mismatch]
            date = date[:-self.mismatch]

        self.last_date = date[-1]
        print('First date: ' + str(date[0]) + ', last date: ' + str(date[-1]))

        return daily

    def _get_ons_data(self):

        '''Retrieves auto-updated ONS data (for England + Wales) on daily deaths'''

        date, daily = [], []

        with open(os.path.join(self.data_dir, 'ONS_daily_deaths.csv'), encoding='utf8') as csvfile:
            header = csvfile.readline()
            assert header, "Got an empty file!"
            header = header.strip().strip().split(',')
            date_index = header.index('date')
            dailys_deaths_index = header.index('dailys_deaths')

            row = csvfile.readline()
            while row:
                row = row.strip().split(',')
                date.append(row[date_index])
                daily.append(int(row[dailys_deaths_index]))
                row = csvfile.readline()
        
        # Check the isolated case before March 2nd and erase it if necessary
        assert date[0] == '2020-01-30' and date[4] == '2020-03-04', "ONS data format changed and doesn't match our method anymore"
        print('Removed: isolated first deaths at ' + date[0] + ' and ' + date[3])
        print('Date of second death: ' + date[4])
        date = date[7:]
        daily = daily[7:]
        
        # Compute remaining mismatch between dates
        self.mismatch = self._get_mismatch(len(date))

        # Truncate daily deaths to same length as Google travel data
        if self.mismatch > 0:
            daily = daily[:-self.mismatch]
            date = date[:-self.mismatch]

        self.last_date = date[-1]
        print('ONS first date: ' + str(date[0]) + ', last date: ' + str(date[-1]))

        return daily

    def _get_google_data(self):

        '''Retrieves auto-updated Google mobility data'''

        date, self.p.alpha = [], []
        with open(os.path.join(self.data_dir, 'Global_Mobility_Report.csv'), encoding='utf8') as csvfile:
            header = csvfile.readline()
            assert header, "Got an empty file!"
            header = header.strip().split(',')
            date_index = header.index('date')
            country_index = header.index('country_region')
            sub_region_1_index = header.index('sub_region_1')
            if self.p.transit:
                transit_index = header.index('transit_stations_percent_change_from_baseline')
            workplaces_index = header.index('workplaces_percent_change_from_baseline')

            try:
                metro_area_index = header.index('metro_area')
            except ValueError:  # older csvs don't have this column
                metro_area_index = -1

            row = csvfile.readline()

            while row:
                row = row.strip().split(',')
                if row[country_index] == self.country_str and row[sub_region_1_index] == '' and (metro_area_index < 0 or row[metro_area_index] == ''):
                    date.append(row[date_index])
                    # Determine optimal weighting!
                    if self.p.transit:
                        self.p.alpha.append(1.0 + (float(row[transit_index]) + float(row[workplaces_index])) / 200)
                    else:
                        self.p.alpha.append(1.0 + (float(row[workplaces_index])) / 100 )
                row = csvfile.readline()

        if self.p.n_days_to_simulate_after_150220 > 0:
            assert self.p.n_days_to_simulate_after_150220 < len(date), "Cannot go beyond existing data"
            cutoff = len(date) - self.p.n_days_to_simulate_after_150220
            date = date[:-cutoff]
            self.p.alpha = self.p.alpha[:-cutoff]

        self.p.alpha_weekdays = np.copy(self.p.alpha)
        self.p.weekdays = len(self.p.alpha_weekdays)
        self.p.alpha_weekdays = [x for i, x in enumerate(self.p.alpha_weekdays) if not (
            (i % 7 == 0) or (i % 7 == 1))]
        self.p.alpha_raw = None

        return date

    def _smooth_truncate_google_data(self, date):
        # smooth function:
        self.p.alpha_raw = self.p.alpha
        self.p.alpha = self._smooth_google_data()

        self._save_tanh_spline()
  
        # Truncate Google travel data to same length as daily deaths
        if self.mismatch < 0:
            date = date[:self.mismatch]
            self.p.alpha = self.p.alpha[:self.mismatch]
            self.p.alpha_raw = self.p.alpha_raw[:self.mismatch]

        print('Google first date: ' + str(date[self.p.day_1st_death_after_150220]) + ', last date: ' + str(date[-1]))
        assert date[-1] == self.last_date, "Dates don't match up"

        # Overwrite maxtime
        self.p.maxtime = len(date) + self.p.numeric_max_age + self.p.extra_days_to_simulate - 1

        return date

    def _save_tanh_spline(self):

        """ Save tanh spline fit for inspection """

        folder = os.path.join(MODULE_DIR if SPLINE_FITS is None else SPLINE_FITS, 'spline_fits')
        os.makedirs(folder, exist_ok=True)  # Create tanh spline fit output destination folder

        d_vec = np.linspace(0, self.lgoog_data - 1, self.lgoog_data)
        d_vec_weekdays = np.copy(d_vec)
        d_vec_weekdays = [x for i, x in enumerate(d_vec_weekdays) if not (
            (i % 7 == 0) or (i % 7 == 1))]

        str2 = self.country_str.replace(' ', '-')
        plt.figure(1e6)
        plt.title(self.country_str)
        plt.plot(d_vec, self.p.alpha_raw, 's', marker='o', label='Google')
        plt.plot(d_vec_weekdays, self.p.alpha_weekdays, 's', marker='o', label='Google without weekends')
        plt.plot(d_vec, self.p.alpha[:-(self.p.numeric_max_age + self.p.extra_days_to_simulate)], label='Smooth function')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel(r'$\alpha$')
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(folder, 'Spline-fit-' + str2 + '-transit-' + str(self.p.transit))
        plt.savefig(filename + '.png')
        plt.close()
    
    def _get_mismatch(self, ldate):

        '''Compute data mismatch'''

        return ldate + self.p.day_1st_death_after_150220 - self.lgoog_data

    def _smooth_google_data(self):

        '''Smooth Google data using tanh-like function'''

        print('Fitting spline to Google data...')
        opts = cma.CMAOptions()
        opts.set("seed", 100)
        opts.set("popsize", 50)
        CMA_stds = [1e-1, 1e-2, 10, 1e-1]
        lbounds, ubounds = [0, 0, 0, 0], [1, 0.1, 100, 1]
        self.x0 = [np.min(self.p.alpha), 1e-3, 15, 0.5]
        if self.p.five_param_spline:
            CMA_stds = np.concatenate((CMA_stds, [1e-1]))
            lbounds = np.concatenate((lbounds, [0]))
            ubounds = np.concatenate((ubounds, [1]))
            self.x0 = np.concatenate((self.x0, [0.5]))
        opts.set("CMA_stds", CMA_stds)
        opts.set("bounds", [lbounds, ubounds])
        es = cma.fmin(self._cma_min, self.x0, sigma0=1, options=opts)

        if self.p.five_param_spline:
            self.p.lockdown_baseline, self.p.lockdown_fatigue, self.p.lockdown_offset, self.p.lockdown_rate, \
                self.p.lockdown_new_normal = es[0]
        else:
            self.p.lockdown_baseline, self.p.lockdown_fatigue, self.p.lockdown_offset, self.p.lockdown_rate = es[0]

        param_dictionary = {'lockdown_baseline': self.p.lockdown_baseline, 'lockdown_fatigue': self.p.lockdown_fatigue,
                            'lockdown_offset': self.p.lockdown_offset, 'lockdown_rate': self.p.lockdown_rate}
        if self.p.five_param_spline:
            param_dictionary.update({'lockdown_new_normal': self.p.lockdown_new_normal})
        return tanh_spline(self.p, self.lgoog_data, param_dictionary)

    def _cma_min(self, params):
        param_dictionary = {'lockdown_baseline': params[0], 'lockdown_fatigue': params[1], 'lockdown_offset': params[2], \
            'lockdown_rate': params[3]}
        if self.p.five_param_spline:
            param_dictionary.update({'lockdown_new_normal': params[4]})
        return calculate_RMSE(tanh_spline(self.p, self.lgoog_data, param_dictionary, weekends=False)[:-(self.p.numeric_max_age + \
            self.p.extra_days_to_simulate)], self.p.alpha_weekdays)

