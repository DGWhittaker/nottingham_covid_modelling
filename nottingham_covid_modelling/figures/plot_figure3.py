import os
import shutil
import glob
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import numpy as np
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.likelihood import NegBinom_LogLikelihood
from nottingham_covid_modelling.lib.settings import Params
from scipy.stats import nbinom, gamma

# Get parameters, p
p = Params()
p.IFR = IFR_dict['United Kingdom']

p.extra_days_to_simulate = 10
p.n_days_to_simulate_after_150220 = 150

# Get Google travel and deaths data
print('Getting data...')
data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', '..', 'data', 'archive', 'current'))
shutil.rmtree('outcmaes')

# Get noise model
noise_str = 'NegBinom'
noise_model = NOISE_MODEL_MAPPING[noise_str]

# Time points (in days)
t = np.linspace(0, p.maxtime, p.maxtime + 1)
t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
    (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

non_herd_immunity_deaths = np.load('../non_herd_immunity_deaths.npy')
herd_immunity_deaths = np.load('../herd_immunity_deaths.npy')

fig = plt.figure(figsize=(6, 4), dpi=200)
ax1 = fig.add_subplot(111)

ax1.plot(t[:-p.numeric_max_age], herd_immunity_deaths[:-p.numeric_max_age], color='limegreen', label=r'constant $\alpha_i$')
ax1.plot(t[:-p.numeric_max_age], non_herd_immunity_deaths[:-p.numeric_max_age], color='dodgerblue', label=r'step change in $\alpha_i$')
ax1.scatter(t_daily, data.daily_deaths, edgecolor='red', facecolor='None', label='Observed data\n(' + data.country_display + ')')
ax1.legend()
ax1.set_ylabel('Daily deaths')
ax1.set_xlabel('Date')
ax1.set_xticks([x for x in (0, 40, 80, 120) if x < len(data.google_data)])
ax1.set_xticklabels([data.google_data[x] for x in (0, 40, 80, 120) if x < len(data.google_data)])
plt.grid(True)
plt.tight_layout()
plt.savefig('Figure4.png')

fig_path = os.path.join(MODULE_DIR, 'figures', 'saved-plots')
if not os.path.exists(fig_path):
   os.mkdir(fig_path)

files = glob.glob('./*.png')
for f in files:
    shutil.move(f, fig_path)
