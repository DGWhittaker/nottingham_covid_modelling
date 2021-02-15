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
from nottingham_covid_modelling.lib.equations import solve_difference_equations, step, store_rate_vectors
from nottingham_covid_modelling.lib.likelihood import NegBinom_LogLikelihood
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from scipy.stats import nbinom, gamma

# Get parameters, p
p = Params()
p.IFR = IFR_dict['United Kingdom']
p.square_lockdown = True

p.extra_days_to_simulate = 10
p.n_days_to_simulate_after_150220 = 150

# Get Google travel and deaths data
print('Getting data...')
data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', '..', 'data', 'archive', 'current'))
shutil.rmtree('outcmaes')

parameters_to_optimise = ['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_offset']

# Get noise model
noise_str = 'NegBinom'
noise_model = NOISE_MODEL_MAPPING[noise_str]

# Get likelihood function
LL = noise_model(p, data.daily_deaths, parameters_to_optimise)

# Time points (in days)
t = np.linspace(0, p.maxtime, p.maxtime + 1)
t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
    (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

# Set up optimisation
folder = os.path.join(MODULE_DIR, 'cmaes_fits')

filename = os.path.join(folder, get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise))
obtained_parameters = np.loadtxt(filename + '.txt')
p_dict = dict(zip(LL.parameter_labels, obtained_parameters))
p_dict['lockdown_baseline'] = 0.35

label = ''
for l in p_dict:
    label = label + str(l) + ': ' + str('%.4g' % p_dict.get(l)) + '\n'

# Calculate beta, gamma and zeta vector rates.
print('Storing fixed parameters...')
store_rate_vectors(p_dict, p)

# Simulate optimised model
cS, cI, cR, cD, cItot = solve_difference_equations(p, p_dict, travel_data=True)
fig = plt.figure(figsize=(10, 3), dpi=200)
ax1 = fig.add_subplot(131)
ax1.grid(True)

lgoog_data = len(p.alpha)
p.alpha = step(p, lgoog_data=lgoog_data, parameters_dictionary=p_dict)[:-p.numeric_max_age]

d_vec = np.linspace(0, p.weekdays - 1, p.weekdays)
d_vec_weekdays = np.copy(d_vec)
d_vec_weekdays = [x for i, x in enumerate(d_vec_weekdays) if not (
    (i % 7 == 0) or (i % 7 == 1))]

ax1.plot(d_vec[:-25], p.alpha[:-(p.numeric_max_age + p.extra_days_to_simulate + 35)], label='Step')
ax1.scatter(d_vec_weekdays[:-20], p.alpha_weekdays[:-20], edgecolor='orange', facecolor='None', \
    label='Google\nmobility data')
ax1.legend()
ax1.set_xticks([x for x in (0, 60, 120, 180) if x < len(data.google_data)])
ax1.set_xticklabels([data.google_data[x] for x in (0, 60, 120, 180) if x < len(data.google_data)])
ax1.set_ylabel(r'Relative mobility, $\alpha_j$')
ax1.set_xlabel(r'Date, $i$')

ax2 = fig.add_subplot(132)
ax2.grid(True)

days = np.linspace(0, 15, 16)

kbeta_mean = p.beta_mean
kbeta_var = p.beta_var
ksc = kbeta_var / kbeta_mean  # scale parameter
ka = kbeta_mean / ksc    # shape parameter

ax2.set_title('Infectiousness profile')
ax2.bar(days, gamma.pdf(days, ka, loc=0, scale=ksc), linewidth=2, color='green', alpha=0.6)
ax2.set_xlabel(r'Day, $j$')
ax2.set_ylabel(r'Probability, $\beta_j$')

ax3 = fig.add_subplot(133)
ax3.grid(True)

days = np.linspace(0, 50, 51)

kdeath_mean = p.death_mean
kdeath_dispersion = p.death_dispersion
kdeath_N_NB = 1 / kdeath_dispersion
kdeath_p_NB = 1 / (1 + kdeath_mean * kdeath_dispersion)

ax3.set_title('Infection-to-death distribution')
ax3.bar(days, nbinom.pmf(days, kdeath_N_NB, kdeath_p_NB), linewidth=2, color='red', alpha=0.6)
ax3.set_xlabel(r'Day, $j$')
ax3.set_ylabel(r'Probability, $\zeta_j$')
plt.tight_layout()
plt.savefig('Figure1.png')

fig_path = os.path.join(MODULE_DIR, 'figures', 'saved-plots')
if not os.path.exists(fig_path):
   os.mkdir(fig_path)

files = glob.glob('./*.png')
for f in files:
    shutil.move(f, fig_path)
