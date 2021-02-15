import os
import shutil
import glob
import matplotlib.pyplot as plt
from nottingham_covid_modelling import MODULE_DIR

from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.likelihood import NegBinom_LogLikelihood
# Load figure plotting functions
from nottingham_covid_modelling.figures.figure5A import plot_figure5A
from nottingham_covid_modelling.figures.figure5BCD import plot_figure5BCD
from nottingham_covid_modelling.figures.figure5E import plot_figure5E

# Get parameters, p
p = Params()
p.n_days_to_simulate_after_150220 = 150
p.extra_days_to_simulate = 10
p.square_lockdown = True
p.IFR = IFR_dict['United Kingdom']

# Get Google travel and deaths data
print('Getting data...')
data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', '..', 'data', 'archive', 'current'))
shutil.rmtree('outcmaes')

parameters_to_optimise = ['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_offset', 'IFR', 'negative_binomial_phi']

# Get noise model
noise_str = 'NegBinom'
noise_model = NOISE_MODEL_MAPPING[noise_str]

filename = get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise)

plot_figure5A(p, data, filename)
plot_figure5BCD(p, data, filename, parameters_to_optimise)
plot_figure5E(p, data, filename, parameters_to_optimise)

fig_path = os.path.join(MODULE_DIR, 'figures', 'saved-plots')
if not os.path.exists(fig_path):
   os.mkdir(fig_path)

files = glob.glob('./*.png')
for f in files:
    shutil.move(f, fig_path)
