import os

import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.likelihood import NegBinom_LogLikelihood
from scipy.stats import gaussian_kde

def plot_figure5E(p, data, filename, parameters_to_optimise, plot=False):
    noise_model = NOISE_MODEL_MAPPING['NegBinom']

    # Get likelihood function
    LL = noise_model(p, data.daily_deaths, parameters_to_optimise)

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = 25000
    chains = chains[:, burn_in:, :]

    values = []
    R0_samples, Rmin_samples, Rl1_samples, maxI_samples = [], [], [], []
    l_alpha = len(p.alpha)

    chains = chains[:, ::10, :]
    file_append = 'half_Bayesian'

    R0 = np.load('../../R0_samples_' + file_append + '.npy')
    Rmin = np.load('../../Rmin_samples_' + file_append + '.npy')
    Rl1 = np.load('../../Rl1_samples_' + file_append + '.npy')
    maxI = np.load('../../maxI_samples_' + file_append + '.npy')

    R0_min, R0_max = np.min(R0), np.max(R0)
    R0x = np.linspace(R0_min, R0_max, 100)

    Rmin_min, Rmin_max = np.min(Rmin), np.max(Rmin)
    Rminx = np.linspace(Rmin_min, Rmin_max, 100)

    Rl1_min, Rl1_max = np.min(Rl1), np.max(Rl1)
    Rl1x = np.linspace(Rl1_min, Rl1_max, 6)

    maxI_min, maxI_max = np.min(maxI), np.max(maxI)
    maxIx = np.linspace(maxI_min, maxI_max, 100)

    fig = plt.figure(figsize=(8, 2), dpi=200)
    ax1 = fig.add_subplot(141)
    ax1.set_title(r'$\mathcal{R}_0$')
    ax1.hist(R0, bins=25, density=True, color='red')
    ax1.plot(R0x, gaussian_kde(R0)(R0x))
    ax2 = fig.add_subplot(142)
    ax2.set_title(r'min$_i\{\mathcal{R}_i\}$')
    ax2.hist(Rmin, bins=25, density=True, color='red')
    ax2.plot(Rminx, gaussian_kde(Rmin)(Rminx))
    ax3 = fig.add_subplot(143)
    ax3.set_title(r'argmin$_i\{\mathcal{R}_i<1\}$')
    ax3.hist(Rl1, bins=2, density=True, color='red')
    ax4 = fig.add_subplot(144)
    ax4.set_title(r'max$_i\{I_{i,1}\}$')
    ax4.hist(maxI, bins=25, density=True, color='red')
    ax4.plot(maxIx, gaussian_kde(maxI)(maxIx))
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        plt.savefig('Figure5E.png')
