import os

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.equations import solve_difference_equations, tanh_spline, step, store_rate_vectors
from nottingham_covid_modelling.lib.likelihood import NegBinom_LogLikelihood
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous
from scipy.stats import nbinom, gamma
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_figure6BCD(p, data, filename, parameters_to_optimise, plot=False):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    noise_model = NOISE_MODEL_MAPPING['NegBinom']

    # Get likelihood function
    LL = noise_model(p, data.daily_deaths, parameters_to_optimise)

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)
    logpdfs = pints.io.load_samples(saveas + '-logpdf.csv', 3) # file contains log-posterior, log-likelihood, log-prior 

    chains = np.array(chains)
    logpdfs = np.array(logpdfs)

    niter = len(chains[1])

    # Discard burn in
    burn_in = 50000
    chains = chains[:, burn_in:, :]
    logpdfs = logpdfs[:, burn_in:, :]
    logpdfs = logpdfs[0]
    MAP_idx = np.argmax(logpdfs[:, 0])
    MLE_idx = np.argmax(logpdfs[:, 1])
    print('Posterior mode log-posterior: ' + str(logpdfs[MAP_idx, 0]))
    print('Posterior mode log-likelihood: ' + str(logpdfs[MAP_idx, 1]))
    print('Best ever log-likelihood: ' + str(logpdfs[MLE_idx, 1]))
    print('Posterior mode log-prior: ' + str(logpdfs[MAP_idx, 2]))

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(311)
    ax1.grid(True)
    ax2 = fig.add_subplot(312)
    ax2.grid(True)
    ax3 = fig.add_subplot(313)
    ax3.grid(True)
    axins = inset_axes(ax3, width=4, height=1.2)

    paras = chains[0][MAP_idx]
    p_dict = dict(zip(LL.parameter_labels, paras))
    # Calculate beta, gamma and zeta vector rates.
    store_rate_vectors(p_dict, p)
    S, I, R, D, Itot = solve_difference_equations(p, p_dict, travel_data=True)
    ax1.plot(t[:-p.numeric_max_age], D[:-p.numeric_max_age], color='blue', label='Posterior mode')
    ax2.plot(t[:-p.numeric_max_age], I[0, :-p.numeric_max_age], color='#ff7f0e')
    sigma1, sigma2, sigma3 = [], [], []
    NB_phi = p_dict.get('negative_binomial_phi', p.fixed_phi)
    for k in D[:-p.numeric_max_age]:
        sigma1.append(np.sqrt(k + NB_phi * k**2))
        sigma2.append(2 * np.sqrt(k + NB_phi * k**2))
        sigma3.append(3 * np.sqrt(k + NB_phi * k**2))
    ax1.fill_between(t[:-p.numeric_max_age], D[:-p.numeric_max_age] - sigma3, D[:-p.numeric_max_age] \
        + sigma3, color='dodgerblue', alpha=0.25)
    ax1.fill_between(t[:-p.numeric_max_age], D[:-p.numeric_max_age] - sigma2, D[:-p.numeric_max_age] \
        + sigma2, color='dodgerblue', alpha=0.5)
    ax1.fill_between(t[:-p.numeric_max_age], D[:-p.numeric_max_age] - sigma1, D[:-p.numeric_max_age] \
        + sigma1, color='dodgerblue', alpha=0.75)

    ax1.scatter(t_daily, data.daily_deaths, edgecolor='red', facecolor='None',
        label='Observed data (' + data.country_display + ')')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Number')
    ax1.set_title('Daily deaths')
    ax1.set_xticks([x for x in (0, 40, 80, 120) if x < len(data.google_data)])
    ax1.set_xticklabels([data.google_data[x] for x in (0, 40, 80, 120) if x < len(data.google_data)])
    ax2.set_title('Daily new infections')
    ax2.set_ylabel('Number')
    ax2.set_xticks([x for x in (0, 40, 80, 120) if x < len(data.google_data)])
    ax2.set_xticklabels([data.google_data[x] for x in (0, 40, 80, 120) if x < len(data.google_data)])

    d_vec = np.linspace(0, p.weekdays - 1, p.weekdays)
    d_vec_weekdays = np.copy(d_vec)
    d_vec_weekdays = [x for i, x in enumerate(d_vec_weekdays) if not (
        (i % 7 == 0) or (i % 7 == 1))]

    l_alpha = len(p.alpha)
    paras = chains[0][MAP_idx]
    p_dict = dict(zip(LL.parameter_labels, paras))
    p.alpha = step(p, lgoog_data=l_alpha, parameters_dictionary=p_dict)[:-p.numeric_max_age]
    R_eff = calculate_R_instantaneous(p, S, p_dict)
    ax3.plot(R_eff, color='red', label='Posterior mode')
    ax3.set_title(r'$\mathcal{R}$')
    ax3.set_xlabel('Date')
    ax3.set_xticks([x for x in (0, 40, 80, 120) if x < len(data.google_data)])
    ax3.set_xticklabels([data.google_data[x] for x in (0, 40, 80, 120) if x < len(data.google_data)])
    axins.plot(p.alpha[:-(p.numeric_max_age + p.extra_days_to_simulate)], color='blue', label=r'$\alpha$')
    axins.scatter(d_vec_weekdays, p.alpha_weekdays, edgecolor='#ff7f0e', facecolor='None', \
        label='Google mobility data')
    axins.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    axins.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    axins.set_xlabel('Date')
    axins.grid(True)
    axins.legend()

    plt.tight_layout()

    # Show graphs
    if plot:
        plt.show()
    else:
        plt.savefig('Figure6BCD.png')
