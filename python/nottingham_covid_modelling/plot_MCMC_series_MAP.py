import os

import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING, POPULATION, get_parser
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, tanh_spline, step, store_rate_vectors
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood, Poiss_LogLikelihood
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_effective, calculate_R_instantaneous
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from scipy.stats import nbinom, gamma
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_mcmc_series_map():
    parser = get_parser()
    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
    parser.add_argument("--chain", type=int, help="which chain to use", default=1)
    parser.add_argument("--show_Itot", action='store_true', help="whether to show Itot or not", default=False)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str, required=True, \
                        help="which parameters to optimise, e.g. -pto rho Iinit1 lockdown_baseline")
    parser.add_argument("--alpha1", action='store_true',
                        help="whether or not to do alpha=1 simulation", default=False)

    args = parser.parse_args()
    if args.ons_data and args.country_str != 'United Kingdom':
        parser.error('Can only use ONS data in combination with country United Kingdom')

    # Get parameters, p
    p = Params()
    if args.country_str in IFR_dict:
        p.IFR = IFR_dict[args.country_str]
    p.n_days_to_simulate_after_150220 = 150
    p.simple = args.simple
    p.fix_phi = args.fix_phi
    p.fixed_phi = args.fixed_phi
    p.fix_sigma = args.fix_sigma
    p.square_lockdown = args.square
    p.flat_priors = args.flat_priors
    if p.simple:
        print('Using simple rates...')
    else:
        print('Using gamma distribution rates...')

    # Simulate days after end of Google mobility data
    p.extra_days_to_simulate = 10

    # Get Google travel and deaths data
    print('Getting data...')
    data = DataLoader(args.ons_data, p, args.country_str, data_dir=args.datafolder)

    parameters_to_optimise = args.params_to_optimise

    # Get noise model
    noise_str = args.noise_model
    noise_model = NOISE_MODEL_MAPPING[noise_str]

    # alpha = 1 scenario
    p.alpha1 = args.alpha1
    if p.alpha1:
        assert p.square_lockdown == True, "Must use --square input for alpha=1 simulation"
        print('Using alpha = 1!!!')
        p.lockdown_baseline = 1.0

    # Get likelihood function
    LL = noise_model(p, data.daily_deaths, parameters_to_optimise)

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

    filename = get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise)
    # filename = filename + '-alejandra-data'
    fit_synthetic_data = False
    if fit_synthetic_data:
        NB_phi = 0.1
        print('Fitting synthetic data with phi = ' + str(NB_phi))
        filename = filename + '-test-phi' + str(NB_phi)

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)
    logpdfs = pints.io.load_samples(saveas + '-logpdf.csv', 3) # file contains log-posterior, log-likelihood, log-prior 

    chains = np.array(chains)
    logpdfs = np.array(logpdfs)

    niter = len(chains[1])

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]
    logpdfs = logpdfs[:, burn_in:, :]
    logpdfs = logpdfs[args.chain-1]
    MAP_idx = np.argmax(logpdfs[:, 0])
    MLE_idx = np.argmax(logpdfs[:, 1])
    print('Posterior mode log-posterior: ' + str(logpdfs[MAP_idx, 0]))
    print('Posterior mode log-likelihood: ' + str(logpdfs[MAP_idx, 1]))
    print('Best ever log-likelihood: ' + str(logpdfs[MLE_idx, 1]))
    print('Posterior mode log-prior: ' + str(logpdfs[MAP_idx, 2]))

    # Compare sampled posterior parameters with real data
    np.random.seed(100)

    upper = len(chains[1])

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(311)
    ax1.grid(True)
    ax2 = fig.add_subplot(312)
    ax2.grid(True)
    ax3 = fig.add_subplot(313)
    ax3.grid(True)
    axins = inset_axes(ax3, width=4, height=1.2)

    paras = chains[args.chain - 1][MAP_idx]
    p_dict = dict(zip(LL.parameter_labels, paras))
    # Calculate beta, gamma and zeta vector rates.
    store_rate_vectors(p_dict, p)
    S, I, R, D, Itot = solve_difference_equations(p, p_dict, travel_data=True)
    ax1.plot(t[:-p.numeric_max_age], D[:-p.numeric_max_age], color='blue', label='Posterior mode')
    if args.show_Itot:
        ax2.plot(t[:-p.numeric_max_age], Itot[:-p.numeric_max_age], color='blue', label='Daily active', alpha=0.5)
    ax2.plot(t[:-p.numeric_max_age], I[0, :-p.numeric_max_age], color='#ff7f0e')
    sigma1, sigma2, sigma3 = [], [], []
    NB_phi = p_dict.get('negative_binomial_phi', p.fixed_phi)
    sigma = p_dict.get('gaussian_noise_sigma', p.fixed_sigma)
    for k in D[:-p.numeric_max_age]:
        if noise_model == NegBinom_LogLikelihood:
            sigma1.append(np.sqrt(k + NB_phi * k**2))
            sigma2.append(2 * np.sqrt(k + NB_phi * k**2))
            sigma3.append(3 * np.sqrt(k + NB_phi * k**2))
        elif noise_model == Gauss_LogLikelihood:
            sigma1.append(sigma)
            sigma2.append(2 * sigma)
            sigma3.append(3 * sigma)
        elif noise_model == Poiss_LogLikelihood:
            sigma1.append(np.sqrt(k))
            sigma2.append(2 * np.sqrt(k))
            sigma3.append(3 * np.sqrt(k))
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
    paras = chains[args.chain - 1][MAP_idx]
    p_dict = dict(zip(LL.parameter_labels, paras))
    if p.square_lockdown:
        p.alpha = step(p, lgoog_data=l_alpha, parameters_dictionary=p_dict)[:-p.numeric_max_age]
    else:
        p.alpha = tanh_spline(p, lgoog_data=l_alpha, parameters_dictionary=p_dict)[:-p.numeric_max_age]
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

    if not args.show_plots:
        plt.savefig(saveas + '_series_chain' + str(args.chain) + '_MAP.png')

    if 'beta_mean' in parameters_to_optimise:
        beta_mean = p_dict['beta_mean']
        beta_var = p_dict['beta_var']

        scale = beta_var / beta_mean
        shape = beta_mean / scale

        kscale = p.beta_var / p.beta_mean
        kshape = p.beta_mean / kscale

        infect_samples = []
        bdays = np.linspace(0, 20, 21)
        for num in bdays:
            infect_samples.append(gamma.pdf(num, shape, loc=0, scale=scale))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Infectiousness profile')
        ax1.plot(bdays, infect_samples, color='dodgerblue', label='Inferred distribution')
        ax1.plot(bdays, gamma.pdf(bdays, kshape, loc=0, scale=kscale), color='black', linewidth=2, \
            linestyle='dashed', label='Default distribution')
        ax1.legend()
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Probability')
        ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        ax1.grid(True)
        plt.tight_layout()

        if not args.show_plots:
            plt.savefig(saveas + '_series_chain' + str(args.chain) + '_beta.png')

    if 'death_mean' in parameters_to_optimise:
        death_mean = p_dict['death_mean']
        death_dispersion = p_dict['death_dispersion']

        death_N_NB = 1 / death_dispersion
        death_p_NB = 1 / (1 + death_mean * death_dispersion)        

        kdeath_mean = p.death_mean
        kdeath_dispersion = p.death_dispersion
        kdeath_N_NB = 1 / kdeath_dispersion
        kdeath_p_NB = 1 / (1 + kdeath_mean * kdeath_dispersion)

        death_samples = []
        days = np.linspace(0, 120, 121)
        for num in days:
            death_samples.append(nbinom.pmf(num, death_N_NB, death_p_NB))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Infection-to-death distribution')
        ax1.plot(days, death_samples, color='red', label='Inferred distribution')
        ax1.plot(days, nbinom.pmf(days, kdeath_N_NB, kdeath_p_NB), color='black', linewidth=2, \
            linestyle='dashed', label='Default distribution')
        ax1.legend()
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Probability')
        ax1.grid(True)
        plt.tight_layout()

        if not args.show_plots:
            plt.savefig(saveas + '_series_chain' + str(args.chain) + '_death.png')

    # Show graphs
    if args.show_plots:
        plt.show()
