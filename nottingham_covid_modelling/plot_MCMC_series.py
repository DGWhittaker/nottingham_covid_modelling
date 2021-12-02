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
from scipy.stats import gaussian_kde


def plot_mcmc_series():
    parser = get_parser()
    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("-ns", "--number_samples", type=int, help="how many posterior samples to use", default=100)
    parser.add_argument("-std", "--standard_deviation", action='store_true',
                        help="whether to show +/- standard deviation on plots or not", default=False)
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

    nsamples = args.number_samples

    # Get parameters, p
    p = Params()
    if args.country_str in IFR_dict:
        p.IFR = IFR_dict[args.country_str]
    p.n_days_to_simulate_after_150220 = 150
    p.simple = args.simple
    p.square_lockdown = args.square
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

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]

    # Compare sampled posterior parameters with real data
    np.random.seed(100)

    upper = len(chains[1])
    posterior_samples = []

    print('Plotting ' + str(nsamples) + ' samples from chain ' + str(args.chain) + '...')
    label_added = False

    # fig = plt.figure(figsize=(10, 6))
    # ax1 = fig.add_subplot(211)
    # ax1.grid(True)
    # ax2 = fig.add_subplot(212)
    # ax2.grid(True)
    # ax3 = ax2.twinx()

    values = []
    R0_samples, Rmin_samples, Rl1_samples, maxI_samples = [], [], [], []
    l_alpha = len(p.alpha)

    chains = chains[:, ::10, :]
    file_append = 'half_Bayesian'

    # for i in chains[args.chain - 1]:
    #     paras = i
    #     p_dict = dict(zip(LL.parameter_labels, paras))
    #     # Calculate beta, gamma and zeta vector rates.
    #     store_rate_vectors(p_dict, p)
    #     S, I, R, D, Itot = solve_difference_equations(p, p_dict, travel_data=True)
    #     maxI_samples.append(np.max(I[0, :-p.numeric_max_age]))
    #     if p.square_lockdown:
    #         p.alpha = step(p, lgoog_data=l_alpha, parameters_dictionary=p_dict)[:-p.numeric_max_age]
    #     else:
    #         p.alpha = tanh_spline(p, lgoog_data=l_alpha, parameters_dictionary=p_dict)[:-p.numeric_max_age]
    #     R_eff = calculate_R_instantaneous(p, S, p_dict)
    #     R0_samples.append(R_eff[0])
    #     Rmin_samples.append(np.min(R_eff))
    #     for j, k in enumerate(R_eff):
    #         if k < 1:
    #             Rl1_samples.append(j)
    #             break

    # np.save('R0_samples_' + file_append + '.npy', R0_samples)
    # np.save('Rmin_samples_' + file_append + '.npy', Rmin_samples)
    # np.save('Rl1_samples_' + file_append + '.npy', Rl1_samples)
    # np.save('maxI_samples_' + file_append + '.npy', maxI_samples)

    R0 = np.load('../R0_samples_' + file_append + '.npy')
    Rmin = np.load('../Rmin_samples_' + file_append + '.npy')
    Rl1 = np.load('../Rl1_samples_' + file_append + '.npy')
    maxI = np.load('../maxI_samples_' + file_append + '.npy')

    R0_min, R0_max = np.min(R0), np.max(R0)
    R0x = np.linspace(R0_min, R0_max, 100)

    Rmin_min, Rmin_max = np.min(Rmin), np.max(Rmin)
    Rminx = np.linspace(Rmin_min, Rmin_max, 100)

    Rl1_min, Rl1_max = np.min(Rl1), np.max(Rl1)
    Rl1x = np.linspace(Rl1_min, Rl1_max, 6)

    maxI_min, maxI_max = np.min(maxI), np.max(maxI)
    maxIx = np.linspace(maxI_min, maxI_max, 100)

    fig = plt.figure(figsize=(8, 2))
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
    ax3.hist(Rl1, bins=2, density=True, color='red') # bins=9 for fully Bayesian, 2 for half Bayesian
    # ax3.plot(Rl1x, gaussian_kde(Rl1)(Rl1x))
    ax4 = fig.add_subplot(144)
    ax4.set_title(r'max$_i\{I_{i,1}\}$')
    ax4.hist(maxI, bins=25, density=True, color='red')
    ax4.plot(maxIx, gaussian_kde(maxI)(maxIx))
    plt.tight_layout()
    # plt.savefig('posterior_outputs_' + file_append + '.svg')
    plt.show()
