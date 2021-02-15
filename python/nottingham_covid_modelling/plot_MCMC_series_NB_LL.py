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
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_effective
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from scipy.stats import nbinom

def plot_mcmc_series_nb_ll():
    parser = get_parser()
    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("-ns", "--number_samples", type=int, help="how many posterior samples to use", default=100)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
    parser.add_argument("--chain", type=int, help="which chain to use", default=1)
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
    assert noise_model == NegBinom_LogLikelihood, "This script is for Neg Binom distribution only"

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

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax2 = fig.add_subplot(212)
    ax2.grid(True)
    # ax3 = ax2.twinx()

    values, S_vec = [], []
    fbest = -1e6
    for i in range(nsamples):
        value = int(np.random.uniform(0, upper))
        values.append(value)
        paras = chains[args.chain - 1][value]
        p_dict = dict(zip(LL.parameter_labels, paras))
        # Calculate beta, gamma and zeta vector rates.
        store_rate_vectors(p_dict, p)
        posterior_samples.append(paras)
        S, I, R, D, Itot = solve_difference_equations(p, p_dict, travel_data=True)

        f, fcum = [], 0
        fix_flag = False
        fix_num = 1e-12
        m = D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]
        y = data.daily_deaths
        for i in range(len(m)):
            mu = m[i]
            if mu < fix_num and y[i] > 0:
                mu = fix_num
                if not fix_flag and debug:
                    print('WARNING: Numerical fix activated.\nModel solution prevented from going below ' + \
                        str(fix_num) + ' to avoid infinite log-likelihood.')
                fix_flag = True
            n = 1 / p_dict['negative_binomial_phi']
            pr = 1 / (1 + mu * p_dict['negative_binomial_phi'])
            fcum = nbinom.logpmf(y[i], n, pr)
            f.append(fcum)

        if LL(paras) > fbest:
            best_ever_boys = f

        S_vec.append(S)
        if not label_added:
            ax1.plot(t[:-p.numeric_max_age], D[:-p.numeric_max_age], color='dodgerblue', alpha=0.5, label='MCMC posterior samples', zorder=-1)
            ax2.plot(t_daily, f, color='orange', alpha=0.5, label='MCMC posterior samples')
            label_added = True
        else:
            ax1.plot(t[:-p.numeric_max_age], D[:-p.numeric_max_age], color='dodgerblue', alpha=0.5, zorder=-1)
            ax2.plot(t_daily, f, color='orange', alpha=0.5)

    ax1.scatter(t_daily, data.daily_deaths, edgecolor='red', facecolor='None',
        label='Observed data (' + data.country_display + ')')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily deaths')
    ax1.set_title(str(nsamples) + ' samples - ' + noise_str + ' log-likelihood')
    ax1.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax1.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend(loc='best')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Log-likelihood')
    ax2.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax2.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    plt.tight_layout()

    if not args.show_plots:
        plt.savefig(saveas + '_series_NB_LL_chain' + str(args.chain) + '.png')

    # Show graphs
    if args.show_plots:
        plt.show()

    np.savetxt('Best_' + saveas + '_series_NB_LL_chain' + str(args.chain) + '.txt', best_ever_boys)
