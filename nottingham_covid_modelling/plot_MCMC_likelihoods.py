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
import nottingham_covid_modelling.lib.priors as priors


def plot_mcmc_likelihoods():
    parser = get_parser()
    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("-std", "--standard_deviation", action='store_true',
                        help="whether to show +/- standard deviation on plots or not", default=False)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
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

    upper_sigma = np.max(data.daily_deaths)
    log_prior = priors.LogPrior(LL, upper_sigma)
    log_posterior = pints.LogPosterior(LL, log_prior)

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

    filename = get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise)
    # filename = filename + '-alejandra-data'

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]

    # Apply thinning
    chains = chains[:, ::1000, :]

    niter = len(chains[1])

    posterior_samples = []

    ten_percent = int(niter / 10)
    j = 0

    chain1_LL, chain2_LL, chain3_LL = [], [], []
    print('Burn-in period = ' + str(args.burn_in) + ' iterations')
    print('Calculating log-likelihoods... may take some time')
    for i in range(niter):
        chain1_LL.append(LL(chains[0][i]))
        chain2_LL.append(LL(chains[1][i]))
        chain3_LL.append(LL(chains[2][i]))

        # Display progress
        if (i + 1) % ten_percent == 0:
            j += 10
            print(str(j) + '% completed...')

    plt.figure(figsize=(10, 6))
    plt.xlabel('Iteration')
    plt.ylabel('log-likelihood')
    plt.plot(chain1_LL, alpha=0.75, label='Samples 1')
    plt.plot(chain2_LL, alpha=0.75, label='Samples 2')
    plt.plot(chain3_LL, alpha=0.75, label='Samples 3')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not args.show_plots:
        plt.savefig(saveas + '_likelihoods.png')

    # Show graphs
    if args.show_plots:
        plt.show()

