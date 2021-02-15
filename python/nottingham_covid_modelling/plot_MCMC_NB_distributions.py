import os

import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING, POPULATION, get_parser
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.equations import store_rate_vectors, make_rate_vectors
from scipy.stats import nbinom, gamma


def plot_mcmc_nb_distributions():
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

    country_str = args.country_str
    nsamples = args.number_samples

    # Get parameters, p
    p = Params()
    p.n_days_to_simulate_after_150220 = 150
    p.simple = args.simple
    p.square_lockdown = args.square
    if p.simple:
        print('Using simple rates...')
    else:
        print('Using gamma distribution rates...')

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

    print('Plotting ' + str(nsamples) + ' samples from chain ' + str(args.chain) + '...')
    print('Burn-in period = ' + str(args.burn_in) + ' iterations')
    label_added = False

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(131)
    ax1.grid(True)
    ax2 = fig.add_subplot(132)
    ax2.grid(True)
    ax3 = fig.add_subplot(133)
    ax3.grid(True)

    values = []
    for i in range(nsamples):
        infect_samples, death_samples, recovery_samples = [], [], []
        death_hazards = []
        value = int(np.random.uniform(0, upper))
        values.append(value)
        paras = chains[args.chain - 1][value]
        p_dict = dict(zip(LL.parameter_labels, paras))
        beta_mean = p_dict.get('beta_mean', p.beta_mean)
        beta_var = p_dict.get('beta_var', p.beta_var)
        death_mean = p_dict.get('death_mean', p.death_mean)
        death_dispersion = p_dict.get('death_dispersion', p.death_dispersion)
        recovery_mean = p_dict.get('recovery_mean', p.recovery_mean)
        recovery_dispersion = p_dict.get('recovery_dispersion', p.recovery_dispersion)
        scale = beta_var / beta_mean
        shape = beta_mean / scale
        death_N_NB = 1 / death_dispersion
        death_p_NB = 1 / (1 + death_mean * death_dispersion)
        recovery_N_NB = 1 / recovery_dispersion
        recovery_p_NB = 1 / (1 + recovery_mean * recovery_dispersion)
        _, _, zeta = make_rate_vectors(p_dict, p)
        # print(zeta[-1])
        # print(zeta[0,:])
        for z in zeta[-1]:
            death_hazards.append(z)
        # death_hazards.append(zeta[0,:])
        bdays = np.linspace(0, 20, 81)
        for num in bdays:
            infect_samples.append(gamma.pdf(num, shape, loc=0, scale=scale))
        days = np.linspace(0, 120, 121)
        for num in days:
            death_samples.append(nbinom.pmf(num, death_N_NB, death_p_NB))
            recovery_samples.append(nbinom.pmf(num, recovery_N_NB, recovery_p_NB))

        ddays = np.linspace(0, 132, 133)
        if not label_added:
            ax1.plot(bdays, infect_samples, color='dodgerblue', alpha=0.5, label='MCMC posterior samples')
            ax2.plot(days, death_samples, color='red', alpha=0.5, label='MCMC posterior samples')
            ax3.plot(ddays, death_hazards, color='limegreen', alpha=0.5, label='MCMC posterior samples')
            label_added = True
        else:
            ax1.plot(bdays, infect_samples, color='dodgerblue', alpha=0.5)
            ax2.plot(days, death_samples, color='red', alpha=0.5)
            ax3.plot(ddays, death_hazards, color='limegreen', alpha=0.5)

    kscale = p.beta_var / p.beta_mean
    kshape = p.beta_mean / kscale

    kdeath_mean = p.death_mean
    kdeath_dispersion = p.death_dispersion
    kdeath_N_NB = 1 / kdeath_dispersion
    kdeath_p_NB = 1 / (1 + kdeath_mean * kdeath_dispersion)

    krecovery_mean = p.recovery_mean
    krecovery_dispersion = p.recovery_dispersion
    krecovery_N_NB = 1 / krecovery_dispersion
    krecovery_p_NB = 1 / (1 + krecovery_mean * krecovery_dispersion)

    ax1.plot(bdays, gamma.pdf(bdays, kshape, loc=0, scale=kscale), color='black', linewidth=2, \
        linestyle='dashed', label='Default distribution')
    ax2.plot(days, nbinom.pmf(days, kdeath_N_NB, kdeath_p_NB), color='black', linewidth=2, \
        linestyle='dashed', label='Default distribution')
    # ax3.plot(days, nbinom.pmf(days, krecovery_N_NB, krecovery_p_NB), color='black', linewidth=2, \
    #     linestyle='dashed', label='Default distribution')
    ax2.legend(loc='upper right')
    ax1.set_xlabel('Day')
    ax2.set_xlabel('Day')
    ax3.set_xlabel('Day')
    ax1.set_ylabel('Probability')
    ax1.set_title(str(nsamples) + ' samples, Infectiousness profile')
    ax2.set_title(str(nsamples) + ' samples, Infection-to-death distribution')
    ax3.set_title(str(nsamples) + ' samples, Death hazards')
    plt.tight_layout()

    if not args.show_plots:
        plt.savefig(saveas + '_NB_distributions_chain' + str(args.chain) + '.png')

    # Show graphs
    if args.show_plots:
        plt.show()
        