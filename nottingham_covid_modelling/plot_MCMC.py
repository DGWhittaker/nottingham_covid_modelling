import os

import matplotlib.pyplot as plt
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
from nottingham_covid_modelling.lib.equations import store_rate_vectors
import nottingham_covid_modelling.lib.priors as priors
from scipy.stats import gaussian_kde, uniform

big_text = True
if big_text:
    import matplotlib as mpl
    label_size = 24
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size 

def plot_mcmc():
    parser = get_parser()
    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
    parser.add_argument("--chain", type=int, help="which chain to use", default=1)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str, required=True, \
                        help="which parameters to optimise, e.g. -pto rho Iinit1 lockdown_baseline")
    parser.add_argument("--alpha1", action='store_true',
                        help="whether or not to do alpha=1 simulation", default=False)
    parser.add_argument("--show_priors", action='store_true',
                        help="whether or not to overlay priors", default=False)
    parser.add_argument("--dpi", help="DPI setting to use for custom figures",
                        default=100, type=int)

    args = parser.parse_args()
    if args.ons_data and args.country_str != 'United Kingdom':
        parser.error('Can only use ONS data in combination with country United Kingdom')

    country_str = args.country_str

    # Get parameters, p
    p = Params()
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

    filename = get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise)
    # filename = filename + '-alejandra-data'
    fit_synthetic_data = False
    if fit_synthetic_data:
        NB_phi = 0.1
        print('Fitting synthetic data with phi = ' + str(NB_phi))
        filename = filename + '-test-phi' + str(NB_phi)

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]

    # Parameter names
    if noise_model == Gauss_LogLikelihood and not p.fix_sigma:
        parameters_to_optimise.append('gaussian_noise_sigma')
    if noise_model == NegBinom_LogLikelihood and not p.fix_phi:
        parameters_to_optimise.append('negative_binomial_phi')

    print('Burn-in period = ' + str(args.burn_in) + ' iterations')

    # Show traces and histograms
    parameter_names = [r'$\rho$', r'$I_0$', r'$\alpha_b$', r'$t^{*}$', r'$\beta_{\mu}$', r'$\beta_{\sigma^2}$', r'$\zeta_{\mu}$', r'$\zeta_{\phi}$', r'$\delta$', r'$\phi$']
    # parameter_names = [r'$\rho$', r'$I_0$', r'$\alpha_b$', r'$t^{*}$', r'$\zeta_{\mu}$', r'$\zeta_{\phi}$', r'$\delta$', r'$\phi$']
    # parameter_names = [r'$\rho$', r'$I_0$', r'$\alpha_b$', r'$t^{*}$', r'$\beta_{\mu}$', r'$\beta_{\sigma^2}$', r'$\delta$', r'$\phi$']
    # parameter_names = [r'$\rho$', r'$I_0$', r'$\alpha_b$', r'$t^{*}$', r'$\delta$', r'$\phi$']
    pints.plot.trace(chains, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_chains.png')

    # Apply thinning
    chains = chains[:, ::10, :]

    # Look at distribution in chain (specified by args.chain)
    # xtrue = [3.20262554161788460e+00,
    #         8.59995741210512392e+02,
    #         2.81393754557459186e-01,
    #         3.15652518519784309e+01,
    #         0.1]
    # xtrue = [3.203,
    #         860,
    #         0.2814,
    #         31.57,
    #         5.2,
    #         2.96,
    #         18.69,
    #         0.0546,
    #         0.00724]#,
    #         #0.002
    #         #]
    xtrue = [3.203,
            860,
            0.2814,
            31.57,
            0.00724,
            0.002]
    pints.plot.pairwise(chains[args.chain - 1], kde=True, n_percentiles=99, parameter_names=parameter_names)#, ref_parameters=xtrue)
    if not args.show_plots:
        plt.savefig(saveas + '_pairwise_posteriors_chain' + str(args.chain) + '.png')

    # # Look at histograms
    pints.plot.histogram(chains, kde=True, n_percentiles=99, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_histograms.png')

    if args.show_priors:
        if big_text:
            import matplotlib as mpl
            label_size = 16
            mpl.rcParams['xtick.labelsize'] = label_size 
            mpl.rcParams['ytick.labelsize'] = label_size 
            mpl.rcParams['axes.labelsize'] = label_size 

        chains = chains[args.chain - 1]
        n_param = len(chains[0])
        LL = noise_model(p, data.daily_deaths, parameters_to_optimise)
        log_prior = priors.LogPrior(LL, np.max(data.daily_deaths))

        # Set up figure
        fig, axes = plt.subplots(n_param, 1, figsize=(6, 2 * n_param), dpi=args.dpi, squeeze=False)

        for i in range(n_param):
            if not p.flat_priors:
                if parameters_to_optimise[i] not in {'beta_mean', 'beta_var', 'IFR', 'lockdown_offset', \
                    'death_mean', 'death_dispersion'}:
                    lower, upper = log_prior.get_priors(parameters_to_optimise[i])
                    uniform_distribution = uniform(loc=lower, scale=upper-lower)
                    u = np.linspace(lower, upper, 100)
                    pdf = uniform_distribution.pdf(u)
                if parameters_to_optimise[i] in {'IFR', 'death_mean', 'death_dispersion', 'lockdown_offset'}:
                    u, pdf = log_prior.get_normal_prior(parameters_to_optimise[i])
                if parameters_to_optimise[i] in {'beta_mean', 'beta_var'}:
                    u, pdf = log_prior.get_gamma_prior(parameters_to_optimise[i])
            else:
                lower, upper = log_prior.get_priors(parameters_to_optimise[i])
                uniform_distribution = uniform(loc=lower, scale=upper-lower)
                u = np.linspace(lower, upper, 100)
                pdf = uniform_distribution.pdf(u)

            xmin, xmax = np.min(chains[:, i]), np.max(chains[:, i])
            x = np.linspace(xmin, xmax, 100)
            if parameters_to_optimise[i] == 'Iinit1':
                axes[i, 0].set_xlim([1, 5e4])
            if parameters_to_optimise[i] == 'IFR':
                axes[i, 0].set_xlim([0, 0.1])
            if parameters_to_optimise[i] == 'negative_binomial_phi':
                axes[i, 0].set_xlim([0, 0.01])
            axes[i, 0].set_xlabel(parameter_names[i])
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].plot(x, gaussian_kde(chains[:, i])(x), label='posterior')
            axes[i, 0].plot(u, pdf, label='prior')
            axes[i, 0].grid(True)
        axes[0, 0].legend()
        plt.tight_layout()
        if not args.show_plots:
            plt.savefig(saveas + '_priors.png')

    # Show graphs
    if args.show_plots:
        plt.show()
        