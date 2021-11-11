import os

import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
import pints.io
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING, POPULATION, get_parser
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.equations import store_rate_vectors

def run_mcmc():
    parser = get_parser()
    parser.add_argument("-n", "--niter", type=int, help='number of MCMC iterations', default=100000)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits'))
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str, required=True, \
                        help="which parameters to optimise, e.g. -pto rho Iinit1 lockdown_baseline")
    parser.add_argument("--alpha1", action='store_true',
                        help="whether or not to do alpha=1 simulation", default=False)

    args = parser.parse_args()
    if args.ons_data and args.country_str != 'United Kingdom':
        parser.error('Can only use ONS data in combination with country United Kingdom')

    n_iter = args.niter

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

    # Get Google travel and deaths data
    print('Getting data...')
    data = DataLoader(args.ons_data, p, args.country_str, data_dir=args.datafolder)

    parameters_to_optimise = args.params_to_optimise
    if 'lockdown_new_normal' in parameters_to_optimise and not p.five_param_spline:
        raise ValueError("Cannot optimise lockdown_new_normal with 4 parameter spline")

    if p.square_lockdown:
        for pto in parameters_to_optimise:
            if pto in {'lockdown_fatigue', 'lockdown_new_normal', 'lockdown_rate'}:
                raise ValueError("Cannot optimise these parameters using square lockdown function")

    for gdp in p.gamma_dist_params:
        if gdp in parameters_to_optimise:
            p.calculate_rate_vectors = True
            break

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

    folder = args.cmaes_fits
    os.makedirs(folder, exist_ok=True)

    param_file = os.path.join(folder, get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise))
    # param_file = param_file + '-alejandra-data'
    fit_synthetic_data = False
    if fit_synthetic_data:
        NB_phi = 0.1
        print('Fitting synthetic data with phi = ' + str(NB_phi))
        param_file = param_file + '-test-phi' + str(NB_phi)
    x0 = np.loadtxt(param_file + '.txt')
    print('MCMC starting point: ', x0)

    filename = get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise)
    if fit_synthetic_data:
        filename = filename + '-test-phi' + str(NB_phi)
    # filename = filename + '-alejandra-data'

    def perturb(x0):
        for i in range(1000):
            x0_2 = np.random.normal(1, 0.1, len(x0)) * x0
            x0_3 = np.random.normal(1, 0.1, len(x0)) * x0
            if np.isfinite(log_posterior(x0_2)) and np.isfinite(log_posterior(x0_3)):
                return x0_2, x0_3
        raise ValueError('Too many iterations')

    # Create 3 chains
    x0_2, x0_3 = perturb(x0)
    x0_list = [x0, x0_2, x0_3]

    # Set up MCMC
    folder = os.path.join(MODULE_DIR, 'out-mcmc')
    os.makedirs(folder, exist_ok=True)  # Create MCMC output destination folder

    print('Selected data source: ' + data.country_display)
    print('Selected noise model: ' + args.noise_model)
    print('Number of iterations: ' + str(n_iter))

    # Run a simple adaptive MCMC routine
    mcmc = pints.MCMCController(log_posterior, len(x0_list), x0_list, method=pints.HaarioBardenetACMC)
    # Enable logging to screen
    mcmc.set_log_to_screen(True)
    # Add stopping criterion
    mcmc.set_max_iterations(n_iter)
    mcmc.set_parallel(True)
    saveas = os.path.join(MODULE_DIR, folder, filename)
    mcmc.set_log_pdf_filename(saveas + '-logpdf.csv')
    # Run MCMC
    chains = mcmc.run()

    # Save results
    pints.io.save_samples(saveas + '-chain.csv', *chains)
