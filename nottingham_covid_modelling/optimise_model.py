import os

import matplotlib.pyplot as plt
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING, POPULATION, get_parser
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, tanh_spline, store_rate_vectors
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood, Poiss_LogLikelihood


def run_optimise():

    parser = get_parser()

    parser.add_argument("-r", "--repeats", type=int, help="number of CMA-ES repeats", default=5)
    parser.add_argument("-d", "--detailed_output", action='store_true',
                        help="whether to output detailed information (CMA-ES logs and all repeat parameters) or not",
                        default=False)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits'))
    parser.add_argument("--limit_pints_iterations", type=int, default=None,
                        help=("limit pints to a maximum number of iterations. NOTE: this is mostly for debug and "
                              "testing purposes, you probably don't want to use this to get meaningful results!"))
    parser.add_argument("--sample_from_priors", action='store_true',
                        help="whether to sample once from priors or not (default is to iterate 1000 times to get "
                             "good log-likelihood)", default=False)
    parser.add_argument("--simulate_full", action='store_true',
                        help="whether to use all Google data (default is 150 days)", default=False)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str, required=True, \
                        help="which parameters to optimise, e.g. -pto rho Iinit1 lockdown_baseline")
    parser.add_argument("--alpha1", action='store_true',
                        help="whether or not to do alpha=1 simulation", default=False)
    parser.add_argument("--optimise_likelihood", action='store_true',
                        help="whether to optimise log-likelihood instead of log-posterior", default=False)

    args = parser.parse_args()
    if args.ons_data and args.country_str != 'United Kingdom':
        parser.error('Can only use ONS data in combination with country United Kingdom')

    repeats = args.repeats

    # Get parameters, p
    p = Params()
    if args.country_str in IFR_dict:
        p.IFR = IFR_dict[args.country_str]
    if not args.simulate_full:
        p.n_days_to_simulate_after_150220 = 150
    p.simple = args.simple
    p.five_param_spline = False
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

    if p.fix_phi:
        assert noise_model == NegBinom_LogLikelihood, "fix_phi can only be used with NB noise model"
    if p.fix_sigma:
        assert noise_model == Gauss_LogLikelihood, "fix_sigma can only be used with Gaussian noise model"

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

    # Set up optimisation
    folder = args.cmaes_fits
    os.makedirs(folder, exist_ok=True)  # Create CMA-ES output destination folder

    filename = os.path.join(folder, get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise))
    # filename = filename + '-alejandra-data'
    print('Selected data source: ' + data.country_display)
    print('Selected noise model: ' + noise_str)
    print('Storing results to: ' + filename + '.txt')
    optimise_likelihood = args.optimise_likelihood
    if optimise_likelihood:
        print('WARNING: optimising log-likelihood, not MAP!!!')
        filename = filename + '-optimiseLL'
    fit_synthetic_data = False
    if fit_synthetic_data:
        NB_phi = 0.001
        print('Fitting synthetic data with phi = ' + str(NB_phi))
        filename = filename + '-test-phi' + str(NB_phi)

    # Fix random seed for reproducibility
    np.random.seed(100)

    parameters, scores = [], []

    # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
    bounds = pints.RectangularBoundaries(log_prior.lower, log_prior.upper)

    # Repeat optimisation multiple times from different initial guesses and pick best
    for i in range(repeats):
        print('Repeat: ' + str(i + 1))
        # Random initial guesses from uniform priors
        if args.sample_from_priors:
            x0 = log_prior.sample()
        else:
            x0 = priors.get_good_starting_point(log_prior, log_posterior, niterations=1000)

        # Create optimiser
        if optimise_likelihood:
            opt = pints.OptimisationController(
                LL, x0, boundaries=bounds, method=pints.CMAES)
        else:
            opt = pints.OptimisationController(
                log_posterior, x0, boundaries=bounds, method=pints.CMAES)            
        if args.detailed_output:
            opt.set_log_to_file(filename + '-log-' + str(i) + '.txt')
        opt.set_max_iterations(args.limit_pints_iterations)
        opt.set_parallel(True)

        # Run optimisation
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            xbest, fbest = opt.run()
            parameters.append(xbest)
            scores.append(-fbest)

    # Sort according to smallest function score
    order = np.argsort(scores)
    scores = np.asarray(scores)[order]
    parameters = np.asarray(parameters)[order]

    # Show results
    print('Best parameters:')
    print(parameters[0])
    print('Best score:')
    print(-scores[0])

    # Extract best
    obtained_parameters = parameters[0]
    p_dict = dict(zip(LL.parameter_labels, obtained_parameters))

    # Store results
    print('Storing best result...')
    with open(filename + '.txt', 'w') as f:
        for x in obtained_parameters:
            f.write(pints.strfloat(x) + '\n')

    print('Storing all errors...')
    with open(filename + '-errors.txt', 'w') as f:
        for score in scores:
            f.write(pints.strfloat(-score) + '\n')

    if args.detailed_output:
        print('Storing all parameters...')
        for i, param in enumerate(parameters):
            with open(filename + '-parameters-' + str(1 + i) + '.txt', 'w') as f:
                for x in param:
                    f.write(pints.strfloat(x) + '\n')
