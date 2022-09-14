import os
import argparse

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib.data import DataLoader
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix_anymodel, get_file_name_suffix, parameter_to_optimise_list


MODEL_FUNCTIONS = {'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, 'SEIUR':get_model_SEIUR_solution}

def run_optimise():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--repeats", type=int, help="number of CMA-ES repeats", default=5)
    parser.add_argument("-d", "--detailed_output", action='store_true',
                        help="whether to output detailed information (CMA-ES logs and all repeat parameters) or not",
                        default=False)
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits_SIR_realdata",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits_SIR_realdata'))
    parser.add_argument("--limit_pints_iterations", type=int, default=None,
                        help=("limit pints to a maximum number of iterations. NOTE: this is mostly for debug and "
                              "testing purposes, you probably don't want to use this to get meaningful results!"))

    parser.add_argument("--model_name", type=str, help="which model to use", choices=MODEL_FUNCTIONS.keys(), default='SIR')
    parser.add_argument("-partial", "--fit_partial", action='store_true', help='Whether to fit a subset of the model  parameters (see \"-pto\"), ', default=False)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str,  \
                        help="If \"--partial\" is true, select which parameters to optimise, e.g. -pto rho Iinit1 eta. This \
                        flag do not include the step parameters (see \"-fitstep\").\
                        OPTIONS FOR EACH MODEL ------ (1)SIR: rho Iinit1 theta; \
                        (2)SIRDeltaD: rho Iinit1 theta DeltaD; (3)SIUR: rho Iinit1 theta xi;  \
                        (4)SEIUR: rho Iinit1 theta eta xi; (5)SItD: rho Iinit1")
    parser.add_argument("--fixed_eta", type=float, help="value of eta. If eta will be fitted this value is ignored. Default value is the best fit for SEIUR model to clinical data: 1/2.974653", default = 0.3362)#0.1923,
    parser.add_argument("--fixed_theta", type=float, help="value of theta. If theta in fitted params, this value is ignored.  Default value is the best fit for SEIUR model to clinical data: 1/2.974653", default=0.3362)    
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--informative_priors", action='store_true', help='Whether to use informative priors', default=False)

    # At the moment, syntethic data sets 2-9 have travel and step options only. There is only one data sets without step and one with neither travel nor step.

    args = parser.parse_args()
    repeats = args.repeats
    FitStep = args.fit_step
    ModelName = args.model_name
    max_iterations = args.limit_pints_iterations
    FitFull = not args.fit_partial
    params_fromOptions = args.params_to_optimise
    Fixed_eta = args.fixed_eta
    Fixed_theta = args.fixed_theta


    if FitFull:
        print("Fitting full parameters, any subset of paramertes will be ignored. \nIf you want to fit only some parameters change -partial and list -pto to fit ")
    else: 
        if params_fromOptions is None:
            parser.error("If -partial, -pto is required. You did not specify -pto.")

    # Define the extra tag for filenames when eta or theta parameters are defined to a fixed value
    fixed_params_tag = "" 
    if not FitFull and ModelName == 'SEIUR':
        if 'eta' not in  params_fromOptions:
            fixed_params_tag = fixed_params_tag + '_fixedEta-' + str(Fixed_eta)
    if not FitFull and ModelName != 'SItD':
        if 'theta' not in  params_fromOptions:
            fixed_params_tag = fixed_params_tag + '_fixedTheta-' + str(Fixed_theta)


    # For reproducibility:
    np.random.seed(100)


    # Get parameters, p
    p = Params()
    # Fixed for  UK google and ONS data:  
    p.IFR = 0.00724 # UK time
    p.n_days_to_simulate_after_150220 = 150
    p.five_param_spline = False
    p.N = 59.1e6

    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    p.square_lockdown = True
    #p.alpha = np.ones(p.maxtime)
    #p.lockdown_baseline = 0.2814 #0.2042884852266899
    #p.lockdown_offset = 31.57 #34.450147247864166

    

    # Storing parameter values. Default values are given by  best the fit from Kirsty
    if ModelName != 'SItD':
        p.theta = Fixed_theta
        p.eta = Fixed_eta

    if ModelName == 'SEIUR':
        p.xi = 1 / 11.744308
    else:
        p.xi = 1 / (18.69 - 1 - 5.2)

    if ModelName == 'SIRDeltaD':
        p.DeltaD = 18.69 - 1 - 5.2
    else:
        p.DeltaD = 0

    # define the params to optimize
    parameters_to_optimise = parameter_to_optimise_list(FitFull, FitStep, ModelName, params_fromOptions)
    print('Parameters to estimate:')
    print(parameters_to_optimise)
    
    # Get noise model
    noise_model = NOISE_MODEL_MAPPING['NegBinom']

    # Get real data
    print('Getting data...')
    data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', 'data', 'archive', 'current'))
    data_D = data.daily_deaths
    
    
    # to get the same data and fit lenghts as in Data_loader
    #p.maxtime = p.maxtime + p.numeric_max_age + p.extra_days_to_simulate #D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]
    #p.day_1st_death_after_150220 = 22

    # OPTIMISATION: 
    print('Starting optimization...')
    # Set up optimisation
    folder = args.cmaes_fits
    os.makedirs(folder, exist_ok=True)  # Create CMA-ES output destination folder
    if FitFull:
        filename1 = get_file_name_suffix(p, 'ONS-UK', 'model-' + ModelName + '_full-fit-' + str(FitFull), parameters_to_optimise)
    else:
        filename1 = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', ModelName , parameters_to_optimise)
    filename = filename1 + fixed_params_tag
    
    # Add tag if optimizing log posterior insted of likelihood
    if not args.informative_priors:
        p.flat_priors = True 
    else:
        filename = filename + '_logposterior'
    filename = os.path.join(folder, filename)  

    print('Selected data source: ONS UK data')
    print('Selected noise model: Negative Binomial')
    print('Storing results to: ' + filename + '.txt')
    
    # Get likelihood function
    model_func = MODEL_FUNCTIONS[ModelName]
    LL = noise_model(p, data_D , parameters_to_optimise, model_func = model_func)
    upper_sigma = np.max(data_D)
    log_prior = priors.LogPrior(LL, upper_sigma, model_name = ModelName)
    log_posterior = pints.LogPosterior(LL, log_prior)

    parameters, scores = [], []
    # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
    bounds = pints.RectangularBoundaries(log_prior.lower, log_prior.upper)
    # Repeat optimisation multiple times from different initial guesses and pick best
    for i in range(repeats):
        print('Repeat: ' + str(i + 1))
        # Random initial guesses from uniform priors
        #x0 = np.array([0.60, 3500, 0.375, 0.62, 23.0, 0.006])
        x0 = priors.get_good_starting_point(log_prior, LL, niterations=1000)
        # Create optimiser
        opt = pints.OptimisationController(log_posterior, x0, boundaries=bounds, method=pints.CMAES)
        opt.set_max_iterations(max_iterations)
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

    print('---- Summary ...')
    print('Best parameters: ')
    print(parameters[0])
    print('Best score:')
    print(-scores[0])

     # Extract best
    obtained_parameters = parameters[0]
    
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
    