import os
import argparse

import cma
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
import pints.io
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib.data import DataLoader
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix, get_file_name_suffix_anymodel, parameter_to_optimise_list


MODEL_FUNCTIONS ={'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, \
'SEIUR':get_model_SEIUR_solution}


def run_mcmc():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--niter", type=int, help='number of MCMC iterations', default=100000)
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits_SIR_realdata",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits_SIR_realdata'))
    parser.add_argument("--out_mcmc", type=str, help="folder to store mcmc runs files in, default: ./out-mcmc-SupTable2",
                        default= 'out-mcmc-SupTable2')
    parser.add_argument("--model_name", type=str, help="Which model to use", choices=MODEL_FUNCTIONS.keys(), default='SIR')
    parser.add_argument("-partial", "--fit_partial", action='store_true', help='Whether to fit a subset of the model  parameters (see \"-pto\"), ', default=False)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str,  \
                        help="If \"--fit_full\" is not given, select which parameters to optimise, e.g. -pto rho Iinit1 eta. This \
                        flag do not include the step parameters (see \"-fitstep\").\
                        OPTIONS FOR EACH MODEL ------ (1)SIR: rho Iinit1 theta; \
                        (2)SIRDeltaD: rho Iinit1 theta DeltaD; (3)SIUR: rho Iinit1 theta xi;  \
                        (4)SEIUR: rho Iinit1 theta eta xi; (5)SItD: rho Iinit1")
    parser.add_argument("--fixed_eta", type=float, help="value of eta. If eta will be fitted this value is ignored. Default value is the best fit for SEIUR model to clinical data:  1/2.974653", default = 0.3362)#0.1923,
    parser.add_argument("--fixed_theta", type=float, help="value of theta. If theta in fitted params, this value is ignored.  Default value is the best fit for SEIUR model to clinical data: 1/2.974653", default=0.3362)       
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--informative_priors", action='store_true', help='Whether to use informative priors', default=False)
    parser.add_argument("--chains5", action='store_true', help='Whether to do 5 instead of 3 chains', default=False)

    args = parser.parse_args()
    n_iter = args.niter
    FitFull = not args.fit_partial
    params_fromOptions = args.params_to_optimise
    FitStep = args.fit_step
    ModelName = args.model_name
    n_chains_5 = args.chains5
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

    # MCMC: 
    print('Setting up MCMC...')
    # Set up MCMC
    folder = args.cmaes_fits
    os.makedirs(folder, exist_ok=True)  # Create CMA-ES output destination folder
    if FitFull:
        filename1 = get_file_name_suffix(p, 'ONS-UK', 'model-' + ModelName + '_full-fit-' + str(FitFull), parameters_to_optimise)
    else:
        filename1 = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', ModelName , parameters_to_optimise)
    filename = filename1 + fixed_params_tag
    


    # Define output name and add label to use logposterior as IC
    if not args.informative_priors:
        p.flat_priors = True 
    else:
        filename = filename + '_logposterior'
    param_file = os.path.join(folder, filename)

    if FitFull:
        filename1 = get_file_name_suffix(p, 'ONS-UK', 'model-' + ModelName + '_full-fit-' + str(FitFull), parameters_to_optimise)
    else:
        filename1 = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', ModelName , parameters_to_optimise)
    filename = filename1 + fixed_params_tag

    # Get likelihood function
    model_func = MODEL_FUNCTIONS[ModelName]
    LL = noise_model(p, data_D, parameters_to_optimise, model_func=model_func)
    upper_sigma = np.max(data_D)
    log_prior = priors.LogPrior(LL, upper_sigma, model_name=ModelName)
    log_posterior = pints.LogPosterior(LL, log_prior)

    x0 = np.loadtxt(param_file + '.txt')
    print('MCMC starting point: ', x0)

    def perturb(x0):
        for i in range(1000):
            x0_2 = np.random.normal(1, 0.1, len(x0)) * x0
            x0_3 = np.random.normal(1, 0.1, len(x0)) * x0
            if np.isfinite(log_posterior(x0_2)) and np.isfinite(log_posterior(x0_3)):
                return x0_2, x0_3
        raise ValueError('Too many iterations')

    def perturb_5chains(x0):
        for i in range(1000):
            x0_2 = np.random.normal(1, 0.1, len(x0)) * x0
            x0_3 = np.random.normal(1, 0.1, len(x0)) * x0
            x0_4 = np.random.normal(1, 0.1, len(x0)) * x0
            if FitFull and ModelName == 'SEIUR':
                # to force one chain to start in the optima of where eta is one of the decent guess
                x0_5 = np.array([0.63, 2000, 0.085, 0.3125, 0.095, 0.088, 32.5, 0.0015])
                x0_5 = np.reshape(x0_5,np.shape(x0))
            else:
                x0_5 = np.random.normal(1, 0.1, len(x0)) * x0
            if np.isfinite(log_posterior(x0_2)) and np.isfinite(log_posterior(x0_3)) and np.isfinite(log_posterior(x0_4)) and np.isfinite(log_posterior(x0_5)):
                return x0_2, x0_3, x0_4, x0_5
        raise ValueError('Too many iterations')


    # Create 3 chains
    if n_chains_5:
        print('Doing 5 chains')
        x0_2, x0_3, x0_4, x0_5 = perturb_5chains(x0)
        x0_list = [x0, x0_2, x0_3, x0_4, x0_5]
    else:
        print('Doing 3 chains')
        x0_2, x0_3 = perturb(x0)
        x0_list = [x0, x0_2, x0_3]

    # Set up MCMC
    folder = os.path.join(MODULE_DIR, args.out_mcmc)
    os.makedirs(folder, exist_ok=True)  # Create MCMC output destination folder

    print('Selected data source: ONS UK data')
    print('Selected noise model: Negative Binomial')
    print('Extracted MLE parameters from: ' + param_file + '.txt')
    print('Number of iterations: ' + str(n_iter))

    # Run a simple adaptive MCMC routine
    mcmc = pints.MCMCController(log_posterior, len(x0_list), x0_list, method=pints.HaarioBardenetACMC) #HaarioBardenetACMC EmceeHammerMCMC PopulationMCMC RaoBlackwellACMC  DifferentialEvolutionMCMC DreamMCMC


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

