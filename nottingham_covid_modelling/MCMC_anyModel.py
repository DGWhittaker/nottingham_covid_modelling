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
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix


MODEL_FUNCTIONS ={'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, \
'SEIUR':get_model_SEIUR_solution}

# Functions
def parameter_to_optimise_list(FitFull, FitStep, model_name):
    # Valid model_names: 'SIR', 'SIRDeltaD', 'SItD', 'SIUR' 
    assert model_name in ['SIR', 'SIRDeltaD', 'SItD', 'SIUR', 'SEIUR'], "Unknown model"
    parameters_to_optimise = ['rho', 'Iinit1']

    if FitFull:
        if model_name != 'SItD':
            parameters_to_optimise.extend(['theta'])
        if model_name == 'SIUR':
            parameters_to_optimise.extend(['xi'])
        if model_name == 'SEIUR':
            parameters_to_optimise.extend(['eta'])
            parameters_to_optimise.extend(['xi'])
        if model_name == 'SIRDeltaD':
           parameters_to_optimise.extend(['DeltaD'])

    if FitStep:  
        parameters_to_optimise.extend(['lockdown_baseline', 'lockdown_offset'])

    return parameters_to_optimise


def run_mcmc():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--niter", type=int, help='number of MCMC iterations', default=100000)
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits_SIR",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits_SIR'))
    parser.add_argument("--model_name", type=str, help="which model to use", choices=MODEL_FUNCTIONS.keys(), default='SIR')
    parser.add_argument("-full", "--fit_full", action='store_false', help='Whether to fit all the model parameters, or only [rho, I0, NB_phi], ', default=True)
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--syndata_num", type=int, help="Give the number of the synthetic data set you want to fit, default 1", default=1)
    parser.add_argument("--informative_priors", action='store_true', help='Whether to use informative priors', default=False)

    # At the moment, syntethic data sets 2-9 have travel and step options only. There is only one data sets without step and one with neither travel nor step.

    args = parser.parse_args()
    n_iter = args.niter
    FitFull = args.fit_full
    FitStep = args.fit_step
    ModelName = args.model_name
    SyntDataNum_file = args.syndata_num

    # For reproducibility:
    np.random.seed(100)

    # Number of days to fit
    maxtime_fit = 150

    # Get parameters, p
    p = Params()
    # Fixed for the synth data, based on UK google and ONS data:  
    p.N = 59.1e6
    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    p.IFR = 0.00724 # UK time
    p.square_lockdown = True
    p.alpha = np.ones(p.maxtime)
    p.lockdown_baseline = 0.2814 #0.2042884852266899
    p.lockdown_offset = 31.57 #34.450147247864166
    # For saving file names:
    rho_label = '_rho_0-2'  
    Noise_label = 'NBphi_2e-3_' 

    # Storing values in the models so there is no error after (since there is no store_params for the simple models)
    if ModelName != 'SItD':
        p.beta = 1
        p.theta = 1 / p.beta_mean
        p.eta = 1 / p.beta_mean
        p.DeltaD = 0
        p.xi = 1 / (p.death_mean -p.beta_mean)

    # define the params to optimize
    parameters_to_optimise = parameter_to_optimise_list(FitFull, FitStep, ModelName)
    
    # Get noise model
    noise_model = NOISE_MODEL_MAPPING['NegBinom']

    # Get simulated Age data from file
    print('Getting simulated data...')

    # folder to load data
    if SyntDataNum_file == 1: # Original default syntethic data
        folder_path =  os.path.join(MODULE_DIR, 'out_SIRvsAGEfits')
        full_fit_data_file = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy'
    else:
        folder_path =  os.path.join(MODULE_DIR, 'out_SIRvsAGE_SuplementaryFig')
        full_fit_data_file = 'SynteticSItD_default_params_travel_TRUE_step_TRUE_' + str(SyntDataNum_file) + '.npy'
    data_filename = full_fit_data_file
    
    # Load data
    data = np.load(os.path.join(folder_path, data_filename ))
    data_S = data[:,0]
    data_Itot = data[:,1]
    data_R = data[:,2]
    data_Dreal = data[:,3]
    data_D = data[:,4] # noise data
    data_I = data[:,5:].T # transpose to get exactly the same shape as other code
    
    if len(data_R) < maxtime_fit:
        p.maxtime = len(data_R) -1
        maxtime_fit = len(data_R) -1
    else:
        p.maxtime = maxtime_fit
    
    # cut the data to the maxtime length:
    data_D = data_D[:p.maxtime+1]
    data_Dreal = data_Dreal[:p.maxtime+1]
    data_S_long = data_S
    data_S = data_S[:p.maxtime+1]
    data_Itot = data_Itot[:p.maxtime+1]
    data_R = data_R[:p.maxtime+1]
    data_I = data_I[:,:p.maxtime+1]
    
    # to get the same data and fit lengths as in Data_loader
    p.maxtime = p.maxtime + p.numeric_max_age + p.extra_days_to_simulate
    p.day_1st_death_after_150220 = 22

    # MCMC: 
    print('Setting up MCMC...')
    # Set up MCMC
    folder = args.cmaes_fits
    os.makedirs(folder, exist_ok=True)  # Create CMA-ES output destination folder
    param_file = os.path.join(folder, get_file_name_suffix(p, 'SimSItD-' + str(SyntDataNum_file) + rho_label, Noise_label + \
        'model-' + ModelName + '_full-fit-' + str(FitFull), parameters_to_optimise))

    # Get likelihood function
    model_func = MODEL_FUNCTIONS[ModelName]
    if not args.informative_priors:
        p.flat_priors = True # This line is crucial, to avoid the normal prior on the lockdown offset
    LL = noise_model(p, data_D[p.day_1st_death_after_150220:], parameters_to_optimise, model_func=model_func)
    upper_sigma = np.max(data_D)
    log_prior = priors.LogPrior(LL, upper_sigma, model_name=ModelName)
    log_posterior = pints.LogPosterior(LL, log_prior)

    x0 = np.loadtxt(param_file + '.txt')
    print('MCMC starting point: ', x0)

    filename = get_file_name_suffix(p, 'SimSItD-' + str(SyntDataNum_file) + rho_label, Noise_label + 'model-' + ModelName + \
        '_full-fit-' + str(FitFull), parameters_to_optimise)

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

    print('Selected data source: ' + data_filename)
    print('Selected noise model: Negative Binomial')
    print('Extracted MLE parameters from: ' + param_file + '.txt')
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
