import os
import argparse
from pickle import NONE, TRUE

import cma
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix


MODEL_FUNCIONS ={'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, 'SEIUR':get_model_SEIUR_solution}

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
    # parameters_to_optimise.extend(['negative_binomial_phi']) <- this one is added in the likelihood class
    if FitStep:  
        parameters_to_optimise.extend(['lockdown_baseline', 'lockdown_offset'])
    return parameters_to_optimise


def run_optimise():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--repeats", type=int, help="number of CMA-ES repeats", default=5)
    parser.add_argument("-d", "--detailed_output", action='store_true',
                        help="whether to output detailed information (CMA-ES logs and all repeat parameters) or not",
                        default=False)
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits_SIR",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits_SIR'))
    parser.add_argument("--limit_pints_iterations", type=int, default=None,
                        help=("limit pints to a maximum number of iterations. NOTE: this is mostly for debug and "
                              "testing purposes, you probably don't want to use this to get meaningful results!"))

    parser.add_argument("--model_name", type=str, help="which model to use", choices=MODEL_FUNCIONS.keys(), default='SIR')
    parser.add_argument("-full", "--fit_full", action='store_false', help='Whether to fit all the model parameters, or only [rho, I0, NB_phi], ', default=True)
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--syndata_num", type=int, help="Give the number of the synthetic data set you want to fit, default 1", default=1)

    # At the moment, syntethic data sets 2-9 have travel and step options only. There is only one data sets without step and one with neither travel nor step.



    args = parser.parse_args()
    repeats = args.repeats
    FitFull = args.fit_full
    FitStep = args.fit_step
    ModelName = args.model_name
    SyntDataNum_file = args.syndata_num
    max_iterations = args.limit_pints_iterations



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
    if SyntDataNum_file == 1: #Original default syntethic data
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
    
    # cut the data to the maxtime lenght:
    data_D = data_D[:p.maxtime+1]
    data_Dreal = data_Dreal[:p.maxtime+1]
    data_S_long = data_S
    data_S = data_S[:p.maxtime+1]
    data_Itot = data_Itot[:p.maxtime+1]
    data_R = data_R[:p.maxtime+1]
    data_I = data_I[:,:p.maxtime+1]
    
    # to get the same data and fit lenghts as in Data_loader
    p.maxtime = p.maxtime + p.numeric_max_age + p.extra_days_to_simulate #D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]
    p.day_1st_death_after_150220 = 22

    # OPTIMISATION: 
    print('Starting optimization...')
    # Set up optimisation
    folder = args.cmaes_fits
    os.makedirs(folder, exist_ok=True)  # Create CMA-ES output destination folder
    filename = os.path.join(folder, get_file_name_suffix(p, 'SimSItD-' + str(SyntDataNum_file) + rho_label, Noise_label + 'model-' + ModelName + '_full-fit-' + str(FitFull), parameters_to_optimise))

    
    print('Selected data source: ' + data_filename)
    print('Selected noise model: Negative Binomial')
    print('Storing results to: ' + filename + '.txt')


    # Get likelihood function
    model_func = MODEL_FUNCIONS[ModelName]
    LL = noise_model(p, data_D[p.day_1st_death_after_150220:] , parameters_to_optimise, model_func = model_func)
    upper_sigma = np.max(data_D)
    log_prior = priors.LogPrior(LL, upper_sigma, model_name = ModelName)
    parameters, scores = [], []
    # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
    bounds = pints.RectangularBoundaries(log_prior.lower, log_prior.upper)
    # Repeat optimisation multiple times from different initial guesses and pick best
    for i in range(repeats):
        print('Repeat: ' + str(i + 1))
        # Random initial guesses from uniform priors
        x0 = priors.get_good_starting_point(log_prior, LL, niterations=1000)
        # Create optimiser
        opt = pints.OptimisationController(LL, x0, boundaries=bounds, method=pints.CMAES)
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
