import os
import argparse

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import numpy as np
import pints
import pints.io
import pints.plot
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

def get_parameter_names(FitFull, FitStep, model_name):
    # Valid model_names: 'SIR', 'SIRDeltaD', 'SItD', 'SIUR' 
    assert model_name in ['SIR', 'SIRDeltaD', 'SItD', 'SIUR', 'SEIUR'], "Unknown model"
    parameter_names = [r'$\rho$', r'$I_0$']
    
    if FitFull:
        if model_name != 'SItD':
            parameter_names.extend([r'$\theta$'])
        if model_name == 'SIUR':
            parameter_names.extend([r'$\xi$'])
        if model_name == 'SEIUR':
            parameter_names.extend([r'$\eta$', r'$\xi$'])
        if model_name == 'SIRDeltaD':
            parameter_names.extend([r'$\Delta$D'])
    
    if FitStep:  
        parameter_names.extend([r'$\alpha_b$', r'$t^{*}$'])
    
    parameter_names.extend([r'$\phi$'])
    return parameter_names

def plot_mcmc():
    parser = argparse.ArgumentParser()

    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("--model_name", type=str, help="which model to use", choices=MODEL_FUNCTIONS.keys(), default='SIR')
    parser.add_argument("-full", "--fit_full", action='store_false', help='Whether to fit all the model parameters, or only [rho, I0, NB_phi], ', default=True)
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--syndata_num", type=int, help="Give the number of the synthetic data set you want to fit, default 1", default=1)
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
    parser.add_argument("--chain", type=int, help="which chain to use", default=1)

    # At the moment, syntethic data sets 2-9 have travel and step options only. There is only one data sets without step and one with neither travel nor step.

    args = parser.parse_args()
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
    p.flat_priors = True
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
    parameter_names = get_parameter_names(FitFull, FitStep, ModelName)

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

    filename = get_file_name_suffix(p, 'SimSItD-' + str(SyntDataNum_file) + rho_label, Noise_label + 'model-' + ModelName + \
        '_full-fit-' + str(FitFull), parameters_to_optimise)

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]

    print('Burn-in period = ' + str(args.burn_in) + ' iterations')

    pints.plot.trace(chains, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_chains.png')

    # Apply thinning
    chains = chains[:, ::10, :]

    pints.plot.pairwise(chains[args.chain - 1], kde=True, n_percentiles=99, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_pairwise_posteriors_chain' + str(args.chain) + '.png')

    # # Look at histograms
    pints.plot.histogram(chains, kde=True, n_percentiles=99, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_histograms.png')

    # Show graphs
    if args.show_plots:
        plt.show()

