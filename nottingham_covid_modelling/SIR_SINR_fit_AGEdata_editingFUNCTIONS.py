import os
import argparse
from pickle import NONE

import cma
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING, POPULATION
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SIUR_difference_equations, step, get_model_SIUR_solution, get_model_solution, get_model_SIR_solution
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous


neg_bin_fix_mu = 1e-12 

# Functions
def parameter_to_optimize_list(FitFull, FitStep, model_name):
    # Valid model_names: 'SIR', 'SIRDeltaD', 'SItD', 'SIUR' 
    assert model_name in ['SIR', 'SIRDeltaD', 'SItD', 'SIUR'], "Unknown model"
    parameters_to_optimise = ['rho', 'Iinit1']
    if FitFull:
        if model_name != 'SItD':
            parameters_to_optimise.extend(['theta'])
        if model_name == 'SIUR':
            parameters_to_optimise.extend(['xi'])
        if model_name == 'SIRDeltaD':
           parameters_to_optimise.extend(['DeltaD'])
    parameters_to_optimise.extend(['negative_binomial_phi'])
    if FitStep:  
        parameters_to_optimise.extend(['lockdown_baseline', 'lockdown_offset'])
    return parameters_to_optimise


def par_bounds(parameters_to_optimize):
    lower = []
    upper = []
    stds = []
    rho_lower, rho_upper, rho_std = 0, 5, 1e-2
    # rho_SIR_lower, rho_SIR_upper = 0, 2
    # rho_lower, rho_upper = 1, 5
    Iinit1_lower, Iinit1_upper, Iinit1_std  = 1, 5e4, 100
    theta_lower, theta_upper, theta_std = 0, 1, 1e-2
    DeltaD_lower, DeltaD_upper, DeltaD_std = 0, 50, 10
    xi_lower, xi_upper, xi_std = 0, 1, 1e-2
    negative_binomial_phi_lower, negative_binomial_phi_upper, negative_binomial_phi_std = 0, 1, 1e-2
    lockdown_baseline_lower, lockdown_baseline_upper, lockdown_baseline_std = 0, 1, 1e-2
    lockdown_offset_lower, lockdown_offset_upper, lockdown_offset_std = 0, 100, 10
    for parameter_name in parameters_to_optimize:
        lower = np.append(lower, eval(parameter_name + '_lower'))
        upper = np.append(upper,eval(parameter_name + '_upper'))            
        stds = np.append(stds, eval(parameter_name + '_std'))
    bounds = np.stack((lower, upper)).tolist()
    return bounds, stds

def NBneglogL_models(params, p, parameters_dictionary, data_D, travel_data, model_func):
    p_dict = dict(zip(parameters_dictionary, params))
    D = model_func(p, parameters_dictionary = p_dict, travel_data = travel_data)
    f = 0
    NB_n = 1 / p_dict['negative_binomial_phi']
    for i in range(len(D)):
        mu = D[i]
        if mu < neg_bin_fix_mu and data_D[i+p.day_1st_death_after_150220] > 0:
            mu = neg_bin_fix_mu
        NB_p = 1 / (1 + mu * p_dict['negative_binomial_phi'])
        f += nbinom.logpmf(data_D[i+p.day_1st_death_after_150220],NB_n, NB_p)
    return -f

def run_optimise():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeats", type=int, help="number of CMA-ES repeats", default=5)
    parser.add_argument("-age", "--age_fit", action='store_true', help="Whether to fit the Age model or not", default=False)
    parser.add_argument("-travel", "--travel_data", action='store_true', help="Whether to use travel data or not", default=False)
    parser.add_argument("-step", "--square_lockdown", action='store_true', help='Whether to use square_lockdown approximation or not. It will be ignored it travel data is FALSE', default=False)
    parser.add_argument("-full", "--fit_full", action='store_true', help='Whether to fit only rho and I0, or all the model parameters', default=False)
    parser.add_argument("-fitstep", "--fit_step", action='store_true', help='Whether to fit step parameters. It will be ignored if travel FALSE and step FALSE', default=False)
    #parser.add_argument("--outputfolder", type=str, help="Folder to upload/store", default='out_SIRvsAGEfits')
    parser.add_argument("--syndata_num", type=int, help="Number of syntetic data set to be fitted", default=1)

    # At the moment, syntethic data sets 2-9 have travel and step options only. There is only one data sets without step and one with neither travel nor step.

    debuging_flag = True
    args = parser.parse_args()
    repeats = args.repeats
    FitAge = args.age_fit
    travel_data = args.travel_data
    FitFull = args.fit_full
    FitStep = args.fit_step
    SyntDataNum_file = args.syndata_num
    SquareLockdown = args.square_lockdown
    
    max_iterations = None


    if debuging_flag:
        FitAge = True
        repeats = 1
        FitFull = True
        FitStep = True
        SquareLockdown = True
        travel_data =True
        np.random.seed(100)
        max_iterations = 10

    if FitFull:
        Fitparams = 'full'
    else:
        Fitparams = 'default'

    # folder to load/ save data
    if SyntDataNum_file == 1: #Original default syntethic data
        folder_path =  os.path.join(MODULE_DIR, 'out_SIRvsAGEfits')
        full_fit_data_file = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy'
    else:
        folder_path =  os.path.join(MODULE_DIR, 'out_SIRvsAGE_SuplementaryFig')
        full_fit_data_file = 'SynteticSItD_default_params_travel_TRUE_step_TRUE_' + str(SyntDataNum_file) + '.npy'
    

    # Label for plots
    Model2_label = r'SI_tD'
    # Number of days to fit
    maxtime_fit = 150
    
    #Noise_flag = 'NBphi_2e-3_' # For file names
    
    # Get parameters, p
    p = Params()
    p.N = 59.1e6
    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    p.IFR = 0.00724 # UK time
    #maxtime = 310 - p.numeric_max_age - p.extra_days_to_simulate


    if travel_data:
        p.square_lockdown = SquareLockdown
        # parameters based on UK google and ONS data
        p.alpha = np.ones(p.maxtime)
        p.lockdown_baseline = 0.2814 #0.2042884852266899
        p.lockdown_offset = 31.57 #34.450147247864166
        if p.square_lockdown:
            data_filename = full_fit_data_file
            travel_label = '_travel_TRUE_step_TRUE_rho_0-2'  # For file names
        else:
            data_filename = 'forward_model_default_params_travel_TRUE.npy'
            travel_label = '_travel_TRUE_step_FALSE'  # For file names
    else:
        p.square_lockdown = False
        data_filename = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_FALSE.npy'
        travel_label = '_travel_FALSE'  # For file names


    # Get Age data from file
    print('Getting simulated data...')
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
    # Fix random seed for reproducibility
    # np.random.seed(100) # not sure if it's working

    
    # Redefinition of parameters for SIR and SIUR ( 1/mean)
    beta_SIR = 1
    theta_SIR = 1 / p.beta_mean
    theta_SIUR = 1 / p.beta_mean
    DeltaD_SIR = int(p.death_mean - p.beta_mean)
    xi_SIUR = 1 / (p.death_mean -p.beta_mean)
    
    # define the params to optimize
    parameters_to_optimise_SIR = parameter_to_optimize_list(FitFull, FitStep, 'SIR')
    parameters_to_optimise_SIRDeltaD = parameter_to_optimize_list(FitFull, FitStep, 'SIRDeltaD')
    parameters_to_optimise_SIUR = parameter_to_optimize_list(FitFull, FitStep, 'SIUR')
    # define bounds and stds:
    bounds_SIR, stds_SIR = par_bounds(parameters_to_optimise_SIR) 
    bounds_SIRDeltaD, stds_SIRDeltaD = par_bounds(parameters_to_optimise_SIRDeltaD)
    bounds_SIUR, stds_SIUR= par_bounds(parameters_to_optimise_SIUR)
    # Get noise model
    noise_model = NOISE_MODEL_MAPPING['NegBinom']


    ## Optimization for the SIR model
    print('--------------------SIR model fitting-----------')
    p.beta = beta_SIR
    p.DeltaD = 0
    p.theta = theta_SIR
    print('*****Using pints****')
    # Get likelihood function
    LL_SIR = noise_model(p, data_D[p.day_1st_death_after_150220:] , parameters_to_optimise_SIR, model_func = get_model_SIR_solution)
    # LL_SIRDel = noise_model(p, data.daily_deaths, parameters_to_optimise_SIR, model_func = get_model_SIR_solution)
    upper_sigma = np.max(data_D)
    log_prior = priors.LogPrior(LL_SIR, upper_sigma)
    parameters_LL, scores_LL = [], []
    # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
    bounds = pints.RectangularBoundaries(log_prior.lower, log_prior.upper)
    # Repeat optimisation multiple times from different initial guesses and pick best
    for i in range(repeats):
        print('Repeat: ' + str(i + 1))
        # Random initial guesses from uniform priors
        x0 = priors.get_good_starting_point(log_prior, LL_SIR, niterations=1000)
        # Create optimiser
        opt = pints.OptimisationController(LL_SIR, x0, boundaries=bounds, method=pints.CMAES)
        opt.set_max_iterations(max_iterations)
        opt.set_parallel(True)
        # Run optimisation
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            xbest, fbest = opt.run()
            parameters_LL.append(xbest)
            scores_LL.append(-fbest)

    # Sort according to smallest function score
    order = np.argsort(scores_LL)
    scores_LL = np.asarray(scores_LL)[order]
    parameters_LL = np.asarray(parameters_LL)[order]

    print('*****SIR Using old method****')
    parametersSIR, scoresSIR, stddevSIR = [], [], []
    for i in range(repeats):
        print('SIR' +  Fitparams +  ' fitting Repeat: ' + str(i + 1))
        optsSIR = cma.CMAOptions()
        optsSIR.set("bounds", bounds_SIR)
        optsSIR.set("CMA_stds", stds_SIR)
        if debuging_flag:
            optsSIR.set("maxfevals",max_iterations)
            #optsSIR.set("seed", 100)
        x0_SIR = np.random.uniform(bounds_SIR[0][0], bounds_SIR[1][0])
        for j in range(len(bounds_SIR[0])-1):
            x0_SIR = np.append(x0_SIR, np.random.uniform(bounds_SIR[0][j+1], bounds_SIR[1][j+1]))
        print(x0_SIR)
        es = cma.fmin(NBneglogL_models, x0_SIR, sigma0=1, args=(p, parameters_to_optimise_SIR, data_D, travel_data, get_model_SIR_solution), options = optsSIR)
        parametersSIR.append(es[0])
        scoresSIR.append(es[1])
        stddevSIR.append(es[6])

    # Sort according to smallest function score
    #SIR
    orderSIR = np.argsort(scoresSIR)
    scoresSIR = np.asarray(scoresSIR)[orderSIR]
    parametersSIR = np.asarray(parametersSIR)[orderSIR]
    stddevSIR = np.asarray(stddevSIR)[orderSIR]

    # Extract best
    obtained_parameters_SIR = parametersSIR[0]
    obtained_stdev_SIR = stddevSIR[0]

    # Store simulations for plotting
    p_dict_SIR = dict(zip(parameters_to_optimise_SIR, obtained_parameters_SIR))
    # update params:
    if p.square_lockdown:
        alpha_SIR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIR)
        alpha_SIR = alpha_SIR[:-p.extra_days_to_simulate]
    else:
        alpha_SIR = np.ones(p.maxtime+1)
    label_SIR = ''
    for l in p_dict_SIR:
        label_SIR = label_SIR + str(l) + ': ' + str('%.4g' % p_dict_SIR.get(l)) + '\n'
    S_s, I_s, Inew_s, R_s, D_s = solve_SIR_difference_equations(p, p_dict_SIR, travel_data)
    rho_SIR = p_dict_SIR.get('rho', p.rho)
    Iinit_SIR = p_dict_SIR.get('Iinit1', p.Iinit1)
    theta_fit_SIR = p_dict_SIR.get('theta',theta_SIR)
    R_0_s = (rho_SIR * p.beta * 1) / theta_fit_SIR
    R_eff_s = ((rho_SIR * p.beta * alpha_SIR) / theta_fit_SIR) * (S_s / p.N)
    
    ## Optimization for the SIR-deltaD model
    print('------------SIR-DeltaD model fitting.-----------')
    # Define theta
    p.beta = beta_SIR
    p.DeltaD = DeltaD_SIR
    p.theta = theta_SIR

    print('***** Using pints ****')
    # Get likelihood function
    LL_SIRDeltaD = noise_model(p, data_D[p.day_1st_death_after_150220:] , parameters_to_optimise_SIRDeltaD, model_func = get_model_SIR_solution)
    # LL_SIRDel = noise_model(p, data.daily_deaths, parameters_to_optimise_SIR, model_func = get_model_SIR_solution)
    upper_sigma = np.max(data_D)
    log_prior_DeltaD = priors.LogPrior(LL_SIRDeltaD, upper_sigma)
    parameters_LLDeltaD, scores_LLDeltaD = [], []
    # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
    bounds_DeltaD = pints.RectangularBoundaries(log_prior_DeltaD.lower, log_prior_DeltaD.upper)
    # Repeat optimisation multiple times from different initial guesses and pick best
    for i in range(repeats):
        print('Repeat: ' + str(i + 1))
        # Random initial guesses from uniform priors
        x0 = priors.get_good_starting_point(log_prior_DeltaD, LL_SIRDeltaD, niterations=1000)
        # Create optimiser
        opt = pints.OptimisationController(LL_SIRDeltaD, x0, boundaries=bounds_DeltaD, method=pints.CMAES)
        opt.set_max_iterations(max_iterations)
        opt.set_parallel(True)
        # Run optimisation
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            xbest, fbest = opt.run()
            parameters_LLDeltaD.append(xbest)
            scores_LLDeltaD.append(-fbest)

    # Sort according to smallest function score
    order = np.argsort(scores_LLDeltaD)
    scores_LLDeltaD = np.asarray(scores_LLDeltaD)[order]
    parameters_LLDeltaD = np.asarray(parameters_LLDeltaD)[order]

    # Optimize only rho and Iinit:
    print('***** SIRDeltaD Using pints ****')
    parametersSIRDeltaD, scoresSIRDeltaD, stddevSIRDeltaD = [], [], []
    for i in range(repeats):
        print('SIR DeltaD' +  Fitparams +  ' fitting Repeat: ' + str(i + 1))
        optsSIRDeltaD = cma.CMAOptions()
        optsSIRDeltaD.set("bounds", bounds_SIRDeltaD)
        optsSIRDeltaD.set("CMA_stds", stds_SIRDeltaD)
        if debuging_flag:
            optsSIRDeltaD.set("maxfevals",max_iterations)
            #optsSIRDeltaD.set("seed", 100)
        x0_SIRDeltaD = np.random.uniform(bounds_SIRDeltaD[0][0], bounds_SIRDeltaD[1][0])
        for j in range(len(bounds_SIRDeltaD[0])-1):
            x0_SIRDeltaD = np.append(x0_SIRDeltaD, np.random.uniform(bounds_SIRDeltaD[0][j+1], bounds_SIRDeltaD[1][j+1]))
        print(x0_SIRDeltaD)
        es = cma.fmin(NBneglogL_models, x0_SIRDeltaD, sigma0=1, args=(p, parameters_to_optimise_SIRDeltaD, data_D, travel_data, get_model_SIR_solution), options = optsSIRDeltaD)
        parametersSIRDeltaD.append(es[0])
        scoresSIRDeltaD.append(es[1])
        stddevSIRDeltaD.append(es[6])

    # Sort according to smallest function score
    #SIR
    orderSIRDeltaD = np.argsort(scoresSIRDeltaD)
    scoresSIRDeltaD = np.asarray(scoresSIRDeltaD)[orderSIRDeltaD]
    parametersSIRDeltaD = np.asarray(parametersSIRDeltaD)[orderSIRDeltaD]
    stddevSIRDeltaD = np.asarray(stddevSIRDeltaD)[orderSIRDeltaD]

    # Extract best
    obtained_parameters_SIRDeltaD = parametersSIRDeltaD[0]
    obtained_stdev_SIRDeltaD = stddevSIRDeltaD[0]
    

    # Store simulations for plotting
    p_dict_SIRDeltaD = dict(zip(parameters_to_optimise_SIRDeltaD, obtained_parameters_SIRDeltaD))
    # update params:
    if p.square_lockdown:
        alpha_SIRDeltaD = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIRDeltaD)
        alpha_SIRDeltaD = alpha_SIRDeltaD[:-p.extra_days_to_simulate]
    else:
        alpha_SIRDeltaD = np.ones(p.maxtime+1)
    label_SIRDeltaD = ''
    for l in p_dict_SIR:
        label_SIRDeltaD = label_SIRDeltaD + str(l) + ': ' + str('%.4g' % p_dict_SIRDeltaD.get(l)) + '\n'
    S_sD, I_sD, Inew_sD, R_sD, D_sD = solve_SIR_difference_equations(p, p_dict_SIRDeltaD, travel_data)
    rho_SIRDeltaD = p_dict_SIRDeltaD.get('rho', p.rho)
    Iinit_SIRDeltaD = p_dict_SIRDeltaD.get('Iinit1', p.Iinit1)
    theta_fit_SIRDeltaD = p_dict_SIRDeltaD.get('theta',theta_SIR)
    R_0_sD = (rho_SIRDeltaD * p.beta * 1) / theta_fit_SIRDeltaD
    R_eff_sD = ((rho_SIRDeltaD * p.beta * alpha_SIRDeltaD) / theta_fit_SIRDeltaD) * (S_sD / p.N)

   

    
    ## Optimization for the SIUR model
    print('----------------------SIUR model fitting-------------')
    # Define theta and xi
    p.theta = theta_SIUR
    p.xi = xi_SIUR

    print('*****Using pints****')
    # Get likelihood function
    LL_SIUR = noise_model(p, data_D[p.day_1st_death_after_150220:] , parameters_to_optimise_SIR, model_func = get_model_SIUR_solution)
    upper_sigma = np.max(data_D)
    log_prior_U = priors.LogPrior(LL_SIUR, upper_sigma)
    parameters_LLU, scores_LLU = [], []
    # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
    bounds_U = pints.RectangularBoundaries(log_prior_U.lower, log_prior_U.upper)
    # Repeat optimisation multiple times from different initial guesses and pick best
    for i in range(repeats):
        print('Repeat: ' + str(i + 1))
        # Random initial guesses from uniform priors
        x0 = priors.get_good_starting_point(log_prior_U, LL_SIUR, niterations=1000)
        # Create optimiser
        opt = pints.OptimisationController(LL_SIUR, x0, boundaries=bounds_U, method=pints.CMAES)
        opt.set_max_iterations(max_iterations)
        opt.set_parallel(True)
        # Run optimisation
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            xbest, fbest = opt.run()
            parameters_LLU.append(xbest)
            scores_LLU.append(-fbest)

    # Sort according to smallest function score
    order = np.argsort(scores_LLU)
    scores_LLU = np.asarray(scores_LLU)[order]
    parameters_LLU = np.asarray(parameters_LLU)[order]

    print('***** SIUR Using old method ****')
    parametersSIUR, scoresSIUR, stddevSIUR = [], [], []
    for i in range(repeats):
        print('SIUR' +  Fitparams +  ' fitting Repeat: ' + str(i + 1))
        optsSIUR = cma.CMAOptions()
        optsSIUR.set("bounds", bounds_SIUR)
        optsSIUR.set("CMA_stds", stds_SIUR)
        if debuging_flag:
            optsSIUR.set("maxfevals",max_iterations)
            #optsSIUR.set("seed", 100)
        x0_SIUR = np.random.uniform(bounds_SIUR[0][0], bounds_SIUR[1][0])
        for j in range(len(bounds_SIUR[0])-1):
            x0_SIUR = np.append(x0_SIUR, np.random.uniform(bounds_SIUR[0][j+1], bounds_SIUR[1][j+1]))
        print(x0_SIUR)
        es = cma.fmin(NBneglogL_models, x0_SIUR, sigma0=1, args=(p, parameters_to_optimise_SIUR, data_D, travel_data, get_model_SIUR_solution), options=optsSIUR)
        parametersSIUR.append(es[0])
        scoresSIUR.append(es[1])
        stddevSIUR.append(es[6])

    # Sort according to smallest function score
    #SIR
    orderSIUR = np.argsort(scoresSIUR)
    scoresSIUR = np.asarray(scoresSIUR)[orderSIUR]
    parametersSIUR = np.asarray(parametersSIUR)[orderSIUR]
    stddevSIUR = np.asarray(stddevSIUR)[orderSIUR]

    # Extract best
    obtained_parameters_SIUR = parametersSIUR[0]
    obtained_stdev_SIUR = stddevSIUR[0]
    

    # Simulations for plots:
    p_dict_SIUR = dict(zip(parameters_to_optimise_SIUR, obtained_parameters_SIUR))
    label_SIUR = ''
    for l in p_dict_SIUR:
        label_SIUR = label_SIUR + str(l) + ': ' + str('%.4g' % p_dict_SIUR.get(l)) + '\n'
    S_u, I_u, Inew_u, N_u, R_u, D_u = solve_SIUR_difference_equations(p, p_dict_SIUR, travel_data)
    if p.square_lockdown:
        alpha_SIUR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIUR)
    else:
        alpha_SIUR = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
    rho_SIUR = p_dict_SIUR.get('rho', p.rho)
    Iinit_SIUR = p_dict_SIUR.get('Iinit1', p.Iinit1)
    theta_fit_SIUR = p_dict_SIUR.get('theta',theta_SIUR)
    R_0_u = (rho_SIUR * p.beta * 1) / theta_fit_SIUR
    R_eff_u = ((rho_SIUR * p.beta * alpha_SIUR[:-p.extra_days_to_simulate]) / theta_fit_SIUR) * (S_u / p.N)

    
    # If Age fit required:
    if FitAge:
        print('Fitting SItR model')
        parameters_to_optimise =  parameter_to_optimize_list(FitFull, FitStep, 'SItD')
        toy_values = [3, 1000, .001]
        if FitStep:
            toy_values.extend([p.lockdown_baseline, p.lockdown_offset])

        #filename_AGE = os.path.join(folder_path, get_file_name_suffix(p, 'syntheticSItRD-'+str(SyntDataNum_file), Noise_flag+'-maxtime-' + str(maxtime_fit), parameters_to_optimise))
        
        bounds_SItR, stds_SItR = par_bounds(parameters_to_optimise)
        # MANUALLY correct that we set rho bigger than 1 for this case
        bounds_SItR[0][0] = 1
        
        print('***** Using pints ****')
        # Get likelihood function
        LL_SItR = noise_model(p, data_D[p.day_1st_death_after_150220:] , parameters_to_optimise_SIR, model_func = get_model_solution)
        upper_sigma = np.max(data_D)
        log_prior_t = priors.LogPrior(LL_SItR, upper_sigma)
        parameters_LLt, scores_LLt = [], []
        # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
        bounds_t = pints.RectangularBoundaries(log_prior_t.lower, log_prior_t.upper)
        # Repeat optimisation multiple times from different initial guesses and pick best
        for i in range(repeats):
            print('Repeat: ' + str(i + 1))
            # Random initial guesses from uniform priors
            x0 = priors.get_good_starting_point(log_prior_t, LL_SItR, niterations=1000)
            # Create optimiser
            opt = pints.OptimisationController(LL_SItR, x0, boundaries=bounds_t, method=pints.CMAES)
            opt.set_max_iterations(max_iterations)
            opt.set_parallel(True)
            # Run optimisation
            with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
                xbest, fbest = opt.run()
                parameters_LLt.append(xbest)
                scores_LLt.append(-fbest)

        # Sort according to smallest function score
        order = np.argsort(scores_LLt)
        scores_LLt = np.asarray(scores_LLt)[order]
        parameters_LLt = np.asarray(parameters_LLt)[order]

        
        print('********** Age model fitting using RMSE.***********')
        # Set up optimisation
        print('Selected data source: ' + 'simulatedAGE')
        #print('Storing results to: ' + filename_AGE + '.txt')
        
        # Calculate beta, gamma and zeta vector rates.
        print('Storing fixed parameters...')
        store_rate_vectors(dict(zip(parameters_to_optimise, toy_values)),p)

        
        ## Optimization for the Age model
        parameters, scores = [], []
        
        # Repeat optimisation multiple times from different initial guesses and pick best
        for i in range(min(repeats, 5)):
            print('Age model fitting using RMSE. Repeat: ' + str(i + 1))
            # CMA-ES (covariance matrix adaptation evolution strategy)
            opts = cma.CMAOptions()
            opts.set("bounds", bounds_SItR)
            opts.set("CMA_stds", stds_SItR)
            if debuging_flag:
                opts.set("maxfevals",max_iterations)
                #opts.set("seed", 100)
            x0 = np.random.uniform(bounds_SItR[0][0], bounds_SItR[1][0])
            for j in range(len(bounds_SItR[0])-1):
                x0 = np.append(x0, np.random.uniform(bounds_SItR[0][j+1], bounds_SItR[1][j+1]))
            print(x0)
            es = cma.fmin(NBneglogL_models, x0, sigma0=1, args=(p, parameters_to_optimise, data_D, travel_data,  get_model_solution), options=opts)
            parameters.append(es[0])
            scores.append(es[1])

        # Sort according to smallest function score
        order = np.argsort(scores)
        scores = np.asarray(scores)[order]
        parameters = np.asarray(parameters)[order]

        # Show results
        print('Age model best parameters:')
        print(parameters[0])
        print('Age model best score:')
        print(-scores[0])

        # Extract best
        obtained_parameters = parameters[0]

        
        # store simulations for plotting
        p_dict_SItRD = dict(zip(parameters_to_optimise, obtained_parameters))
        label_SItRD = ''
        for l in p_dict_SItRD:
            label_SItRD = label_SItRD + str(l) + ': ' + str('%.4g' % p_dict_SItRD.get(l)) + '\n'
        store_rate_vectors(p_dict_SItRD,p)
        S_a, Iday_a, R_a, D_a, Itot_a = solve_difference_equations(p, p_dict_SItRD, travel_data)
        # R0 and Reff
        if p.square_lockdown:
            p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SItRD)
        else:
            p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)

        R_eff_a = calculate_R_instantaneous(p, S_a, p_dict_SItRD)
        R_0_a = R_eff_a[0]
    else:
        Iday_a = data_I
        Itot_a = data_Itot
        S_a = data_S
        S_a_long = data_S_long
        D_a = data_Dreal
        R_a = data_R
        p_dict_SItRD = dict(zip(['rho', 'Iinit1','negative_binomial_phi'], [3.203,860,.002])) # Use the true parameters
        store_rate_vectors(p_dict_SItRD,p)
        if p.square_lockdown:
            p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SItRD)
        else:
            p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
        print(len(p.Gjoint))
        print(len(p.beta))
        print(len(S_a))
        R_eff_a = calculate_R_instantaneous(p, S_a_long, p_dict_SItRD)
        R_0_a = R_eff_a[0]
        
    
    
    ## PLOTS for SIR and SIUR

    

    print('---- Summary ...')
    # get the correct theta:

    print('Old method fits:')
    print('------ Best SIR parameters:------ ')
    print(parametersSIR[0])
    print('Best SIR score:')
    print(-scoresSIR[0])
    print('------ Best SIR-DeltaD parameters:------ ')
    print(parametersSIRDeltaD[0])
    print('Best SIR score:')
    print(-scoresSIRDeltaD[0])
    print('------ Best SIUR parameters:------ ')
    print(parametersSIUR[0])
    print('Best SIUR score:')
    print(-scoresSIUR[0])
    if FitAge:
        print('------ Best SItRD parameters:------ ')
        print(parameters[0])
        print('Best SItD score:')
        print(-scores[0])

    print('New method fits:')
    print('------ Best SIR parameters:------ ')
    print(parameters_LL[0])
    print('Best SIR score:')
    print(-scores_LL[0])
    print('------ Best SIR-DeltaD parameters:------ ')
    print(parameters_LLDeltaD[0])
    print('Best SIR score:')
    print(-scores_LLDeltaD[0])
    print('------ Best SIUR parameters:------ ')
    print(parameters_LLU[0])
    print('Best SIUR score:')
    print(-scores_LLU[0])
    if FitAge:
        print('------ Best SItRD parameters:------ ')
        print(parameters_LLt[0])
        print('Best SItD score:')
        print(-scores_LLt[0])
    
  
    # figure with R_eff
    print('Ploting ...')
    
    # xticks:
    Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
    date_list = [Feb15 + timedelta(days=x) for x in range(maxtime_fit+1)]
    # time
    t = np.linspace(0, maxtime_fit-1, maxtime_fit)
    fig, (ax2, ax, ax4) = plt.subplots(3, 1, figsize=(8.0, 5.5))
    ax.plot(t, Iday_a[0,:maxtime_fit], label = Model2_label)
    ax.plot(t, Inew_s[:maxtime_fit], label='SIRD')
    ax.plot(t, Inew_sD[:maxtime_fit], label=r'SIRD-\Delta D')
    ax.plot(t, Inew_u[:maxtime_fit], label='SIURD')
    ax.legend()
    ax.set_title('Daily new infections')
    ax.set_ylabel('Number')
    ax.set_xticks([x for x in (0, 80, 160, 240) if x < len(date_list)])
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.plot(t[p.day_1st_death_after_150220:], data_D[p.day_1st_death_after_150220:maxtime_fit],'b.', label = 'Synt data')
    ax2.plot(t, D_a[:maxtime_fit], label = Model2_label)
    ax2.plot(t, D_s[:maxtime_fit], label='SIRD')
    ax2.plot(t, D_sD[:maxtime_fit], label=r'SIRD-\Delta D')
    ax2.plot(t, D_u[:maxtime_fit], label='SIURD')
    ax2.legend()
    ax2.set_title('Daily deaths')
    ax2.set_ylabel('Number')
    ax2.set_xticks([x for x in (0, 80, 160, 240) if x < len(date_list)])
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax4.plot(R_eff_a[:maxtime_fit]  , label = Model2_label + ' R_0 = ' + str( round(R_0_a, 2)))
    ax4.plot(R_eff_s[:maxtime_fit], label = 'SIRD R_0 = ' + str( round(R_0_s, 2)))
    ax4.plot(R_eff_sD[:maxtime_fit], label = 'SIRD-D R_0 = ' + str( round(R_0_sD, 2)))
    ax4.plot(R_eff_u[:maxtime_fit], label = 'SIURD R_0 = ' + str( round(R_0_u, 2)))
    ax4.legend()
    ax4.set_xlabel('Date')
    ax4.set_title(r'R$_{\rm{effective}}$')
    ax4.set_xticks([x for x in (0, 80, 160, 240) if x < len(date_list)])
    ax4.set_xticklabels([date_list[x] for x in (0, 80, 160, 240) if x < len(date_list)])
    ax4.grid(True)
    plt.tight_layout()
    # save plot
    #plt.savefig(filename_plot + '.png')

    plt.show()
    
    
    
