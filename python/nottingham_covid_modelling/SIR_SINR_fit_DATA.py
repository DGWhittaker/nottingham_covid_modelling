import os
import argparse

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
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SINR_difference_equations, step
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous



def run_optimise():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repeats", type=int, help="number of CMA-ES repeats", default=5)
    parser.add_argument("-age", "--age_fit", action='store_true', help="Wheter to fit the Age model or not", default=False)
    parser.add_argument("-travel", "--travel_data", action='store_true', help="Wheter to use travel data or not", default=False)
    parser.add_argument("-step", "--square_lockdown", action='store_true', help='Wheter to use square_lockdown approximation or not. It will be ignored it travel data is FALSE', default=False)
    parser.add_argument("-fitstep", "--fit_step", action='store_true', help='Wheter to fit step parameters. It will be ignored if travel FALSE and step FALSE', default=False)

    args = parser.parse_args()
    repeats = args.repeats
    FitAge = args.age_fit
    travel_data = args.travel_data
    FitStep = args.fit_step
    
    # Define paramters to estimate:
    parameters_to_optimise_SINR = ['rho', 'Iinit1', 'theta', 'xi', 'negative_binomial_phi']
    parameters_to_optimise_SIR = ['rho', 'Iinit1', 'theta', 'negative_binomial_phi']
    parameters_to_optimise_SIRDeltaD = ['rho', 'Iinit1', 'theta', 'DeltaD', 'negative_binomial_phi']
    if FitStep:
        parameters_to_optimise_SIR.extend(['lockdown_baseline', 'lockdown_offset'])
        parameters_to_optimise_SIRDeltaD.extend(['lockdown_baseline', 'lockdown_offset'])
        parameters_to_optimise_SINR.extend(['lockdown_baseline', 'lockdown_offset'])
    

    # Label for plots
    Model2_label = r'SI_tD'
    # Number of days to fit
    maxtime_fit = 150
    Noise_flag = 'NBnoise'
    neg_bin_fix_mu = 1e-12
    
    # Get parameters, p
    p = Params()
    p.IFR = 0.00724 # UK time
    
    # Get data:
    print('Getting data...')
    data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', '..', 'data', 'archive', 'current'))
    
    # The length of fit should be larger than the the first death since feb 15
    assert p.day_1st_death_after_150220 < maxtime_fit-1, "The fit length happen before the first death"
        
    # overwrite maxtime to the fitting one if desired fitted one is smaller
    if len(data.daily_deaths) > maxtime_fit - p.day_1st_death_after_150220:
        data_D = data.daily_deaths[:maxtime_fit - p.day_1st_death_after_150220 + 1]
         # to get the same data and fit lenghts as in Data_loader
        p.maxtime = maxtime_fit + p.numeric_max_age + p.extra_days_to_simulate
    else:
        data_D = data.daily_deaths
        
    
    # Define folder and filenames to save:
    folder_path =  os.path.join(MODULE_DIR, 'out_SIRvsDATAfits')
    if travel_data:
        p.square_lockdown = args.square_lockdown
        if p.square_lockdown:
            travel_label = '_travel_TRUE_step_TRUE'
        else:
            travel_label = '_travel_TRUE_step_FALSE'
    else:
        p.square_lockdown = False
        travel_label = '_travel_FALSE'
    filename_SIR = os.path.join(folder_path, get_file_name_suffix(p, 'realDataUK-ONS-'+Noise_flag, 'SIR-FULL-fitStep-' + str(FitStep) + '-maxtime-' + str(maxtime_fit), parameters_to_optimise_SIR) + travel_label)
    filename_SIRDeltaD = os.path.join(folder_path, get_file_name_suffix(p, 'realDataUK-ONS-'+Noise_flag, 'SIRDeltaD-FULL-fitStep-' + str(FitStep) + '-maxtime-' + str(maxtime_fit), parameters_to_optimise_SIR) + travel_label)
    filename_SINR = os.path.join(folder_path, get_file_name_suffix(p, 'realDataUK-ONS-'+Noise_flag, 'SINR-FULL-fitStep-' + str(FitStep) + '-maxtime-' + str(maxtime_fit), parameters_to_optimise_SINR) + travel_label)
    filename_plot = os.path.join(folder_path, 'RealDataUK-ONS-NBnoise-fit-' + Noise_flag + '-FULL-fitStep-' + str(FitStep) + '-fitSItR-' + str(FitAge) + '-maxtime-' + str(maxtime_fit) + travel_label)
    
    
    # Fix random seed for reproducibility
    np.random.seed(100) # not sure if it's working
    
    

    # NB likelihoods:
    def NBlike_SItRD(params, p, parameters_dictionary, travel_data, data_D):
        p_dict = dict(zip(parameters_dictionary, params))
        _, _, _, D, _ = solve_difference_equations(p, parameters_dictionary = p_dict, travel_data = travel_data)
        D = D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]
        f = 0
        NB_n = 1 / p_dict['negative_binomial_phi']
        for i in range(len(D)):
            mu = D[i]
            if mu < neg_bin_fix_mu and data_D[i] > 0:
                mu = neg_bin_fix_mu
            NB_p = 1 / (1 + mu * p_dict['negative_binomial_phi'])
            f += nbinom.logpmf(data_D[i], NB_n, NB_p)
        return -f
    def NBlike_SIR(params, p, parameters_dictionary, data_D, travel_data):
        p_dict = dict(zip(parameters_dictionary, params))
        _, _, _, _, D = solve_SIR_difference_equations(p, parameters_dictionary = p_dict, travel_data = travel_data)
        DeltaD = p_dict.get('DeltaD', p.DeltaD)
        D = D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate + int(DeltaD))]
        f = 0
        NB_n = 1 / p_dict['negative_binomial_phi']
        for i in range(len(D)):
            mu = D[i]
            if mu < neg_bin_fix_mu and data_D[i] > 0:
                mu = neg_bin_fix_mu
            NB_p = 1 / (1 + mu * p_dict['negative_binomial_phi'])
            f += nbinom.logpmf(data_D[i], NB_n, NB_p)
        return -f
    def NBlike_SINR(params, p, parameters_dictionary, data_D, travel_data):
        p_dict = dict(zip(parameters_dictionary, params))
        _, _, _, _, _, D = solve_SINR_difference_equations(p, parameters_dictionary = p_dict, travel_data = travel_data)
        D = D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]
        f = 0
        NB_n = 1 / p_dict['negative_binomial_phi']
        for i in range(len(D)):
            mu = D[i]
            if mu < neg_bin_fix_mu and data_D[i] > 0:
                mu = neg_bin_fix_mu
            NB_p = 1 / (1 + mu * p_dict['negative_binomial_phi'])
            f += nbinom.logpmf(data_D[i],NB_n, NB_p)
        return -f
    
    # Redefinition of parameters
    beta_SIR = 1 # to fit all in the rho paramters
    # TOy parameters:
    theta = 1 / p.beta_mean
    DeltaD_SIR = int(p.death_mean - p.beta_mean)
    xi_SINR = 1 / (p.death_mean -p.beta_mean)
    
    # Define priors for SIR and SINR
    rho_lower = 1
    rho_upper =  5
    rho_SIR_lower = 0
    rho_SIR_upper = 2 # 1
    Iinit1_lower = 1
    Iinit1_upper = 5e4
    theta_lower = 0
    theta_upper = 1
    DeltaD_lower = 0
    DeltaD_upper = 50
    xi_lower = 0
    xi_upper = 1
    negative_binomial_phi_lower, negative_binomial_phi_upper = 0, 1
    lockdown_baseline_lower, lockdown_baseline_upper = 0, 1
    lockdown_offset_lower, lockdown_offset_upper = 0, 100
    
    # define bounds and stds:
    bounds_SIR = [[rho_SIR_lower, Iinit1_lower, theta_lower, negative_binomial_phi_lower], [ rho_SIR_upper, Iinit1_upper, theta_upper, negative_binomial_phi_upper]]
    bounds_SIRDeltaD = [[rho_SIR_lower, Iinit1_lower, theta_lower, DeltaD_lower, negative_binomial_phi_lower], [ rho_SIR_upper, Iinit1_upper, theta_upper, DeltaD_upper, negative_binomial_phi_upper]]
    bounds_SINR = [[rho_SIR_lower, Iinit1_lower, theta_lower, xi_lower, negative_binomial_phi_lower], [ rho_SIR_upper, Iinit1_upper, theta_upper, xi_upper, negative_binomial_phi_upper]]
    stds_SIR = [1e-2, 100, 1e-2, 1e-2]
    stds_SIRDeltaD = [1e-2, 100, 1e-2, 10, 1e-2]
    stds_SINR = [1e-2, 100, 1e-2, 1e-2,1e-2]
    if FitStep:
        bounds_SIR[0].extend([lockdown_baseline_lower, lockdown_offset_lower])
        bounds_SIR[1].extend([lockdown_baseline_upper, lockdown_offset_upper])
        bounds_SIRDeltaD[0].extend([lockdown_baseline_lower, lockdown_offset_lower])
        bounds_SIRDeltaD[1].extend([lockdown_baseline_upper, lockdown_offset_upper])
        bounds_SINR[0].extend([lockdown_baseline_lower, lockdown_offset_lower])
        bounds_SINR[1].extend([lockdown_baseline_upper, lockdown_offset_upper])
        stds_SIR.extend([1e-2, 10])
        stds_SIRDeltaD.extend([1e-2, 10])
        stds_SINR.extend([1e-2, 10])
    
    ## Optimization for the SIR model
    print('SIR model fitting.')
    # Optimize only rho and Iinit:
    print('------- SIR: Fitting -----------')
    # Define theta
    p.beta = beta_SIR
    p.DeltaD = 0
    p.theta = theta
    parametersSIR, scoresSIR, stddevSIR = [], [], []
    for i in range(repeats):
        print('SIR FULL fitting Repeat: ' + str(i + 1))
        optsSIR = cma.CMAOptions()
        optsSIR.set("bounds", bounds_SIR)
        optsSIR.set("CMA_stds", stds_SIR)
        x0_SIR = np.random.uniform(bounds_SIR[0][0], bounds_SIR[1][0])
        for j in range(len(bounds_SIR[0])-1):
            x0_SIR = np.append(x0_SIR, np.random.uniform(bounds_SIR[0][j+1], bounds_SIR[1][j+1]))
        print(x0_SIR)
        es = cma.fmin(NBlike_SIR, x0_SIR, sigma0=1, args=(p, parameters_to_optimise_SIR, data_D, travel_data), options = optsSIR)
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
    # Store SIR resutls
    print('Storing default fit SIR model best result...')
    with open(filename_SIR + '.txt', 'w') as f:
        for x in obtained_parameters_SIR:
            f.write(pints.strfloat(x) + '\n')

    print('Storing default fit all_SIR SIR model errors...')
    with open(filename_SIR + '-errors.txt', 'w') as f:
        for score in scoresSIR:
            f.write(pints.strfloat(-score) + '\n')

    # Store simulations for plotting
    p_dict_SIR = dict(zip(parameters_to_optimise_SIR, obtained_parameters_SIR))
    # update params:
    if p.square_lockdown:
        alpha_SIR = step(p, lgoog_data=p.maxtime-(p.numeric_max_age + p.extra_days_to_simulate)+1, parameters_dictionary = p_dict_SIR)
        #alpha_SIR = alpha_SIR[:-p.extra_days_to_simulate]
    else:
        alpha_SIR = np.ones(p.maxtime+1)
    S_s, I_s, Inew_s, R_s, D_s = solve_SIR_difference_equations(p, p_dict_SIR, travel_data)
    rho_SIR = p_dict_SIR.get('rho', p.rho)
    Iinit_SIR = p_dict_SIR.get('Iinit1', p.Iinit1)
    theta_fit_SIR = p_dict_SIR.get('theta',theta)
    R_0_s = (rho_SIR * p.beta * 1) / theta_fit_SIR
    R_eff_s = ((rho_SIR * p.beta * alpha_SIR) / theta_fit_SIR) * (S_s / p.N)
    
    ## Optimization for the SIR-deltaD model
    print('SIR-DeltaD model fitting.')
    # Optimize only rho and Iinit:
    print('------- SIR-DeltaD: Fitting -----------')
    # Define theta
    p.beta = beta_SIR
    parametersSIRDeltaD, scoresSIRDeltaD, stddevSIRDeltaD = [], [], []
    for i in range(repeats):
        print('SIR DeltaD FULL fitting Repeat: ' + str(i + 1))
        optsSIRDeltaD = cma.CMAOptions()
        optsSIRDeltaD.set("bounds", bounds_SIRDeltaD)
        optsSIRDeltaD.set("CMA_stds", stds_SIRDeltaD)
        x0_SIRDeltaD = np.random.uniform(bounds_SIRDeltaD[0][0], bounds_SIRDeltaD[1][0])
        for j in range(len(bounds_SIRDeltaD[0])-1):
            x0_SIRDeltaD = np.append(x0_SIRDeltaD, np.random.uniform(bounds_SIRDeltaD[0][j+1], bounds_SIRDeltaD[1][j+1]))
        print(x0_SIRDeltaD)
        es = cma.fmin(NBlike_SIR, x0_SIRDeltaD, sigma0=1, args=(p, parameters_to_optimise_SIRDeltaD, data_D, travel_data), options = optsSIRDeltaD)
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
    # Store SIR resutls
    print('Storing default fit SIR-DeltaD model best result...')
    with open(filename_SIRDeltaD + '.txt', 'w') as f:
        for x in obtained_parameters_SIRDeltaD:
            f.write(pints.strfloat(x) + '\n')

    print('Storing default fit all_SIR SIR model errors...')
    with open(filename_SIRDeltaD + '-errors.txt', 'w') as f:
        for score in scoresSIRDeltaD:
            f.write(pints.strfloat(-score) + '\n')

    # Store simulations for plotting
    p_dict_SIRDeltaD = dict(zip(parameters_to_optimise_SIRDeltaD, obtained_parameters_SIRDeltaD))
    # update params:
    if p.square_lockdown:
        alpha_SIRDeltaD = step(p, lgoog_data=p.maxtime-(p.numeric_max_age + p.extra_days_to_simulate)+1, parameters_dictionary = p_dict_SIRDeltaD)
        #alpha_SIRDeltaD = alpha_SIR[:-p.extra_days_to_simulate]
    else:
            alpha_SIR = np.ones(p.maxtime+1)
    S_sD, I_sD, Inew_sD, R_sD, D_sD = solve_SIR_difference_equations(p, p_dict_SIRDeltaD, travel_data)
    rho_SIRDeltaD = p_dict_SIRDeltaD.get('rho', p.rho)
    Iinit_SIRDeltaD = p_dict_SIRDeltaD.get('Iinit1', p.Iinit1)
    theta_fit_SIRDeltaD = p_dict_SIRDeltaD.get('theta',theta)
    R_0_sD = (rho_SIRDeltaD * p.beta * 1) / theta_fit_SIRDeltaD
    R_eff_sD = ((rho_SIRDeltaD * p.beta * alpha_SIRDeltaD) / theta_fit_SIRDeltaD) * (S_sD / p.N)

   

    
    ## Optimization for the SINR model
    print('SIUR model fitting.')
    # Optimize only rho and Iinit:
    print('------- SIUR: Fitting rho and Iinit -----------')
    p.beta = beta_SIR
    p.theta = theta
    p.xi = xi_SINR
    parametersSINR, scoresSINR, stddevSINR = [], [], []
    for i in range(repeats):
        print('SIUR FULL fitting Repeat: ' + str(i + 1))
        optsSINR = cma.CMAOptions()
        optsSINR.set("bounds", bounds_SINR)
        optsSINR.set("CMA_stds", stds_SINR)
        x0_SINR = np.random.uniform(bounds_SINR[0][0], bounds_SINR[1][0])
        for j in range(len(bounds_SINR[0])-1):
            x0_SINR = np.append(x0_SINR, np.random.uniform(bounds_SINR[0][j+1], bounds_SINR[1][j+1]))
        print(x0_SINR)
        es = cma.fmin(NBlike_SINR, x0_SINR, sigma0=1, args=(p, parameters_to_optimise_SINR, data_D, travel_data), options=optsSINR)
        parametersSINR.append(es[0])
        scoresSINR.append(es[1])
        stddevSINR.append(es[6])

    # Sort according to smallest function score
    #SIR
    orderSINR = np.argsort(scoresSINR)
    scoresSINR = np.asarray(scoresSINR)[orderSINR]
    parametersSINR = np.asarray(parametersSINR)[orderSINR]
    stddevSINR = np.asarray(stddevSINR)[orderSINR]

    # Extract best
    obtained_parameters_SINR = parametersSINR[0]
    obtained_stdev_SINR = stddevSINR[0]
    # Store SIR resutls
    print('Storing default fit SINR model best result...')
    with open(filename_SINR + '.txt', 'w') as f:
        for x in obtained_parameters_SINR:
            f.write(pints.strfloat(x) + '\n')

    print('Storing default fit SINR model errors...')
    with open(filename_SINR + '-errors.txt', 'w') as f:
        for score in scoresSINR:
            f.write(pints.strfloat(-score) + '\n')

    # Simulations for plots:
    p_dict_SINR = dict(zip(parameters_to_optimise_SINR, obtained_parameters_SINR))
    S_u, I_u, Inew_u, N_u, R_u, D_u = solve_SINR_difference_equations(p, p_dict_SINR, travel_data)
    
    if p.square_lockdown:
        alpha_SINR = step(p, lgoog_data = p.maxtime-(p.numeric_max_age + p.extra_days_to_simulate)+1, parameters_dictionary = p_dict_SINR)
    else:
        alpha_SINR = np.ones(p.maxtime+1)
    rho_SINR = p_dict_SINR.get('rho', p.rho)
    Iinit_SINR = p_dict_SINR.get('Iinit1', p.Iinit1)
    theta_fit_SINR = p_dict_SINR.get('theta',theta)
    R_0_u = (rho_SINR * p.beta * 1) / theta_fit_SINR
    R_eff_u = ((rho_SINR * p.beta * alpha_SINR) / theta_fit_SINR) * (S_u / p.N)

    
    # If Age fit required:
    if FitAge:
        print('Fitting SItR model')
        parameters_to_optimise = ['rho', 'Iinit1', 'negative_binomial_phi']
        toy_values = [3, 1000, .001]
        filename_AGE = os.path.join(folder_path, get_file_name_suffix(p, 'RealDataO-ONS-', Noise_flag+'-maxtime-' + str(maxtime_fit), parameters_to_optimise))
        
        bounds_SItR = [[ rho_lower, Iinit1_lower, negative_binomial_phi_lower], [ rho_upper, Iinit1_upper, negative_binomial_phi_upper]]
        stds_SItR = [1e-1, 100, 1e-2]

        if FitStep:
            parameters_to_optimise.extend(['lockdown_baseline', 'lockdown_offset'])
            bounds_SItR[0].extend([lockdown_baseline_lower, lockdown_offset_lower])
            bounds_SItR[1].extend([lockdown_baseline_upper, lockdown_offset_upper])
            toy_values.extend([p.lockdown_baseline, p.lockdown_offset])
            stds_SItR.extend([1e-2, 10])

        
        print('Age model fitting using RMSE.')
        # Set up optimisation
        print('Selected data source: ' + 'simulatedAGE')
        print('Storing results to: ' + filename_AGE + '.txt')
        
        # Calculate beta, gamma and zeta vector rates.
        print('Storing fixed parameters...')
        store_rate_vectors(dict(zip(parameters_to_optimise, toy_values)),p)

        
        ## Optimization for the Age model
        parameters, scores = [], []
        
        # Repeat optimisation multiple times from different initial guesses and pick best
        for i in range(min(repeats, 2)):
            print('Age model fitting using RMSE. Repeat: ' + str(i + 1))
            # CMA-ES (covariance matrix adaptation evolution strategy)
            opts = cma.CMAOptions()
            #opts.set("seed", 100)
            opts.set("bounds", bounds_SItR)
            opts.set("CMA_stds", stds_SItR)
            x0 = np.random.uniform(bounds_SItR[0][0], bounds_SItR[1][0])
            for j in range(len(bounds_SItR[0])-1):
                x0 = np.append(x0, np.random.uniform(bounds_SItR[0][j+1], bounds_SItR[1][j+1]))
            print(x0)
            es = cma.fmin(NBlike_SItRD, x0, sigma0=1, args=(p, parameters_to_optimise, travel_data, data_D), options=opts)
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

        # Store results
        print('Storing age modelbest result...')
        with open(filename_AGE + travel_label + str(travel_data) + '.txt', 'w') as f:
            for x in obtained_parameters:
                f.write(pints.strfloat(x) + '\n')

        print('Storing all age model errors...')
        with open(filename_AGE + travel_label + str(travel_data) + '-errors.txt', 'w') as f:
            for score in scores:
                f.write(pints.strfloat(-score) + '\n')
        
        # store simulations for plotting
        p_dict_SItRD = dict(zip(parameters_to_optimise, obtained_parameters))
        store_rate_vectors(p_dict_SItRD,p)
        S_a, Iday_a, R_a, D_a, Itot_a = solve_difference_equations(p, p_dict_SItRD, travel_data)
        # R0 and Reff
        if p.square_lockdown:
            p.alpha = step(p, lgoog_data = p.maxtime-(p.numeric_max_age + p.extra_days_to_simulate)+1, parameters_dictionary = p_dict_SItRD)
        else:
            p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)

        R_eff_a = calculate_R_instantaneous(p, S_a, p_dict_SItRD)
        R_0_a = R_eff_a[0]
#    else:
#        Iday_a = data_I
#        Itot_a = data_Itot
#        S_a = data_S
#        S_a_long = data_S_long
#        D_a = data_D
#        R_a = data_R
#        p_dict_SItRD = dict(zip(['rho', 'Iinit1'], [3.203,860])) # Use the true parameters
#        store_rate_vectors(p_dict_SItRD,p)
#        if p.square_lockdown:
#            p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SItRD)
#        else:
#            p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
#        print(len(p.Gjoint))
#        print(len(p.beta))
#        print(len(S_a))
#        R_eff_a = calculate_R_instantaneous(p, S_a_long, p_dict_SItRD)
#        R_0_a = R_eff_a[0]
#
    
    
    ## PLOTS for SIR and SINR

    

    print('---- Summary ...')
    # get the correct theta:

    print('Data day of max new infections: Not available from death data')
    #print(np.argmax(data_I[0,:]))

    print('------ Best SIR parameters:------ ')
    print(parametersSIR[0])
    print('Std Dev:')
    print(stddevSIR[0])
    print('Best SIR score:')
    print(-scoresSIR[0])
    print('Total deaths and recoveries:')
    print([np.sum(D_s),np.sum(R_s)])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, t of Inew max ')
    print([round(R_0_s,2),round(min(R_eff_s),2),round(max(R_eff_s),2), np.where(R_eff_s<1)[0][0], max(Inew_s[: -(p.numeric_max_age + p.extra_days_to_simulate)])])
    
    print('------ Best SIR-DeltaD parameters:------ ')
    print(parametersSIRDeltaD[0])
    print('Std Dev:')
    print(stddevSIRDeltaD[0])
    print('Best SIR score:')
    print(-scoresSIRDeltaD[0])
    print('Total deaths and recoveries:')
    print([np.sum(D_sD),np.sum(R_sD)])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, Inew max ')
    print([round(R_0_sD,2),round(min(R_eff_sD),2),round(max(R_eff_sD),2), np.where(R_eff_sD<1)[0][0], max(Inew_sD[: -(p.numeric_max_age + p.extra_days_to_simulate)])])



    print('------ Best SINR parameters:------ ')
    print(parametersSINR[0])
    print('Std Dev:')
    print(stddevSINR[0])
    print('Best SINR score:')
    print(-scoresSINR[0])
    print('Day of max new infections:')
    print(np.argmax(Inew_u[: -(p.numeric_max_age + p.extra_days_to_simulate)]))
    print('Total deaths and recoveries with 2-DF:')
    print([np.sum(D_u),np.sum(R_u)])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1,Inew max ')
    print([round(R_0_u,2),round(min(R_eff_u),2),round(max(R_eff_u),2), np.where(R_eff_u<1)[0][0], max(Inew_u[: -(p.numeric_max_age + p.extra_days_to_simulate)])])

    if FitAge:
        print('------ Best SItRD parameters:------ ')
        print(parameters[0])
        print('Std Dev:')
        print(-scores[0])
        print('Total deaths and recoveries with 2-DF:')
        print([np.sum(D_a),R_a[-1]])
        print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, Inew max ')
        print([round(R_0_a,2),round(min(R_eff_a),2),round(max(R_eff_a),2), np.where(R_eff_a<1)[0][0], max(Iday_a[0,: -(p.numeric_max_age + p.extra_days_to_simulate)])])
    
    # figure with R_eff
    print('Ploting ...')
    
    # xticks:
    Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
    date_list = [Feb15 + timedelta(days=x) for x in range(maxtime_fit+1)]
    # time
    t = np.linspace(0, maxtime_fit-1, maxtime_fit)
    fig, (ax2, ax, ax4) = plt.subplots(3, 1, figsize=(8.0, 5.5))
    if FitAge:
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
    ax2.plot(t[p.day_1st_death_after_150220:], data_D[:maxtime_fit-p.day_1st_death_after_150220],'b.', label = 'Synt data')
    if FitAge:
        ax2.plot(t[p.day_1st_death_after_150220:], D_a[:maxtime_fit-p.day_1st_death_after_150220], label = Model2_label)
    ax2.plot(t, D_s[:maxtime_fit], label='SIRD')
    ax2.plot(t, D_sD[:maxtime_fit], label=r'SIRD-\Delta D')
    ax2.plot(t, D_u[:maxtime_fit], label='SIURD')
    ax2.legend()
    ax2.set_title('Daily deaths')
    ax2.set_ylabel('Number')
    ax2.set_xticks([x for x in (0, 80, 160, 240) if x < len(date_list)])
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)
    if FitAge:
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
    plt.savefig(filename_plot + '.png')

    plt.show()
    
    
    
    
    
