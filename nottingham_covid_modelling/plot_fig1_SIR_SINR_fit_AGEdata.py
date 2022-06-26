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
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SIUR_difference_equations, step
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous

FITING_MAPPING = {'default': ['rho', 'Iinit1', 'negative_binomial_phi'], 'full': ['rho', 'Iinit1', 'theta', 'xi','negative_binomial_phi']}

def plot_SIR_fits_fig2():

    travel_data = True
    Fitparams = 'full'
    FitStep = True
    step_flag = True
    save_plot = True
    
    # Definiiton of parameters
    folder = os.path.join(MODULE_DIR, 'out_SIRvsAGEfits')
    filename_plot = 'SyntData_NB_e2-3_data_rho_0-5_and_allFits_fullfit_travel-step_TRUE_fitstep_TRUE'

    filename_SIR = 'Data_syntheticSItRD-NBphi_2e-3__noise-model_SIR-full-fitStep-True-maxtime-150_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset_travel_TRUE_step_TRUE.txt'
    filename_SINR = 'Data_syntheticSItRD-NBphi_2e-3__noise-model_SINR-full-fitStep-True-maxtime-150_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset_travel_TRUE_step_TRUE_rho_0-2.txt'
    filename_SIRDeltaD = 'Data_syntheticSItRD-NBphi_2e-3__noise-model_SIRDeltaD-full-fitStep-True-maxtime-150_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset_travel_TRUE_step_TRUE_rho_0-2.txt'
    filename_AGE = 'Data_syntheticSItRD_noise-model_NBphi_2e-3_-maxtime-150_square-lockdown_rho_Init1_travel_TRUE_step_TRUETrue.txt'
    parameters_to_optimise_SINR = ['rho', 'Iinit1', 'theta','xi', 'negative_binomial_phi', 'lockdown_baseline', 'lockdown_offset']
    parameters_to_optimise_SIR = ['rho', 'Iinit1', 'theta', 'negative_binomial_phi', 'lockdown_baseline', 'lockdown_offset']
    parameters_to_optimise_SIRDeltaD = ['rho', 'Iinit1', 'theta', 'DeltaD', 'negative_binomial_phi', 'lockdown_baseline', 'lockdown_offset']
    parameters_to_optimise = ['rho', 'Iinit1','negative_binomial_phi', 'lockdown_baseline', 'lockdown_offset']

    # Load parameters
    obtained_parameters_SIR = np.loadtxt(os.path.join(folder, filename_SIR))
    p_dict_SIR = dict(zip(parameters_to_optimise_SIR, obtained_parameters_SIR))
    obtained_parameters_SIRDeltaD = np.loadtxt(os.path.join(folder, filename_SIRDeltaD))
    p_dict_SIRDeltaD = dict(zip(parameters_to_optimise_SIRDeltaD, obtained_parameters_SIRDeltaD))
    obtained_parameters_SINR = np.loadtxt(os.path.join(folder, filename_SINR))
    p_dict_SINR = dict(zip(parameters_to_optimise_SINR, obtained_parameters_SINR))
    obtained_parameters = np.loadtxt(os.path.join(folder, filename_AGE))
    p_dict_SItRD = dict(zip(parameters_to_optimise, obtained_parameters))
    
    # True simulation paramters:
    p_dict_data = dict(zip(parameters_to_optimise, [3.203,860,0.002,0.2814, 31.57]))

    # Label for plots
    Model2_label = r'SI$_t$D'
    # Number of days to fit
    maxtime_fit = 150
    neg_bin_fix_mu = 1e-12
    
    
    # Get parameters, p
    p = Params()
    p.N = 59.1e6
    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    p.IFR = 0.00724 # UK time

    
    if travel_data:
        p.square_lockdown = step_flag
        # parameters based on UK google and ONS data
        p.alpha = np.ones(p.maxtime)
        p.lockdown_baseline = 0.2814 #0.2042884852266899
        p.lockdown_offset = 31.57 #34.450147247864166
        if p.square_lockdown:
            data_filename = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy' # 'SItRDmodel_ONSparams_noise_NB-phi_1e-1_NO-R_travel_TRUE_step_TRUE.npy'#
            travel_label = '_travel_TRUE_step_TRUE'
        else:
            data_filename = 'forward_model_default_params_travel_TRUE.npy'
            travel_label = '_travel_TRUE_step_FALSE'
    else:
        p.square_lockdown = False
        data_filename = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_FALSE.npy'
        travel_label = '_travel_FALSE'

    

    # Get Age data from file
    print('Getting simulated data...')
    data = np.load(os.path.join(folder, data_filename))
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
    
    # Redefinition of parameters for SIR and SINR ( 1/mean)
    #beta_SIR = 1
    theta_SIR = 1 / p.beta_mean
    theta_SINR = 1 / p.beta_mean
    DeltaD_SIR = int(p.death_mean - p.beta_mean)
    xi_SINR = 1 / (p.death_mean -p.beta_mean)


    # Simulations
    #print('------- gettin SIR model -----------')
    # Define theta
    #p.beta = beta_SIR
    p.DeltaD = 0
    p.theta = theta_SIR
    # update alpha
    if p.square_lockdown:
        alpha_SIR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIR)
        alpha_SIR = alpha_SIR[:-p.extra_days_to_simulate]
    else:
        alpha_SIR = np.ones(p.maxtime+1)
    
    S_s, I_s, Inew_s, R_s, D_s = solve_SIR_difference_equations(p, p_dict_SIR, travel_data)
    rho_SIR = p_dict_SIR.get('rho', p.rho)
    Iinit_SIR = p_dict_SIR.get('Iinit1', p.Iinit1)
    theta_fit_SIR = p_dict_SIR.get('theta',theta_SIR)
    R_0_s = (rho_SIR  * 1) / theta_fit_SIR
    R_eff_s = ((rho_SIR * alpha_SIR) / theta_fit_SIR) * (S_s / p.N)
    
    
    
    #print('------- SIR-DeltaD: Fitting -----------')
    # Define theta
    #p.beta = beta_SIR
    p.DeltaD = DeltaD_SIR
    p.theta = theta_SIR
    # update params:
    if p.square_lockdown:
        alpha_SIRDeltaD = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIRDeltaD)
        alpha_SIRDeltaD = alpha_SIRDeltaD[:-p.extra_days_to_simulate]
    else:
        alpha_SIRDeltaD = np.ones(p.maxtime+1)
    
    S_sD, I_sD, Inew_sD, R_sD, D_sD = solve_SIR_difference_equations(p, p_dict_SIRDeltaD, travel_data)
    rho_SIRDeltaD = p_dict_SIRDeltaD.get('rho', p.rho)
    Iinit_SIRDeltaD = p_dict_SIRDeltaD.get('Iinit1', p.Iinit1)
    theta_fit_SIRDeltaD = p_dict_SIRDeltaD.get('theta',theta_SIR)
    R_0_sD = (rho_SIRDeltaD  * 1) / theta_fit_SIRDeltaD
    R_eff_sD = ((rho_SIRDeltaD  * alpha_SIRDeltaD) / theta_fit_SIRDeltaD) * (S_sD / p.N)

   


    # print('------- SIUR: Fitting rho and Iinit -----------')
    # Define theta and xi
    p.theta = theta_SINR
    p.xi = xi_SINR
    if p.square_lockdown:
        alpha_SINR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SINR)
    else:
        alpha_SINR = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
    
    S_u, I_u, Inew_u, N_u, R_u, D_u = solve_SIUR_difference_equations(p, p_dict_SINR, travel_data)
    rho_SINR = p_dict_SINR.get('rho', p.rho)
    Iinit_SINR = p_dict_SINR.get('Iinit1', p.Iinit1)
    theta_fit_SINR = p_dict_SINR.get('theta',theta_SINR)
    R_0_u = (rho_SINR * 1) / theta_fit_SINR
    R_eff_u = ((rho_SINR *  alpha_SINR[:-p.extra_days_to_simulate]) / theta_fit_SINR) * (S_u / p.N)

    
    #print('Fitting SItR model')
    store_rate_vectors(p_dict_SItRD,p)
    S_a, Iday_a, R_a, D_a, Itot_a = solve_difference_equations(p, p_dict_SItRD, travel_data)
    # R0 and Reff
    if p.square_lockdown:
        p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SItRD)
    else:
        p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)

    R_eff_a = calculate_R_instantaneous(p, S_a, p_dict_SItRD)
    R_0_a = R_eff_a[0]
    
    # data Reff
    
    store_rate_vectors(p_dict_data,p)
    # R0 and Reff
    if p.square_lockdown:
        p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_data)
    else:
        p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)

    R_eff_data = calculate_R_instantaneous(p, data_S_long, p_dict_data)
    R_0_data = R_eff_data[0]
    
    
    
    ## PLOTS for SIR and SINR
    print('Data day of max new infections:')
    print(np.argmax(data_I[0,:]))
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, t of Inew max ')
    print([round(R_0_data,2),round(min(R_eff_data),2),round(max(R_eff_data),2), np.where(R_eff_data<1)[0][0], np.max(data_I[0,:])])
    print(np.max(data_Itot))

    print('------ Best SIR parameters:------ ')
    print(obtained_parameters_SIR)
    print('Total deaths and recoveries:')
    print([np.sum(D_s),np.sum(R_s)])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, t of Inew max ')
    print([round(R_0_s,2),round(min(R_eff_s),2),round(max(R_eff_s),2), np.where(R_eff_s<1)[0][0], np.max(Inew_s[: -(p.numeric_max_age + p.extra_days_to_simulate)])])
    
    print('------ Best SIR-DeltaD parameters:------ ')
    print(obtained_parameters_SIRDeltaD)
    print('Total deaths and recoveries:')
    print([np.sum(D_sD),np.sum(R_sD)])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, t of Inew max ')
    print([round(R_0_sD,2),round(min(R_eff_sD),2),round(max(R_eff_sD),2), np.where(R_eff_sD<1)[0][0], np.max(Inew_sD[: -(p.numeric_max_age + p.extra_days_to_simulate)])])



    print('------ Best SINR parameters:------ ')
    print(obtained_parameters_SINR)
    print('Day of max new infections:')
    print(np.argmax(Inew_u[: -(p.numeric_max_age + p.extra_days_to_simulate)]))
    print('Total deaths and recoveries:')
    print([np.sum(D_u),np.sum(R_u)])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, t of Inew max ')
    print([round(R_0_u,2),round(min(R_eff_u),2),round(max(R_eff_u),2), np.where(R_eff_u<1)[0][0], np.max(Inew_u[: -(p.numeric_max_age + p.extra_days_to_simulate)])])

    
    print('------ Best SItRD parameters:------ ')
    print(obtained_parameters)
    print('Total deaths and recoveries:')
    print([np.sum(D_a),R_a[-1]])
    print('R_0, R_eff_min, R_eff_max, t where R_eff_a<1, t of Inew max ')
    print([round(R_0_a,2),round(min(R_eff_a),2),round(max(R_eff_a),2), np.where(R_eff_a<1)[0][0], np.max(Iday_a[0,: -(p.numeric_max_age + p.extra_days_to_simulate)])])
    
    # figure with R_eff
    print('Ploting ...')
    
    # xticks:
    Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
    date_list = [Feb15 + timedelta(days=x) for x in range(maxtime_fit+1)]
    # time
    t = np.linspace(0, maxtime_fit-1, maxtime_fit)
    fig, (ax2, ax, ax4) = plt.subplots(3, 1, figsize=(8.0, 6), dpi=300)
    ax.plot(t, data_I[0,:maxtime_fit],'b', label = 'Synt data')
    ax.plot(t, Iday_a[0,:maxtime_fit], label = Model2_label)
    ax.plot(t, Inew_s[:maxtime_fit], label='SIRD')
    ax.plot(t, Inew_sD[:maxtime_fit], label=r'SIRD$_{\Delta D}$')
    ax.plot(t, Inew_u[:maxtime_fit], label='SIURD')
    ax.legend()
    ax.set_title('Daily new infections')
    ax.set_ylabel('Number')
    ax.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
    ax.grid(True)
    ax.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
    ax2.plot(t[p.day_1st_death_after_150220:], data_D[p.day_1st_death_after_150220:maxtime_fit],'b.', label = 'Synt data')
    ax2.plot(t, D_a[:maxtime_fit], label = Model2_label)
    ax2.plot(t, D_s[:maxtime_fit], label='SIRD')
    ax2.plot(t, D_sD[:maxtime_fit], label=r'SIRD$_{\Delta D}$')
    ax2.plot(t, D_u[:maxtime_fit], label='SIURD')
    ax2.legend()
    ax2.set_title('Daily deaths')
    ax2.set_ylabel('Number')
    ax2.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
    ax2.grid(True)
    ax2.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
    ax4.plot(R_eff_data[:maxtime_fit], 'b', label = r'Synt data $R_0 =$ ' + str( round(R_0_data, 2)))
    ax4.plot(R_eff_a[:maxtime_fit]  , label = Model2_label + r' $R_0 =$ ' + str( round(R_0_a, 2)))
    ax4.plot(R_eff_s[:maxtime_fit], label = r'SIRD $R_0 = $' + str( round(R_0_s, 2)))
    ax4.plot(R_eff_sD[:maxtime_fit], label = r'SIRD$_{\Delta D}$ $R_0 =$ ' + str( round(R_0_sD, 2)))
    ax4.plot(R_eff_u[:maxtime_fit], label = r'SIURD $R_0 = $' + str( round(R_0_u, 2)))
    ax4.legend()
    ax4.set_xlabel('Date')
    ax4.set_title(r'$\mathcal{R}$')
    ax4.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
    ax4.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
    ax4.grid(True)
    plt.tight_layout()
    # save plot
    if save_plot:
        plt.savefig(os.path.join(folder, filename_plot) + '.png')

    plt.show()
    
    
    
