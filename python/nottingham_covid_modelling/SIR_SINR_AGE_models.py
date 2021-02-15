import argparse
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
# Load project modules
from nottingham_covid_modelling.lib.equations import solve_SIR_difference_equations, solve_difference_equations, solve_SINR_difference_equations, store_rate_vectors, step, tanh_spline
from nottingham_covid_modelling.lib.settings import Params
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous


def SIR_SINR_AGE_model_default(skip_data_folder=True):
    # Get parameters, p
    p = Params()
    START_DATE = 'Feb15'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--startmonth", type=str, default=START_DATE,
                        help='Starting month for simulation (default=' + START_DATE + ')')
    parser.add_argument("--outputnumbers", type=str, default=None, help='Output path for saving the numbers')
    parser.add_argument("-travel", "--travel_data", action='store_true', help='Wheter to use travel data or not', default=False)
    parser.add_argument("-step", "--square_lockdown", action='store_true', help='Wheter to use square_lockdown approximation or not. It will be ignored it travel data is FALSE', default=False)
    args = parser.parse_args()

    # For plots:
    STEP = 30  # How big is the step between labelled items on the graph
    start_date = datetime.strptime(args.startmonth, '%b%y').date()  # Get start date as date object
    xticklabels = [start_date]  # Start building xticket lables with the start date
    Model2_label = r'$SI_tRD$'
    
    negative_binomial_phi = 0.002#0.002
    p.N = 59.1e6
    p.IFR = 0.00724 # UK time
    
    
    # Default length data
    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    #p.maxtime = 310 - (p.numeric_max_age + p.extra_days_to_simulate)
    
    # Take only the first 150 days
    p.maxtime = 150 + (p.numeric_max_age + p.extra_days_to_simulate) - 1
    
    # Define travel data:
    travel_data = args.travel_data
    if travel_data:
        p.square_lockdown = args.square_lockdown
        # parameters based on UK google and ONS data
        p.lockdown_baseline = 0.2814 #0.2042884852266899
        p.lockdown_offset = 31.57 #34.450147247864166
    else:
        p.square_lockdown = False

    
    # Default parameters for the 3 models
    rho = 3.203 #2.4
    rho_SIR = 3.203/p.beta_mean
    Iinit1 = 860 #1000
    beta_SIR = 1
    theta = 1/p.beta_mean
    xi = 1/(p.death_mean - p.beta_mean)
    DeltaD = 18-5
    default_params = {'rho': rho, 'Iinit1': Iinit1}
    
    if travel_data:
        if p.square_lockdown:
            p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = default_params)
        else:
            p.alpha = tanh_spline(p, lgoog_data = p.maxtime - p.numeric_max_age + 1, parameters_dictionary = default_params)
    else:
        p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
    
    
    
    # Simulate age model
    store_rate_vectors(default_params,p)
    S_a, Iday_a, R_a, D_a, Itot_a = solve_difference_equations(p, default_params, travel_data)
    # R0 and Reff
    R_eff_a = calculate_R_instantaneous(p, S_a, default_params)
    R_0_a = R_eff_a[0]
    print(round(R_0_a,2))
    print([round(min(R_eff_a),2),round(max(R_eff_a),2)])
    print(np.where(R_eff_a<1)[0][0])
    print(np.argmax(Iday_a[0,: -(p.numeric_max_age + p.extra_days_to_simulate)]))
    
    # Add noise to the data:
    NB_n = 1 / negative_binomial_phi
    D_noise =  np.zeros(len(D_a))
    for i in range(len(D_a)):
        NB_p = 1 / (1 + D_a[i] * negative_binomial_phi)
        D_noise[i] = nbinom.rvs(NB_n, NB_p)
    
    # Simulate basic SIR model
    # Parameters:
    #rho = 5
    #Iinit1 = 141.3
    #p.lockdown_baseline = 0.804
    #p.lockdown_offset = 51.92
    #p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = default_params)
    
    
    default_params_SIR = {'rho': rho_SIR, 'Iinit1': Iinit1}
    p.beta = beta_SIR
    p.theta = theta
    p.DeltaD = DeltaD
    S_i, I_i, Inew_i, R_i, D_i = solve_SIR_difference_equations(p, default_params_SIR, travel_data)
    R_0_i = (rho_SIR * p.beta * 1) / p.theta
    R_eff_i = ((rho_SIR * p.beta * p.alpha[:-p.extra_days_to_simulate]) / p.theta) * (S_i / p.N)
    print('SIR')
    print(round(R_0_i,2))
    print([round(min(R_eff_i),2),round(max(R_eff_i),2)])
    print(np.where(R_eff_i<1)[0][0])
    print(np.argmax(Inew_i[: -(p.numeric_max_age + p.extra_days_to_simulate)]))
    
    # Simulate basic SINR model
    # Parameters:
    default_params_SIR = {'rho': rho_SIR, 'Iinit1': Iinit1}
    p.theta = theta
    p.xi = xi
    
    #p.lockdown_baseline = 0.721
    #p.lockdown_offset = 46.1
    p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = default_params)
    
    S_u, I_u, Inew_u, N_u, R_u, D_u = solve_SINR_difference_equations(p, default_params_SIR, travel_data)
    R_0_u = (rho_SIR * p.beta * 1) / p.theta
    R_eff_u = ((rho_SIR * p.beta * p.alpha[:-p.extra_days_to_simulate]) / p.theta) * (S_u / p.N)
    print('SIUR')
    print(round(R_0_u,2))
    print([round(min(R_eff_u),2),round(max(R_eff_u),2)])
    print(np.where(R_eff_u<1)[0][0])
    print(np.argmax(Inew_u[: -(p.numeric_max_age + p.extra_days_to_simulate)]))
    
    # plot parameters
    # get a list of the xticklabels (one every STEP up to & including args.maxtime)
    xtickets = [i for i in range(150 + 1) if i % STEP == 0]
    # For each xticket we want a label with the months so:
    # Starting with the starting date add the next month to the list until we have as many as xtickets
    while len(xticklabels) < len(xtickets):
        xticklabels.append((xticklabels[-1].replace(day=1) + timedelta(days=31)).replace(day=1))
    xticklabels = [m.strftime("%b%y") for m in xticklabels]  # Format the xticklabels

    # Time points (in days)
    t = np.linspace(0, 150-1, 150)
    np.linspace(p.day_1st_death_after_150220, p.maxtime, (p.maxtime - p.day_1st_death_after_150220) + 1)
    

    
    # Plots comparing models
    fig, (ax2, ax, ax3, ax4) = plt.subplots(4, 1, figsize=(8.0, 7.0))
    ax.plot(t, Iday_a[0,:150], label = Model2_label)
    ax.plot(t, Inew_i[:150], label='SIRD')
    ax.plot(t, Inew_u[:150], label='SIURD')
    ax.legend()
    ax.set_title('Daily new infections')
    ax.set_ylabel('Number')
    ax.set_xticks(xtickets)
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.plot(t, D_a[:150], label = Model2_label)
    ax2.plot(t, D_noise[:150],'b.', label = Model2_label+'+NB')
    #ax2.plot(t, D_i[:150], label='SIRD')
    #ax2.plot(t, D_u[:150], label='SIURD')
    ax2.legend()
    ax2.set_title('Daily deaths')
    ax2.set_ylabel('Number')
    ax2.set_xticks(xtickets)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.plot(t, Itot_a[:150], label = Model2_label)
    ax3.plot(t, I_i[:150], label='SIRD')
    ax3.plot(t, I_u[:150] + N_u[:-(p.numeric_max_age + p.extra_days_to_simulate)], label='SIURD')
    ax3.legend()
    ax3.set_title('Daily active infections')
    ax3.set_ylabel('Number')
    ax3.set_xticks(xtickets)
    ax3.grid(True)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4.plot(R_eff_a[:150]  , label = Model2_label + ' R_0 = ' + str( round(R_0_a, 2)))
    ax4.plot(R_eff_i[:150], label = 'SIRD R_0 = ' + str( round(R_0_i, 2)))
    ax4.plot(R_eff_u[:150], label = 'SIURD R_0 = ' + str( round(R_0_u, 2)))
    ax4.legend()
    ax4.set_xlabel('Date')
    ax4.set_title(r'R$_{\rm{effective}}$')
    ax4.set_xticks(xtickets)
    ax4.set_xticklabels(xticklabels)
    ax4.grid(True)
    plt.tight_layout()
    

    # Save to file if path is provided
    if args.outputnumbers is not None:
        S_a = S_a * np.ones((1, S_a.size))
        R_a = R_a * np.ones((1, S_a.size))
        D_a = D_a * np.ones((1, S_a.size))
        D_noise = D_noise * np.ones((1, S_a.size))
        Itot_a = Itot_a * np.ones((1, S_a.size))
        save_data = np.vstack((S_a, Itot_a, R_a, D_a, D_noise, Iday_a))
        save_data = save_data.T
        save_np_to_file(save_data, args.outputnumbers)

    plt.show()

