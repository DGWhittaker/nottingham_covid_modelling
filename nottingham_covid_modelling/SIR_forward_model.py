import argparse
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
# Load project modules
from nottingham_covid_modelling.lib.equations import solve_SIR_difference_equations, solve_difference_equations
from nottingham_covid_modelling.lib.settings import Params
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_effective


def SIR_forward_model(skip_data_folder=True):
    # Get parameters, p
    p = Params()
    START_DATE = 'Feb15'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputgraph", type=str, default='',
                        help='Output path for saving the graph (default=\'\' i.e. the graph is not saved)')
    parser.add_argument("--startmonth", type=str, default=START_DATE,
                        help='Starting month for simulation (default=' + START_DATE + ')')
    parser.add_argument("--outputnumbers", type=str, default=None, help='Output path for saving the numbers')
    parser.add_argument("-travel", "--travel_data", action='store_true', help='Wheter to use travel data or not', default=False)
    args = parser.parse_args()

    STEP = 30  # How big is the step between labelled items on the graph
    start_date = datetime.strptime(args.startmonth, '%b%y').date()  # Get start date as date object
    xticklabels = [start_date]  # Start building xticket lables with the start date


    # Parameters to compare models:
    travel_data = args.travel_data
    rho = 2.4
    gamma = (1-p.IFR) / p.recovery_mean # 0.2
    zeta = p.IFR / p.death_mean #0.002
    theta = gamma + zeta
    phi = zeta / theta
    Iinit1 = 1000
    
    if travel_data:
        p.numeric_max_age = 35
        p.extra_days_to_simulate = 10
        p.maxtime = 310 - p.numeric_max_age - p.extra_days_to_simulate
        p.alpha = np.ones(p.maxtime)
        # Dom's fit UK spline December:
        p.lockdown_baseline = 0.2042884852266899
        p.lockdown_fatigue = 0.004907334933161455
        p.lockdown_offset = 34.450147247864166
        p.lockdown_rate = 0.15917289268970847
    
    p.max_infected_age = p.maxtime +1
    
    # Simulate baseline model
    default_params = {'rho': rho, 'Iinit1': Iinit1}
    p.beta = 1 * np.ones((1, p.max_infected_age))
    p.gamma = gamma * np.ones((1, p.max_infected_age))
    p.zeta = zeta * np.ones((1, p.max_infected_age))
    
    S_a, I_a, R_a, D_a, Itot_a = solve_difference_equations(p, default_params, travel_data)

    # Simulate basic SIR model
    default_params_SIR = {'rho': rho, 'Iinit1': Iinit1}
    p.IFR = phi
    #p.gamma = gamma
    #p.zeta = zeta
    p.theta = theta
    p.DeltaD = 0
    S_s, I_s, Inew_s, R_s, D_s = solve_SIR_difference_equations(p, default_params_SIR, travel_data)
    
    
    # Differences
    S_diff = S_a - S_s
    I_diff = Itot_a - I_s
    R_diff = R_a - np.cumsum(R_s)
    D_diff = D_a - D_s
    
    
    # plot parameters
    # get a list of the xticklabels (one every STEP up to & including args.maxtime)
    xtickets = [i for i in range(p.maxtime + 1) if i % STEP == 0]
    # For each xticket we want a label with the months so:
    # Starting with the starting date add the next month to the list until we have as many as xtickets
    while len(xticklabels) < len(xtickets):
        xticklabels.append((xticklabels[-1].replace(day=1) + timedelta(days=31)).replace(day=1))
    xticklabels = [m.strftime("%b%y") for m in xticklabels]  # Format the xticklabels

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    np.linspace(p.day_1st_death_after_150220, p.maxtime, (p.maxtime - p.day_1st_death_after_150220) + 1)
    
    
    # Plot comparing both models
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.0))
    #ax.plot(t, S_s, label='Susceptible')
    ax.plot(t, Inew_s, label='Daily I')
    ax.plot(t, R_s, label='Daily R')
    ax.plot(t, D_s, label='Daily D')
    ax.legend()
    ax.set_ylabel('SIR sim numbers')
    ax.set_xticks(xtickets)
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    #ax2.plot(t, S_a, label='Susceptible')
    ax2.plot(t, I_a[0,:], label='Daily I')
    ax2.plot(t, np.append(R_a[0], -R_a[0:-1] + R_a[1:]), label='Daily R')
    ax2.plot(t, D_a, label='Daily D')
    ax2.legend()
    ax2.set_ylabel('Model-2 sim numbers')
    ax2.set_xticks(xtickets)
    ax2.grid(True)
    ax2.set_xticklabels(xticklabels)
    plt.tight_layout()
    
    
    # Plot differences between the models
    fig = plt.figure(figsize=(8.0, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(t, S_diff, label='Susceptible')
    ax.plot(t, I_diff, label='Active I')
    ax.plot(t, R_diff, label='Cumulative R')
    ax.plot(t, D_diff, label='Daily D')
    ax.legend()
    ax.set_ylabel('Model 2 - SIR')
    ax.set_xticks(xtickets)
    ax.grid(True)
    ax.set_xticklabels(xticklabels)
    plt.tight_layout()

    # plot tanh spline if travel data on
    #ax.plot( tanh_spline(p,len(p.alpha),default_params), label = 'tanh')
    


    if args.outputgraph != '':
        os.makedirs(os.path.dirname(args.outputgraph), exist_ok=True)  # Make sure the path exists
        plt.savefig(args.outputgraph)

    # Save to file if path is provided
    if args.outputnumbers is not None:
        save_data = np.append(S_s, I_s)
        save_data = np.append(save_data, R_s)
        save_data = np.append(save_data, D_s)
        save_data = np.append(save_data, Itot_s)
        save_np_to_file(save_data, args.outputnumbers)

    plt.show()

