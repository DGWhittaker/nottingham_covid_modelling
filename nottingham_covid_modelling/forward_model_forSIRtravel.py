import argparse
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
# Load project modules
from nottingham_covid_modelling.lib.equations import solve_difference_equations, store_rate_vectors, tanh_spline
from nottingham_covid_modelling.lib.settings import Params

from nottingham_covid_modelling.lib.ratefunctions import calculate_R_effective

def forward_model_forSIRtravel(skip_data_folder=True):
    # Get parameters, p
    p = Params()
    START_DATE = 'Feb15'
    I0 = 1000  # Initial number of infecteds
    rho = 2.4  # Approximate reproduction number

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputgraph", type=str, default='',
                        help='Output path for saving the graph (default=\'\' i.e. the graph is not saved)')
    parser.add_argument("--outputnumbers", type=str, default=None, help='Output path for saving the numbers')
    parser.add_argument("-travel", "--travel_data", action='store_true', help='Wheter to use travel data or not', default=False)
    parser.add_argument("-step", "--square_lockdown", action='store_true', help='Wheter to use square_lockdown approximation or not. It will be ignored it travel data is FALSE', default=False)
    args = parser.parse_args()
    

    # Input parameters
    default_params = {'rho': rho, 'Iinit1': I0}

    # Simulate baseline model
    store_rate_vectors(default_params, p)
    # to add travel data
    travel_data = args.travel_data
    # to add step fit
    if travel_data:
        p.square_lockdown = args.square_lockdown
    else:
        p.square_lockdown = False
    
    # parameters based on Google UK data wiht ONS data
    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    p.maxtime = 310 - p.numeric_max_age - p.extra_days_to_simulate
    p.alpha = np.ones(p.maxtime)
    p.lockdown_baseline = 0.2042884852266899 #3.6
    p.lockdown_fatigue = 0.004907334933161455
    p.lockdown_offset = 34.450147247864166
    p.lockdown_rate = 0.15917289268970847

    S, I, R, D, Itot = solve_difference_equations(p, default_params, travel_data)
    
    print(np.sum(I[0,:-1]))
    print(np.sum(I[0,:]))
    print(Itot[-1])
    print(np.sum(D))
    print(R[-1])
    
    
    STEP = 30  # How big is the step between labelled items on the graph
    start_date = datetime.strptime(START_DATE, '%b%y').date()  # Get start date as date object
    xticklabels = [start_date]  # Start building xticket lables with the start date

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

    
    # Plot detailed figure including death rate per day and infection profiles for individual days of infectiousness
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8.0, 5.0))
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, Itot, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.legend()
    ax.set_ylabel('Number')
    ax.set_xticks(xtickets)
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.plot(t, D)
    ax2.set_ylabel('Daily deaths')
    ax2.set_xticks(xtickets)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.plot(t, I[0, :], label='1st day of infection')  # Number of people on 1st day of infection
    ax3.plot(t, I[10, :], label='11th day of infection')  # Number of people on 11th day of infection
    ax3.plot(t, I[20, :], label='21st day of infection')  # Number of people on 21st day of infection
    ax3.legend()
    ax3.set_ylabel('Number')
    ax3.set_xticks(xtickets)
    ax3.grid(True)
    ax3.set_xticklabels(xticklabels)
    plt.tight_layout()

    # plot the initial condition
    age = np.linspace(0, p.max_infected_age - 1, p.max_infected_age)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.0))
    ax.plot(age, I[:,0], label='I(0)')
    ax.legend()
    ax.set_ylabel('Infected')
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.plot(age, I[:,0] / Itot[0], label='Pseudo st st dist')
    ax2.legend()
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Age')
    ax2.grid(True)
    plt.tight_layout()


    if args.outputgraph != '':
        os.makedirs(os.path.dirname(args.outputgraph), exist_ok=True)  # Make sure the path exists
        plt.savefig(args.outputgraph)

    # Save to file if path is provided
    if args.outputnumbers is not None:
        S = S * np.ones((1, S.size))
        R = R * np.ones((1, S.size))
        D = D * np.ones((1, S.size))
        Itot = Itot * np.ones((1, S.size))
        save_data = np.vstack((S, Itot, R, D, I))
        #save_data = np.append(save_data, R)
        #save_data = np.append(save_data, D)
        #save_data = np.append(save_data, Itot)
        save_data = save_data.T
        save_np_to_file(save_data, args.outputnumbers)

    plt.show()
