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

def hazards_exploration(skip_data_folder=True):
    # Get parameters, p
    p = Params()
    START_DATE = 'Mar20'
    
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

    # Hazards test parameters:
    # 1) Old fits parameters (same for recovery)
    #   self.death_mean = 17.8 and self.death_var = 71.2
    #   Translate to p.death_N_NB = 5.188 and p.death_p_NB = 0.236
    # 2) p.recovery_--_NB = p.death_--_NB
    # 3 and 4) When changing the mean of eaither to be 80 days:
    #   p.---_N_NB = 1195 and p.---_p_NB = 0.94
    # 5) Change to fater recovery: p.recovery_N_NB = 3.12  p.recovery_p_NB = 0.18
    test_N_NB = [p.death_N_NB, p.recovery_N_NB, 5.188, 5.188, p.death_N_NB, p.death_N_NB, p.death_N_NB, 1195, 1195,  p.recovery_N_NB, p.death_N_NB, 3.12]
    test_p_NB = [p.death_p_NB, p.recovery_p_NB, 0.236, 0.236, p.death_p_NB, p.death_p_NB, p.death_p_NB, 0.94, 0.94, p.recovery_p_NB, p.death_p_NB, 0.18]


    # Simulate baseline model
    S, I, R, D = np.zeros((6, p.maxtime + 1)), np.zeros((6, p.maxtime + 1)), np.zeros((6, p.maxtime + 1)), np.zeros((6, p.maxtime + 1))
    
    for i in range(6):
        p.death_N_NB = test_N_NB[2 * i]
        p.death_p_NB = test_p_NB[2 * i]
        p.recovery_N_NB = test_N_NB[(2 * i) + 1]
        p.recovery_p_NB = test_p_NB[(2 * i) + 1]
        store_rate_vectors(p)
        S[i, :], Iday, R[i, :], D[i, :], I[i, :] = solve_difference_equations(p, {} , travel_data=False)

    
    
    fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8.0, 7.0))
    ax.plot(t, S[0, :], label='Default')
    ax.plot(t, S[1, :], '--', label='Old equivalent')
    ax.plot(t, S[2, :], '-.', label='R time = D time')
    ax.plot(t, S[3, :], '-.', label='R mean = 70')
    ax.plot(t, S[4, :], '-.', label='D mean = 70')
    ax.plot(t, S[5, :], '-.', label='R mean = 10')
    ax.legend()
    ax.set_ylabel('S')
    ax.set_xticks(xtickets)
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.plot(t, I[0, :], label='Default')
    ax2.plot(t, I[1, :], '-.', label='Old equivalent')
    ax2.plot(t, I[2, :], '-.', label='R time = D time')
    ax2.plot(t, I[3, :], '-.', label='R mean = 70')
    ax2.plot(t, I[4, :], '-.', label='D mean = 70')
    ax2.plot(t, I[5, :], '-.', label='R mean = 10')
    ax2.set_ylabel('I')
    ax2.set_xticks(xtickets)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.plot(t, R[0, :],  label='Default')
    ax3.plot(t, R[1, :], '-.', label='Old equivalent')
    ax3.plot(t, R[2, :], '-.', label='R time = D time')
    ax3.plot(t, R[3, :], '-.', label='R mean = 70')
    ax3.plot(t, R[4, :], '-.', label='D mean = 70')
    ax3.plot(t, R[5, :], '-.', label='R mean = 10')
    ax3.set_ylabel('R')
    ax3.set_xticks(xtickets)
    ax3.grid(True)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4.plot(t, D[0, :], label='Default')
    ax4.plot(t, D[1, :], '-.', label='Old equivalent')
    ax4.plot(t, D[2, :], '-.', label='R time = D time')
    ax4.plot(t, D[3, :], '-.', label='R mean = 70')
    ax4.plot(t, D[4, :], '-.', label='D mean = 70')
    ax4.plot(t, D[5, :], '-.', label='R mean = 10')
    ax4.set_ylabel('D (daily)')
    ax4.set_xticks(xtickets)
    ax4.grid(True)
    ax4.set_xticklabels(xticklabels)
    plt.tight_layout()

    plt.show()
