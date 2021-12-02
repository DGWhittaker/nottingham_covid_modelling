import os
import argparse
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True

from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
import pints.io
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING, POPULATION
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SIUR_difference_equations, step, solve_SEIUR_difference_equations
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous

MODEL_FUNCTIONS ={'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, \
'SEIUR':get_model_SEIUR_solution}

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
    
    parameters_to_optimise.extend(['negative_binomial_phi']) #<- this one is added in the likelihood class
    return parameters_to_optimise

def plot_mcmc_series():
    parser = argparse.ArgumentParser()

    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("--model_name", type=str, help="which model to use", choices=MODEL_FUNCTIONS.keys(), default='SIR')
    parser.add_argument("-full", "--fit_full", action='store_false', help='Whether to fit all the model parameters, or only [rho, I0, NB_phi], ', default=True)
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--syndata_num", type=int, help="Give the number of the synthetic data set you want to fit, default 1", default=1)
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
    parser.add_argument("--chain", type=int, help="which chain to use", default=1)

    args = parser.parse_args()
    FitFull = args.fit_full
    FitStep = args.fit_step
    ModelName = args.model_name
    SyntDataNum_file = args.syndata_num

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
        p.xi = 1 / (p.death_mean - p.beta_mean)

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

    filename = get_file_name_suffix(p, 'SimSItD-' + str(SyntDataNum_file) + rho_label, Noise_label + 'model-' + ModelName + \
        '_full-fit-' + str(FitFull), parameters_to_optimise)

    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)
    logpdfs = pints.io.load_samples(saveas + '-logpdf.csv', 3) # file contains log-posterior, log-likelihood, log-prior 

    chains = np.array(chains)
    logpdfs = np.array(logpdfs)

    niter = len(chains[1])

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]
    logpdfs = logpdfs[:, burn_in:, :]
    logpdfs = logpdfs[args.chain-1]
    MAP_idx = np.argmax(logpdfs[:, 0])
    MLE_idx = np.argmax(logpdfs[:, 1])
    print('Posterior mode log-posterior: ' + str(logpdfs[MAP_idx, 0]))
    print('Posterior mode log-likelihood: ' + str(logpdfs[MAP_idx, 1]))
    print('Best ever log-likelihood: ' + str(logpdfs[MLE_idx, 1]))
    print('Posterior mode log-prior: ' + str(logpdfs[MAP_idx, 2]))

    paras = chains[args.chain - 1][MLE_idx]
    p_dict = dict(zip(parameters_to_optimise, paras))

    # Compare sampled posterior parameters with real data
    np.random.seed(100)

    upper = len(chains[1])

    if ModelName != 'SItD':
        alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary=p_dict)[:-p.extra_days_to_simulate]
    else:
        store_rate_vectors(p_dict, p)
        p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary=p_dict)

    # Solve forward model
    if ModelName in {'SIR', 'SIRDeltaD'}:
        S, _, Inew, _, D = solve_SIR_difference_equations(p, p_dict, travel_data=True)
    elif ModelName == 'SItD':
        S, Iday, _, D, _ = solve_difference_equations(p, p_dict, travel_data=True)
        Inew = Iday[0, :]
    elif ModelName == 'SIUR':
        S, _, Inew, _, _, D = solve_SIUR_difference_equations(p, p_dict, travel_data=True)    
    elif ModelName == 'SEIUR':
        S, _, _, _, _, _, Inew, _, _, _, D = solve_SEIUR_difference_equations(p, p_dict, travel_data=True)

    # Calculate R_0 and R_eff
    if ModelName != 'SItD':
        rho = p_dict.get('rho', p.rho)
        Iinit = p_dict.get('Iinit1', p.Iinit1)
        theta_fit = p_dict.get('theta', p.theta)
        R_0 = (rho * p.beta * 1) / theta_fit
        R_eff = ((rho * p.beta * alpha) / theta_fit) * (S / p.N)
    else:
        R_eff = calculate_R_instantaneous(p, S, p_dict)
        R_0 = R_eff[0]

    # ('----------- data Reff -----------')
    # True simulation paramters:
    p_dict_data = dict(zip(parameter_to_optimise_list(True, True, 'SItD'), [3.203, 860, 0.2814, 31.57, 0.002]))

    # Update params
    store_rate_vectors(p_dict_data, p)
    p.alpha = step(p, lgoog_data=p.maxtime + 1  - p.numeric_max_age, parameters_dictionary=p_dict_data)
    R_eff_data = calculate_R_instantaneous(p, data_S_long, p_dict_data)
    R_0_data = R_eff_data[0]

    # xticks
    Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
    date_list = [Feb15 + timedelta(days=x) for x in range(maxtime_fit+1)]

    # time
    t = np.linspace(0, maxtime_fit - 1, maxtime_fit)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(311)
    ax1.grid(True)
    ax1.plot(t, D[:maxtime_fit], label='Posterior mode (' + ModelName + ')')
    sigma1, sigma2, sigma3 = [], [], []
    NB_phi = p_dict.get('negative_binomial_phi', p.fixed_phi)
    for k in D[:maxtime_fit]:
        sigma1.append(np.sqrt(k + NB_phi * k**2))
        sigma2.append(2 * np.sqrt(k + NB_phi * k**2))
        sigma3.append(3 * np.sqrt(k + NB_phi * k**2))
    ax1.fill_between(t, D[:maxtime_fit] - sigma3, D[:maxtime_fit] \
        + sigma3, color='dodgerblue', alpha=0.25)
    ax1.fill_between(t, D[:maxtime_fit] - sigma2, D[:maxtime_fit] \
        + sigma2, color='dodgerblue', alpha=0.5)
    ax1.fill_between(t, D[:maxtime_fit] - sigma1, D[:maxtime_fit] \
        + sigma1, color='dodgerblue', alpha=0.75)
    ax1.scatter(t[p.day_1st_death_after_150220:], data_D[p.day_1st_death_after_150220:maxtime_fit], edgecolor='red', facecolor='None', label='Synthetic data')
    ax1.legend()
    ax1.set_title('Daily deaths')
    ax1.set_ylabel('Number')
    ax1.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
    ax1.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
    ax2 = fig.add_subplot(312)
    ax2.grid(True)
    ax2.plot(t, Inew[:maxtime_fit], label=ModelName)
    ax2.plot(t, data_I[0, :maxtime_fit], color='r', label='Synthetic data')
    ax2.legend()
    ax2.set_title('Daily new infections')
    ax2.set_ylabel('Number')
    ax2.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
    ax2.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
    ax3 = fig.add_subplot(313)
    ax3.grid(True)
    ax3.plot(R_eff_data[:maxtime_fit], color='red', label = r'Synthetic data $R_0 =$ ' + str( round(R_0_data, 2)))
    ax3.plot(R_eff[:maxtime_fit], label = r'SIRD $R_0 = $' + str( round(R_0, 2)))
    ax3.legend()
    ax3.set_xlabel('Date')
    ax3.set_title(r'$\mathcal{R}$')
    ax3.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
    ax3.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])

    plt.tight_layout()

    if args.show_plots:
        plt.show()
    else:
        plt.savefig(saveas + '_series_chain' + str(args.chain) + '.png')
