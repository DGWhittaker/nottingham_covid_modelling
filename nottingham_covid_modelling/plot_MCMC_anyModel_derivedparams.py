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
from scipy.stats import gaussian_kde

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

def plot_mcmc_derivedparams():
    parser = argparse.ArgumentParser()

    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("--derive", action='store_true', help="whether to derive params or not", default=False)
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

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]

    # Compare sampled posterior parameters with real data
    np.random.seed(100)

    posterior_samples = []

    values = []
    R0_samples, Rmin_samples, Rl1_samples, maxI_samples = [], [], [], []
    l_alpha = len(p.alpha)

    thinning_factor = 10
    chains = chains[:, ::thinning_factor, :]
    file_append = ModelName

    debug = False

    if args.derive:
        for n, i in enumerate(chains[args.chain - 1]):
            if debug:
                if ((n+1) % thinning_factor == 0):
                    print(str("{:.2f}".format(100 * thinning_factor * (n+1)/(niter-burn_in))) + '% done...')
            paras = i
            p_dict = dict(zip(parameters_to_optimise, paras))

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
            maxI_samples.append(np.max(Inew[:-p.numeric_max_age]))

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
            R0_samples.append(R_eff[0])
            Rmin_samples.append(np.min(R_eff))
            for j, k in enumerate(R_eff):
                if k < 1:
                    Rl1_samples.append(j)
                    break

        np.save('../R0_samples_' + file_append + '.npy', R0_samples)
        np.save('../Rmin_samples_' + file_append + '.npy', Rmin_samples)
        np.save('../Rl1_samples_' + file_append + '.npy', Rl1_samples)
        np.save('../maxI_samples_' + file_append + '.npy', maxI_samples)
    else:
        R0 = np.load('../R0_samples_' + file_append + '.npy')
        Rmin = np.load('../Rmin_samples_' + file_append + '.npy')
        Rl1 = np.load('../Rl1_samples_' + file_append + '.npy')
        maxI = np.load('../maxI_samples_' + file_append + '.npy')

        R0_min, R0_max = np.min(R0), np.max(R0)
        R0x = np.linspace(R0_min, R0_max, 100)

        Rmin_min, Rmin_max = np.min(Rmin), np.max(Rmin)
        Rminx = np.linspace(Rmin_min, Rmin_max, 100)

        Rl1_min, Rl1_max = np.min(Rl1), np.max(Rl1)
        Rl1x = np.linspace(Rl1_min, Rl1_max, 6)

        maxI_min, maxI_max = np.min(maxI), np.max(maxI)
        maxIx = np.linspace(maxI_min, maxI_max, 100)

        if ModelName in {'SIR', 'SIRDeltaD', 'SIUR', 'SItD'}:
            nbins = 2
        else:
            nbins = 3

        fig = plt.figure(figsize=(8, 2))
        ax1 = fig.add_subplot(141)
        ax1.set_title(r'$\mathcal{R}_0$')
        ax1.hist(R0, bins=25, density=True, color='red')
        ax1.plot(R0x, gaussian_kde(R0)(R0x))
        ax2 = fig.add_subplot(142)
        ax2.set_title(r'min$_i\{\mathcal{R}_i\}$')
        ax2.hist(Rmin, bins=25, density=True, color='red')
        ax2.plot(Rminx, gaussian_kde(Rmin)(Rminx))
        ax3 = fig.add_subplot(143)
        ax3.set_title(r'argmin$_i\{\mathcal{R}_i<1\}$')
        ax3.hist(Rl1, bins=nbins, density=True, color='red')
        ax4 = fig.add_subplot(144)
        ax4.set_title(r'max$_i\{I_{i,1}\}$')
        ax4.hist(maxI, bins=25, density=True, color='red')
        ax4.plot(maxIx, gaussian_kde(maxI)(maxIx))
        plt.tight_layout()

        if args.show_plots:
            plt.show()
        else:
            plt.savefig('posterior_outputs_' + file_append + '.png')

