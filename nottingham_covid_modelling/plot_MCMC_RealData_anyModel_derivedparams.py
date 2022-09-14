import os
import argparse
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True

from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib.data import DataLoader
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING, POPULATION
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SIUR_difference_equations, step, solve_SEIUR_difference_equations
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix, get_file_name_suffix_anymodel, get_parameter_names, parameter_to_optimise_list
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
from scipy.stats import gaussian_kde

MODEL_FUNCTIONS ={'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, \
'SEIUR':get_model_SEIUR_solution}

def plot_mcmc_derivedparams():
    parser = argparse.ArgumentParser()

    parser.add_argument("--show_plots", action='store_true', help="whether to show plots or not", default=False)
    parser.add_argument("--model_name", type=str, help="which model to use", choices=MODEL_FUNCTIONS.keys(), default='SIR')
    parser.add_argument("--out_mcmc", type=str, help="folder to read/store mcmc runs files in, default: ./out-mcmc-SupTable2",
                        default= 'out-mcmc-SupTable2')
    parser.add_argument("-partial", "--fit_partial", action='store_true', help='Whether to fit a subset of the model  parameters (see \"-pto\"), ', default=False)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str,  \
                        help="If \"--fit_full\" is not given, select which parameters to optimise, e.g. -pto rho Iinit1 eta. This \
                        flag do not include the step parameters (see \"-fitstep\").\
                        OPTIONS FOR EACH MODEL ------ (1)SIR: rho Iinit1 theta; \
                        (2)SIRDeltaD: rho Iinit1 theta DeltaD; (3)SIUR: rho Iinit1 theta xi;  \
                        (4)SEIUR: rho Iinit1 theta eta xi; (5)SItD: rho Iinit1")
    parser.add_argument("--fixed_eta", type=float, help="value of eta. If eta will be fitted this value is ignored. Default value is the best fit for SEIUR model to clinical data:  1/2.974653", default = 0.3362)#0.1923,
    parser.add_argument("--fixed_theta", type=float, help="value of theta. If theta in fitted params, this value is ignored.  Default value is the best fit for SEIUR model to clinical data: 1/2.974653", default=0.3362)       
    parser.add_argument("-fitstep", "--fit_step", action='store_false', help='Whether to fit step parameters', default=True)
    parser.add_argument("--burn_in", help="number of MCMC iterations to ignore",
                        default=25000, type=int)
    parser.add_argument("--chain", type=int, help="which chain to use", default=1)
    parser.add_argument("--informative_priors", action='store_true', help='Whether to use informative priors', default=False)
    parser.add_argument("--chains5", action='store_true', help='Whether to do 5 instead of 3 chains', default=False)


    args = parser.parse_args()
    FitFull = not args.fit_partial
    params_fromOptions = args.params_to_optimise
    FitStep = args.fit_step
    ModelName = args.model_name
    Fixed_eta = args.fixed_eta
    Fixed_theta = args.fixed_theta

    if FitFull:
        print("Fitting full parameters, any subset of paramertes will be ignored. \nIf you want to fit only some parameters change -partial and list -pto to fit ")
    else: 
        if params_fromOptions is None:
            parser.error("If -partial, -pto is required. You did not specify -pto.")

    fixed_params_tag = "" 
    if not FitFull and ModelName == 'SEIUR':
        if 'eta' not in  params_fromOptions:
            fixed_params_tag = fixed_params_tag + '_fixedEta-' + str(Fixed_eta)
    if not FitFull and ModelName != 'SItD':
        if 'theta' not in  params_fromOptions:
            fixed_params_tag = fixed_params_tag + '_fixedTheta-' + str(Fixed_theta)

    if args.chains5:
        nchains = 5
    else:
        nchains = 3
        
    # For reproducibility:
    np.random.seed(100)
    burn_in = args.burn_in
    thinning_factor = 10

    # For reproducibility:
    np.random.seed(100)

    # Get parameters, p
    p = Params()
    # Fixed for  UK google and ONS data:  
    p.IFR = 0.00724 # UK time
    p.n_days_to_simulate_after_150220 = 150
    p.five_param_spline = False
    p.N = 59.1e6

    p.numeric_max_age = 35
    p.extra_days_to_simulate = 10
    p.square_lockdown = True
    if not args.informative_priors:
        p.flat_priors = True

    # Storing parameter values. Default values are given by  best the fit from Kirsty
    if ModelName != 'SItD':
        p.theta = Fixed_theta
        p.eta = Fixed_eta

    if ModelName == 'SEIUR':
        p.xi = 1 / 11.744308
    else:
        p.xi = 1 / (18.69 - 1 - 5.2)

    if ModelName == 'SIRDeltaD':
        p.DeltaD = 18.69 - 1 - 5.2
    else:
        p.DeltaD = 0

    # define the params to optimize
    parameters_to_optimise = parameter_to_optimise_list(FitFull, FitStep, ModelName, params_fromOptions)
    parameter_names = get_parameter_names(parameters_to_optimise, FitStep)

    # Get noise model
    noise_model = NOISE_MODEL_MAPPING['NegBinom']

    # Get real data
    print('Getting data...')
    data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', 'data', 'archive', 'current'))
    data_D = data.daily_deaths

    if FitFull:
        filename1 = get_file_name_suffix(p, 'ONS-UK', 'model-' + ModelName + '_full-fit-' + str(FitFull), parameters_to_optimise)
    else:
        filename1 = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', ModelName , parameters_to_optimise)
    filename = filename1 + fixed_params_tag
    folder = args.out_mcmc
    filename_epiparams = os.path.join(MODULE_DIR, folder, filename + '_epiParams_chain_' + str(args.chain) + '_burnIn_' + str(burn_in) + '_thinning_' + str(thinning_factor) + '.npy')

    if os.path.exists(filename_epiparams):
        epiparams = np.load(filename_epiparams)
        R0 = epiparams[0,:]
        Rmin = epiparams[1,:]
        Rl1 = epiparams[2,:]
        maxI = epiparams[3,:]
    else:
        saveas = os.path.join(MODULE_DIR, folder, filename)
        chains = pints.io.load_samples(saveas + '-chain.csv', nchains)
        chains = np.array(chains)
        logpdfs = pints.io.load_samples(saveas + '-logpdf.csv', nchains) # file contains log-posterior, log-likelihood, log-prior 
        logpdfs = np.array(logpdfs)
        niter = len(chains[1])

        # Discard burn in
        chains = chains[:, burn_in:, :]
        logpdfs = logpdfs[:, burn_in:, :-1]
        chains = chains[:, ::thinning_factor, :]
        logpdfs = logpdfs[:, ::thinning_factor, :]
        R0, Rmin, Rl1, maxI = [], [], [], []
        debug = True
        
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
                S, _, _, Enew, _, _, _, _, _, _, D = solve_SEIUR_difference_equations(p, p_dict, travel_data=True)
                Inew = Enew
            maxI.append(np.max(Inew[:-p.numeric_max_age]))

            # Calculate R_0 and R_eff
            if ModelName != 'SItD':
                rho = p_dict.get('rho', p.rho)
                Iinit = p_dict.get('Iinit1', p.Iinit1)
                theta_fit = p_dict.get('theta', p.theta)
                R_0 = (rho * 1) / theta_fit
                R_eff = ((rho * alpha) / theta_fit) * (S / p.N)
            else:
                R_eff = calculate_R_instantaneous(p, S, p_dict)
                R_0 = R_eff[0]
            R0.append(R_0)
            Rmin.append(np.min(R_eff))
            for j, k in enumerate(R_eff):
                if k < 1:
                    Rl1.append(j)
                    break
        
        save_data = np.vstack((R0, Rmin, Rl1, maxI))
        save_np_to_file(save_data, filename_epiparams)
        
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
        plt.savefig(os.path.join(MODULE_DIR, folder, filename + '_posterior_outputs.png'))
 
