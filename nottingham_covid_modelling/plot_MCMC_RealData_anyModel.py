import os
import argparse

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib.data import DataLoader
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING
from nottingham_covid_modelling.lib.equations import  get_model_SIUR_solution, get_model_solution, get_model_SIR_solution, get_model_SEIUR_solution
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix, get_file_name_suffix_anymodel, get_parameter_names, parameter_to_optimise_list


MODEL_FUNCTIONS ={'SItD':get_model_solution, 'SIR': get_model_SIR_solution, 'SIRDeltaD': get_model_SIR_solution, 'SIUR':get_model_SIUR_solution, \
'SEIUR':get_model_SEIUR_solution}



def plot_mcmc():
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
        print("Fitting full parameters, any subset of parameters will be ignored. \nIf you want to fit only some parameters change -partial and list -pto to fit ")
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

    # filename = filename + '-PopulationMCMC' #HaarioBardenetACMC EmceeHammerMCMC PopulationMCMC RaoBlackwellACMC
    folder = args.out_mcmc
    saveas = os.path.join(MODULE_DIR, folder, filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', nchains)
    logpdfs = pints.io.load_samples(saveas + '-logpdf.csv', nchains) # file contains log-posterior, log-likelihood, log-prior 
    chains = np.array(chains)
    logpdfs = np.array(logpdfs)

    model_func = MODEL_FUNCTIONS[ModelName]
    LL = noise_model(p, data_D, parameters_to_optimise, model_func=model_func)

    # Discard burn in
    burn_in = args.burn_in
    chains = chains[:, burn_in:, :]
    logpdfs = logpdfs[:, burn_in:, :-1]
    print('Burn-in period = ' + str(args.burn_in) + ' iterations')
    
    print('Plots for log-likelihoods... ')
    pints.plot.trace(logpdfs, parameter_names=['log-posterior', 'log-L'])
    if not args.show_plots:
        plt.savefig(saveas + '_TraceLikelihoods.png')

    
    print('Plots for the parameter estimates...')
    pints.plot.trace(chains, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_chains.png')

    # Apply thinning
    chains = chains[:, ::10, :]
    logpdfs = logpdfs[:, ::10, :]

     # # Look at histograms
    pints.plot.histogram(chains, kde=True, n_percentiles=99, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_histograms.png')

    # Specific to the selected chain:
    pints.plot.pairwise(chains[args.chain - 1], kde=True, n_percentiles=99, parameter_names=parameter_names)
    if not args.show_plots:
        plt.savefig(saveas + '_pairwise_posteriors_chain' + str(args.chain) + '.png')

    print('Chains summary:')
    mcmc_summary = pints.MCMCSummary(chains, parameter_names=parameter_names).summary()
    print(*mcmc_summary, sep='\n')
    mcmc_logL_summary = pints.MCMCSummary(logpdfs, parameter_names=['log-posterior', 'log-L']).summary()
    print(*mcmc_logL_summary, sep='\n')

    # Show graphs
    if args.show_plots:
        plt.show()

