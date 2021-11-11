import os

import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING, POPULATION, get_parser
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, tanh_spline, step, store_rate_vectors
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_effective, calculate_R_instantaneous, calculate_R0
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from scipy.stats import nbinom, gamma

def plot_cma():

    parser = get_parser()

    parser.add_argument("-p", "--plot_fit", action='store_true', help="whether to plot best fit or not", \
                        default=False)
    parser.add_argument("-c", "--country_str", type=str, help="which country to use",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--cmaes_fits", type=str, help="folder to store cmaes fits files in, default: ./cmaes_fits",
                        default=os.path.join(MODULE_DIR, 'cmaes_fits'))
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str, required=True, \
                        help="which parameters to optimise, e.g. -pto rho Iinit1 lockdown_baseline")
    parser.add_argument("--plot_SIRD", action='store_true', help="whether to plot S, I, R, D or not", \
                        default=False)
    parser.add_argument("--simulate_full", action='store_true',
                        help="whether to use all Google data (default is 150 days)", default=False)
    parser.add_argument("--alpha1", action='store_true',
                        help="whether or not to do alpha=1 simulation", default=False)
    parser.add_argument("--optimise_likelihood", action='store_true',
                        help="whether to optimise log-likelihood instead of log-posterior", default=False)

    args = parser.parse_args()
    if args.ons_data and args.country_str != 'United Kingdom':
        parser.error('Can only use ONS data in combination with country United Kingdom')

    plot_infectiousness_profile = False
    plot_death_distributions = False
    plot_recovery_distributions = False

    # Get parameters, p
    p = Params()
    if args.country_str in IFR_dict:
        p.IFR = IFR_dict[args.country_str]
    p.simple = args.simple
    p.fix_phi = args.fix_phi
    p.fixed_phi = args.fixed_phi
    p.fix_sigma = args.fix_sigma
    p.square_lockdown = args.square
    if p.simple:
        print('Using simple rates...')
    else:
        print('Using gamma distribution rates...')

    p.extra_days_to_simulate = 10
    if not args.simulate_full:
        p.n_days_to_simulate_after_150220 = 150

    # Get Google travel and deaths data
    print('Getting data...')
    data = DataLoader(args.ons_data, p, args.country_str, data_dir=args.datafolder)
    
    parameters_to_optimise = args.params_to_optimise

    # Get noise model
    noise_str = args.noise_model
    noise_model = NOISE_MODEL_MAPPING[noise_str]

    # alpha = 1 scenario
    p.alpha1 = args.alpha1
    if p.alpha1:
        assert p.square_lockdown == True, "Must use --square input for alpha=1 simulation"
        print('Using alpha = 1!!!')
        p.lockdown_baseline = 1.0

    # Get likelihood function
    LL = noise_model(p, data.daily_deaths, parameters_to_optimise)

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

    # Set up optimisation
    folder = args.cmaes_fits

    filename = os.path.join(folder, get_file_name_suffix(p, data.country_display, noise_str, parameters_to_optimise))
    # filename = filename + '-alejandra-data'
    fit_synthetic_data = False
    if fit_synthetic_data:
        filename = filename + '-test-phi0.001'
    optimise_likelihood = args.optimise_likelihood
    if optimise_likelihood:
        print('WARNING: optimising log-likelihood, not MAP!!!')
        filename = filename + '-optimiseLL'
    obtained_parameters = np.loadtxt(filename + '.txt')
    # obtained_parameters = [3.203,
    #                         860,
    #                         0.2814,
    #                         31.57,
    #                         0.002]
    p_dict = dict(zip(LL.parameter_labels, obtained_parameters))
    IFR = p_dict.get('IFR', p.IFR)
    if IFR is not p.IFR:
        IFR_label = 'inferred'
    else:
        IFR_label = 'fixed'

    label = ''
    for l in p_dict:
        label = label + str(l) + ': ' + str('%.4g' % p_dict.get(l)) + '\n'

    # Calculate beta, gamma and zeta vector rates.
    print('Storing fixed parameters...')
    store_rate_vectors(p_dict, p)

    # Simulate optimised model
    cS, cI, cR, cD, cItot = solve_difference_equations(p, p_dict, travel_data=True)
    fig = plt.figure(figsize=(9, 7))
    ax1 = fig.add_subplot(311)

    ax1.plot(t[:-p.numeric_max_age], cD[:-p.numeric_max_age], label=label)

    # np.save('herd_immunity_deaths.npy', cD)

    # import csv
    # with open('model_data.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["model","data"])
    #     writer.writerows(zip(cD[p.day_1st_death_after_150220:-(p.numeric_max_age + p.extra_days_to_simulate)], data.daily_deaths))

    generate_synthetic_data = False
    if fit_synthetic_data or generate_synthetic_data:
        f = []
        if generate_synthetic_data:
            NB_phi = 0.001
        else:
            NB_phi = p_dict['negative_binomial_phi']
        n = 1 / NB_phi
        for k in cD[:-p.numeric_max_age]:
            pr = 1 / (1 + k * NB_phi)
            f.append(nbinom.rvs(n, pr))
        ax1.plot(t[p.day_1st_death_after_150220:-(p.numeric_max_age + p.extra_days_to_simulate)], \
            f[p.day_1st_death_after_150220:-p.extra_days_to_simulate], linewidth=2, color='limegreen', \
            label='Synthetic data')
        if generate_synthetic_data:
            np.save('syn_data_phi' + str(NB_phi) + '.npy', f[p.day_1st_death_after_150220:-p.extra_days_to_simulate])  
    
    plot_variance = True
    if plot_variance:
        sigma1, sigma2, sigma3 = [], [], []
        NB_phi = p_dict.get('negative_binomial_phi', p.fixed_phi)
        sigma = p_dict.get('gaussian_noise_sigma', p.fixed_sigma)
        for k in cD[:-p.numeric_max_age]:
            if noise_model == NegBinom_LogLikelihood:
                sigma1.append(np.sqrt(k + NB_phi * k**2))
                sigma2.append(2 * np.sqrt(k + NB_phi * k**2))
                sigma3.append(3 * np.sqrt(k + NB_phi * k**2))
            elif noise_model == Gauss_LogLikelihood:
                sigma1.append(sigma)
                sigma2.append(2 * sigma)
                sigma3.append(3 * sigma)
            else:
                sigma1.append(np.sqrt(k))
                sigma2.append(2 * np.sqrt(k))
                sigma3.append(3 * np.sqrt(k))
        ax1.fill_between(t[:-p.numeric_max_age], cD[:-p.numeric_max_age] - sigma3, cD[:-p.numeric_max_age] \
            + sigma3, color='dodgerblue', alpha=0.25)
        ax1.fill_between(t[:-p.numeric_max_age], cD[:-p.numeric_max_age] - sigma2, cD[:-p.numeric_max_age] \
            + sigma2, color='dodgerblue', alpha=0.5)
        ax1.fill_between(t[:-p.numeric_max_age], cD[:-p.numeric_max_age] - sigma1, cD[:-p.numeric_max_age] \
            + sigma1, color='dodgerblue', alpha=0.75)
    ax1.scatter(t_daily, data.daily_deaths, edgecolor='red', facecolor='None', label='Observed data (' + data.country_display + ')')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_ylabel('Daily deaths')
    ax1.set_title('Optimised model - ' + noise_str + ' log-likelihood')
    ax1.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax1.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    plt.grid(True)

    lgoog_data = len(p.alpha)
    if p.square_lockdown == True:
        p.alpha = step(p, lgoog_data=lgoog_data, parameters_dictionary=p_dict)[:-p.numeric_max_age]
    else:
        p.alpha = tanh_spline(p, lgoog_data=lgoog_data, parameters_dictionary=p_dict)[:-p.numeric_max_age]

    d_vec = np.linspace(0, p.weekdays - 1, p.weekdays)
    d_vec_weekdays = np.copy(d_vec)
    d_vec_weekdays = [x for i, x in enumerate(d_vec_weekdays) if not (
        (i % 7 == 0) or (i % 7 == 1))]

    ax2 = fig.add_subplot(312)
    ax2.plot(p.alpha[:-(p.numeric_max_age + p.extra_days_to_simulate)], label='Inferred spline')
    ax2.scatter(d_vec_weekdays, p.alpha_weekdays, edgecolor='orange', facecolor='None', \
        label='Google mobility data')
    ax2.legend()
    ax2.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax2.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax2.set_ylabel(r'$\alpha$')
    plt.grid(True)

    ax3 = fig.add_subplot(313)
    rho = calculate_R0(p, p_dict)
    R_eff = calculate_R_instantaneous(p, cS, p_dict)

    maxI = np.max(cI[0, :-p.numeric_max_age])
    R0 = R_eff[0]
    Rmin = np.min(R_eff)
    for j, k in enumerate(R_eff):
        if k < 1:
            Rl1 = j
            break

    print('maxI: ' + str(maxI))
    print('R0: ' + str(R0))
    print('Rmin: ' + str(Rmin))
    print('Rl1: ' + str(Rl1))

    ax3.plot(rho, label=r'R$_0$')
    ax3.plot(R_eff, label=r'R$_{\rm{effective}}$')
    ax3.legend()
    ax3.set_xlabel('Date')
    ax3.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax3.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    plt.grid(True)

    plt.tight_layout()

    if args.plot_fit:
        print('Plotting best result...')
        plt.show()
    else:
        plt.savefig(filename + '.png')

    if args.plot_SIRD:
        cumD, cumDeaths = [], 0
        cumI, cumInfecteds = [], 0
        for i in cD:
            cumDeaths += i
            cumD.append(cumDeaths)
        for i in cI[0,:]:
            cumInfecteds += i
            cumI.append(cumInfecteds)
        approx_IFR = cumD[-p.numeric_max_age] / cumI[-p.numeric_max_age]

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(411)
        ax1.plot(t[:-p.numeric_max_age], cS[:-p.numeric_max_age], label='Susceptible')
        ax1.plot(t[:-p.numeric_max_age], cItot[:-p.numeric_max_age], label='Infected')
        ax1.plot(t[:-p.numeric_max_age], cR[:-p.numeric_max_age], label='Recovered')
        ax1.legend()
        ax1.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
        [label.set_visible(False) for label in ax1.get_xticklabels()]
        ax1.set_ylabel('Number')
        ax1.grid(True)
        ax2 = fig.add_subplot(412)
        ax2.plot(t[:-p.numeric_max_age], cI[0, :-p.numeric_max_age], label='Daily new infections')
        ax2.legend()
        ax2.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
        [label.set_visible(False) for label in ax2.get_xticklabels()]
        ax2.set_ylabel('Number')
        ax2.grid(True)
        ax3 = fig.add_subplot(413)
        ax3.plot(t[:-p.numeric_max_age], cItot[:-p.numeric_max_age], label='Daily active infections')
        ax3.legend()
        ax3.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
        [label.set_visible(False) for label in ax3.get_xticklabels()]
        ax3.set_ylabel('Number')
        ax3.grid(True)
        ax4 = fig.add_subplot(414)
        ax5 = ax4.twinx()
        ax4.set_title(r'Simulated IFR $\approx$ ' + str('%1f' % approx_IFR) + ', ' + IFR_label + ' IFR = ' \
            + str('%1f' % IFR))
        ax4.plot(t[:-p.numeric_max_age], cumI[:-p.numeric_max_age], label='Cumulative infecteds')
        ax5.plot(t[:-p.numeric_max_age], cumD[:-p.numeric_max_age], color='#ff7f0e', \
            label='Cumulative deaths')
        ax4.legend(loc='upper left')
        ax5.legend(loc='lower right')
        ax4.set_xlabel('Date')
        ax4.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
        ax4.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
        ax4.set_ylabel('Cumulative infecteds')
        ax5.set_ylabel('Cumulative deaths')
        ax4.grid(True)
        plt.tight_layout()

        if args.plot_fit:
            print('Plotting S, I, R, D...')
            plt.show()
        else:
            plt.savefig(filename + '-SIRD.png')

    for pto in parameters_to_optimise:
        if pto in {'beta_mean', 'beta_var'}:
            plot_infectiousness_profile = True
            break

    for pto in parameters_to_optimise:
        if pto in {'death_mean', 'death_dispersion'}:
            plot_death_distributions = True
            break

    for pto in parameters_to_optimise:
        if pto in {'recovery_mean', 'recovery_dispersion'}:
            plot_recovery_distributions = True
            break

    if plot_infectiousness_profile:
        days = np.linspace(0, 20, 41)

        beta_mean = p_dict['beta_mean']
        beta_var = p_dict['beta_var']
        sc = beta_var / beta_mean  # scale parameter
        a = beta_mean / sc    # shape parameter
        kbeta_mean = p.beta_mean
        kbeta_var = p.beta_var
        ksc = kbeta_var / kbeta_mean  # scale parameter
        ka = kbeta_mean / ksc    # shape parameter

        fig = plt.figure()
        plt.title('Infectiousness profile')
        plt.plot(days, gamma.pdf(days, a, loc=0, scale=sc), linewidth=2, \
            label='Inferred distribution')
        plt.plot(days, gamma.pdf(days, ka, loc=0, scale=ksc), linewidth=2, \
            label='Default distribution')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.tight_layout()

        if args.plot_fit:
            plt.show()
        else:
            plt.savefig(filename + '-infectiousness-profile.png')

    if plot_death_distributions:
        days = np.linspace(0, 120, 121)

        death_mean = p_dict['death_mean']
        death_dispersion = p_dict['death_dispersion']
        death_N_NB = 1 / death_dispersion
        death_p_NB = 1 / (1 + death_mean * death_dispersion)
        kdeath_mean = p.death_mean
        kdeath_dispersion = p.death_dispersion
        kdeath_N_NB = 1 / kdeath_dispersion
        kdeath_p_NB = 1 / (1 + kdeath_mean * kdeath_dispersion)

        fig = plt.figure()
        plt.title('Infection-to-death distribution')
        plt.plot(days, nbinom.pmf(days, death_N_NB, death_p_NB), linewidth=2, \
            label='Inferred distribution')
        plt.plot(days, nbinom.pmf(days, kdeath_N_NB, kdeath_p_NB), linewidth=2, \
            label='Default distribution')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.tight_layout()

        if args.plot_fit:
            plt.show()
        else:
            plt.savefig(filename + '-infection-to-death-distribution.png')

    if plot_recovery_distributions:
        days = np.linspace(0, 120, 121)

        recovery_mean = p_dict['recovery_mean']
        recovery_dispersion = p_dict['recovery_dispersion']
        recovery_N_NB = 1 / recovery_dispersion
        recovery_p_NB = 1 / (1 + recovery_mean * recovery_dispersion)
        krecovery_mean = p.recovery_mean
        krecovery_dispersion = p.recovery_dispersion
        krecovery_N_NB = 1 / krecovery_dispersion
        krecovery_p_NB = 1 / (1 + krecovery_mean * krecovery_dispersion)

        fig = plt.figure()
        plt.title('Infection-to-recovery distribution')
        plt.plot(days, nbinom.pmf(days, recovery_N_NB, recovery_p_NB), linewidth=2, \
            label='Inferred distribution')
        plt.plot(days, nbinom.pmf(days, krecovery_N_NB, krecovery_p_NB), linewidth=2, \
            label='Default distribution')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.tight_layout()

        if args.plot_fit:
            plt.show()
        else:
            plt.savefig(filename + '-infection-to-recovery-distribution.png')
