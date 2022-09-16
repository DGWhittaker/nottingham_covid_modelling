import os
import argparse
import glob
import shutil
from turtle import color
import cma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import nbinom, gamma
from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
import pints.io
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING, POPULATION
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SIUR_difference_equations, step, solve_SEIUR_difference_equations
from nottingham_covid_modelling.lib.settings import Params,  get_file_name_suffix, get_file_name_suffix_anymodel, get_parameter_names, parameter_to_optimise_list
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file

def profiles_SIR_DeltaD(p, j_max):
    theta = p.theta
    DeltaD = int(p.DeltaD)
    betaj = np.zeros(j_max + 1)
    pzetaj = np.zeros(j_max + 1 + DeltaD)
    betaj[1] = theta
    pzetaj[2 + DeltaD] = theta
    for i in range(1, j_max):
        betaj[i+1] = betaj[i]*(1-theta)
    for i in range(2,j_max):
        pzetaj[i + 1 + DeltaD] = pzetaj[i + DeltaD]*(1-theta)
    return betaj, pzetaj

def profiles_SIUR(p, j_max):
    theta = p.theta
    xi = p.xi
    betaj = np.zeros(j_max + 1)
    pzetaj = np.zeros(j_max + 1 )
    betaj[1] = theta
    pzetaj[3] = theta * xi
    for i in range(1, j_max):
        betaj[i+1] = betaj[i]*(1-theta)
    for i in range(3, j_max):
        pzetaj[i + 1] = (pzetaj[i] * (1 - xi)) + (theta * xi * ((1-theta)**(i + 1 - 3)))
    return betaj, pzetaj

def T_matrix_SEIUR(p):
    eta = p.eta
    theta = p.theta
    xi = p.xi
    deltas = p.IFR
    T_1 =  np.diagflat([1 - 2*eta, 1 - 2*eta, 1 - 2*theta, 1 - 2*theta, 1 - 2*xi, 1 - 2*xi, 1, 1])
    T_2 = np.diagflat([2*eta, 2*eta, 2*theta, 2*theta, 2*xi, 2*xi*(1 - deltas), 0 ], k = 1)
    T = T_1 + T_2 
    T[5,-1] = 2 * xi * deltas
    return T

def profiles_SEIUR(p, j_max):
    betaj = np.zeros(j_max+1)
    pzetaj = np.zeros(j_max+1)
    T = T_matrix_SEIUR(p)
    Ti = T
    Ti2 = np.matmul(Ti,T)
    for i in range(2, j_max):
        betaj[i+1] = p.theta*(Ti2[0,2]+Ti2[0,3])
        pzetaj[i+1] = (Ti2[0,7] - Ti[0,7]) / p.IFR
        Ti = Ti2
        Ti2 = np.matmul(Ti,T)
    return betaj, pzetaj

MODEL_Profile_FUNCTIONS ={'SIR': profiles_SIR_DeltaD, 'SIRDeltaD': profiles_SIR_DeltaD, 'SIUR':profiles_SIUR, 'SEIUR':profiles_SEIUR}


# Definiiton of parameters and  fits
folder_fits = os.path.join(MODULE_DIR, 'out-mcmc-SupTable2')

Figure_name = 'SupFigure4B.png'
fig_path = os.path.join(MODULE_DIR, 'figures', 'saved-plots')
if not os.path.exists(fig_path):
   os.mkdir(fig_path)


plt.rcParams.update({'font.size': 10})
# Definiiton of parameters and  fits

default_chain = 1
burn_in = 25000
Model_name = ['SItD', 'SIR', 'SIRDeltaD', 'SIUR', 'SEIUR', 'SEIUR']
Model_label = [r'$SI_tD$', r'$SIRD$', r'$SIRD_{\Delta_D}$', r'$SIURD$', r'$SE^2I^2U^2RD$ full', r'$SE^2I^2U^2RD$ $\eta = 1/5.2$']

# PARAMETERS
# labels for saving/plotting
travel_data = True



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
p.flat_priors = False


# Get real data
print('Getting data...')
data = DataLoader(True, p, 'United Kingdom', data_dir=os.path.join(MODULE_DIR, '..', 'data', 'archive', 'current'))
data_D = data.daily_deaths


# Initialize table + data entry
table_2_params_labels = ['$I_0$', '$\\rho$', '$t^*$' , '$\\alpha_b$', '$\\theta$', '$\eta$', '$\\xi$', '$\\Delta D$', '$\\phi$' ]
table_2_derived_labels = ['$\mathcal{R}_0$','$\min_i \{\mathcal{R}_i\}$','argmin$_i \{\mathcal{R}_i<1\}$','$\max_i \{S_i - S_{i+1} \} (\\times 1000)', '$S_{end}/N$']
table_2_params = ['Iinit1', 'rho', 'lockdown_offset' , 'lockdown_baseline', 'theta', 'eta', 'xi', 'DeltaD', 'negative_binomial_phi']
table_2_MAPS = np.zeros([6,9])
table_2_derived = np.zeros([6,5])

# Initialize plot
Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
date_list = [Feb15 + timedelta(days=x) for x in range(p.maxtime+1)]
# time
maxtime_fit = p.maxtime -(p.numeric_max_age + p.extra_days_to_simulate)
t =  np.linspace(0, maxtime_fit + 1, maxtime_fit )
t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

fig = plt.figure(figsize=(8.0, 11), dpi=200)
grid = gridspec.GridSpec(2,1)
grid0 =grid[0].subgridspec(3, 1,wspace=0.2, hspace=0.05)
grid1 =grid[1].subgridspec(2, 1, hspace=0.35)
ax = fig.add_subplot(grid0[1])
ax2 = fig.add_subplot(grid0[0])
ax4 =  fig.add_subplot(grid0[2])
ax2.plot(t_daily, data_D,'b.', label = 'Data')

params_opt_names = {}
for i, model_i in enumerate(Model_name): 
    p.eta = 0.1923
    p.theta = 1/5.2
    p.xi = 1/18
    p.DeltaD = 0
    p.alpha = np.ones(p.maxtime)
    if i<5:
        params_opt_names[model_i] = parameter_to_optimise_list(True, True, model_i, [])
        params_opt_names[model_i].extend(['negative_binomial_phi'])
        filename = filename1 = get_file_name_suffix(p, 'ONS-UK', 'model-' + model_i + '_full-fit-True', params_opt_names[model_i])
    else:
        params_opt_names[model_i+'_fixEta'] = parameter_to_optimise_list(False, True, model_i,['rho', 'Iinit1', 'theta', 'xi'])
        filename = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', model_i , params_opt_names[model_i+'_fixEta'])
        filename = filename + '_fixedEta-' + str(0.1923)
        params_opt_names[model_i+'_fixEta'].extend(['negative_binomial_phi'])
    saveas = os.path.join(folder_fits, filename)
    filename_MAPS = saveas + '-chain-'+ str(default_chain) + '-MAPS.npy'
    if os.path.exists(filename_MAPS):
        MAPS_model_i = np.load(filename_MAPS)
        print([*params_opt_names][i] + ' MAP loaded')
    else:
        chains = pints.io.load_samples(saveas + '-chain.csv', default_chain)
        logpdfs = pints.io.load_samples(saveas + '-logpdf.csv', default_chain) # file contains log-posterior, log-likelihood, log-prior 
        chains = np.array(chains)
        logpdfs = np.array(logpdfs)
        logpdfs = logpdfs[default_chain-1]
        chains = chains[default_chain-1]
        chains = chains[burn_in:, :]
        logpdfs = logpdfs[burn_in:, :]
        MAP_idx = np.argmax(logpdfs[:, 0])
        print([*params_opt_names][i] + ' MAP and index after burn-in: ' + str([logpdfs[MAP_idx, 0],MAP_idx]))
        print([*params_opt_names][i] + ' MAP log-likelihood: ' + str(logpdfs[MAP_idx, 1]))
        MAPS_model_i = chains[MAP_idx,:]
        save_np_to_file(MAPS_model_i, filename_MAPS)
        print([*params_opt_names][i] + ' MAP saved')
    p_dic_model_i = dict(zip(params_opt_names[[*params_opt_names][i]], MAPS_model_i))
    for j, par in enumerate(table_2_params):
        table_2_MAPS[i,j] = p_dic_model_i.get(par, -150)
    
    if model_i == 'SItD':
        store_rate_vectors(p_dic_model_i,p)
        S, Iday, R, D, Itot = solve_difference_equations(p, p_dic_model_i, travel_data)
        p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dic_model_i)
        R_eff = calculate_R_instantaneous(p, S, p_dic_model_i)
        R_0 = R_eff[0]
        New_I = Iday[0,: -(p.numeric_max_age + p.extra_days_to_simulate)]
    else:
        alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dic_model_i)
        alpha= alpha[:-p.extra_days_to_simulate]
        rho_fit = p_dic_model_i.get('rho', p.rho)
        theta_fit = p_dic_model_i.get('theta', 1/5.2)
        R_0 = (rho_fit * 1) / theta_fit
        if model_i == 'SEIUR':
            S, E1, E2, New_I, I1 ,I2, Inew, U1, U2, R, D = solve_SEIUR_difference_equations(p, p_dic_model_i, travel_data)
            I = I1 + I2
        elif model_i == 'SIUR':
            S, I, New_I, U, R, D = solve_SIUR_difference_equations(p, p_dic_model_i, travel_data)
        else:
            S, I, New_I, R, D = solve_SIR_difference_equations(p, p_dic_model_i, travel_data)
        R_eff = ((rho_fit * alpha) / theta_fit) * (S / p.N)
    table_2_derived[i,:] = [round(R_0,2),round(min(R_eff),2), np.where(R_eff<1)[0][0], round(np.max(New_I[:maxtime_fit])/1000,1), round(S[-1]/p.N, 3)]

    ax.plot(t, New_I[:maxtime_fit], label = Model_label[i])
    ax2.plot(t, D[:maxtime_fit], label = Model_label[i])
    ax4.plot(R_eff[:maxtime_fit]  , label = Model_label[i] + r' $R_0 =$ ' + str( round(R_0, 2)))

ax.set_ylabel('Daily new infections')
ax.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
ax.grid(True)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_title(), visible=False)
ax2.legend(bbox_to_anchor=(0, 1.02), loc='lower left', ncol = 4)
ax2.set_ylabel('Daily deaths')
ax2.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.grid(True)
ax4.set_xlabel('Date')
ax4.set_ylabel(r'$\mathcal{R}$')
ax4.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
ax4.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
ax4.grid(True)


table_2_MAPS = np.around(table_2_MAPS, decimals= 4)
table_2_MAPS_list = table_2_MAPS.tolist()
table_2_MAPS_list.insert(0, table_2_params_labels)
table_2_MAPS_list_t = [list(x) for x in zip(*table_2_MAPS_list)]
for s in table_2_MAPS_list_t:
    print(*s, sep = " & ")

print('Derived parameters')
table_2_derived_list = table_2_derived.tolist()
table_2_derived_list.insert(0, table_2_derived_labels)
table_2_derived_list_t = [list(x) for x in zip(*table_2_derived_list)]
for s in table_2_derived_list_t:
    print(*s, sep = " & ")



# Derived parameters plot:
Derived_labels = [r'$\mathcal{R}_0$',r'$min_i \{\mathcal{R}_i\}$',r'$argmin_i \{\mathcal{R}_i<1\}$', r'$max_i \{S_i - S_{i+1} \}$', r'$S_{end}/N$']
Model_label = [r'$SI_tD$', r'$SIRD$', r'$SIRD_{\Delta_D}$', r'$SIURD$', r'$SE^2I^2U^2RD$' + ' \n  full',r'$SE^2I^2U^2RD$' + '\n' + r'$\eta = 1/5.2$']


R0 = []
Remin =[]
New_I =[]
tstart =[]

burn_in = [25000, 25000, 25000, 100000, 200000, 25000]
thinning_factor = 10
params_opt_names = {}
for i, model_i in enumerate(Model_name): 
    p.eta = 0.1923
    p.theta = 1/5.2
    p.xi = 1/18
    p.DeltaD = 0
    p.alpha = np.ones(p.maxtime)
    if i<5:
        params_opt_names[model_i] = parameter_to_optimise_list(True, True, model_i, [])
        params_opt_names[model_i].extend(['negative_binomial_phi'])
        filename = get_file_name_suffix(p, 'ONS-UK', 'model-' + model_i + '_full-fit-True', params_opt_names[model_i])
    else:
        params_opt_names[model_i+'_fixEta'] = parameter_to_optimise_list(False, True, model_i,['rho', 'Iinit1', 'theta', 'xi'])
        filename = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', model_i , params_opt_names[model_i+'_fixEta'])
        filename = filename + '_fixedEta-' + str(0.1923)
        params_opt_names[model_i+'_fixEta'].extend(['negative_binomial_phi'])
    filename_epiparams = os.path.join(folder_fits, filename + '_epiParams_chain_' + str(default_chain) + '_burnIn_' + str(burn_in[i]) + '_thinning_' + str(thinning_factor) + '.npy')
    Epipar_model_i = np.load(filename_epiparams)
    print(model_i + ' epi pars loaded')
    R0.append(Epipar_model_i[0,:])
    Remin.append(Epipar_model_i[1,:])
    tstart.append(Epipar_model_i[2,:])
    New_I.append(Epipar_model_i[3,:])
    


colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b']

gridR =grid1[0].subgridspec(1, 4,wspace=0.05)

axR = fig.add_subplot(gridR[0])
axR2 = fig.add_subplot(gridR[1])
axR3 = fig.add_subplot(gridR[2])
axR4 =  fig.add_subplot(gridR[3])

# Create an axes instance
R0vp = axR.violinplot(R0,showmeans = True, vert = False, positions = [6, 5,4,3,2,1])
axR.set_yticks(np.arange(1, len(Model_label) + 1))
axR.set_yticklabels(reversed(Model_label))
#ax.set_xlim([0,18])
Reminvp = axR2.violinplot(Remin, showmeans = True, vert = False, positions = [6, 5,4,3,2,1])
axR2.set_yticks(np.arange(1, len(Model_label) + 1))
axR2.set_yticklabels([])
tstartvp = axR3.violinplot(tstart, showmeans = True, vert = False, positions = [6, 5,4,3,2,1], bw_method = 0.5)
axR3.set_yticks(np.arange(1, len(Model_label) + 1))
axR3.set_yticklabels([])
NewIvp = axR4.violinplot(New_I, showmeans = True, vert = False, positions = [6, 5,4,3,2,1])
axR4.set_yticks(np.arange(1, len(Model_label) + 1))
axR4.set_yticklabels([])


for vpname in [R0vp, Reminvp, tstartvp, NewIvp]:
    for partname in ['bodies','cbars','cmins','cmaxes','cmeans']:
        vp = vpname[partname]
        if partname == 'bodies':
            for i, pc in enumerate(vp):
                pc.set_color(colors[i])
        else:
            vp.set_color(colors)

for i, axname in enumerate([axR,axR2, axR3, axR4]):
    axname.set_title(Derived_labels[i])


# Distributions plot

Model_name = ['SIR', 'SIRDeltaD', 'SIUR', 'SEIUR', 'SEIUR']
Model_line = ['-','-.', '-','-','-']
Model_label = [r'$SI_tD$', r'$SIRD$', r'$SIRD_{\Delta_D}$', r'$SIURD$', r'$SE^2I^2U^RD$ full', r'$SE^2I^2U^2RD$' + '\n' + r'$\eta = 1/5.2$']

# Clinical data params:
days_infect = np.linspace(0, 15, 16)
kbeta_mean = p.beta_mean
kbeta_var = p.beta_var
ksc = kbeta_var / kbeta_mean  # scale parameter
ka = kbeta_mean / ksc    # shape parameter

days_D = np.linspace(0, 50, 51)
kdeath_mean = p.death_mean
kdeath_dispersion = p.death_dispersion
kdeath_N_NB = 1 / kdeath_dispersion
kdeath_p_NB = 1 / (1 + kdeath_mean * kdeath_dispersion)


# Start figure with histograms of the data
gridD =grid1[1].subgridspec(1,2,wspace=0.23, hspace=0.2)
axD2 = fig.add_subplot(gridD[0])
axD3 = fig.add_subplot(gridD[1])
#plt.subplots_adjust(wspace=0, hspace=0)
axD2.grid(True)
#axD2.set_title('Infectiousness profile')
axD2.bar(days_infect, gamma.pdf(days_infect, ka, loc=0, scale=ksc), linewidth=2, color='blue', alpha=0.6, label = 'Data')
axD2.plot(days_infect, gamma.pdf(days_infect, ka, loc=0, scale=ksc), color = colors[0],  label = Model_label[0])
axD3.grid(True)
#axD3.set_title('Infection-to-death distribution')
axD3.bar(days_D, nbinom.pmf(days_D, kdeath_N_NB, kdeath_p_NB), linewidth=2, color='blue', alpha=0.6, label = 'Data')
axD3.plot(days_D, nbinom.pmf(days_D, kdeath_N_NB, kdeath_p_NB),  color = colors[0],  label = Model_label[0])


tmax = 500
mean_beta = np.zeros(len(Model_name))
mean_pzeta = np.zeros(len(Model_name))

params_opt_names = {}
for i, model_i in enumerate(Model_name): 
    p.eta = 0.1923
    p.theta = 1/5.2
    p.xi = 1/18
    p.DeltaD = 0
    if i<4:
        params_opt_names[model_i] = parameter_to_optimise_list(True, True, model_i, [])
        params_opt_names[model_i].extend(['negative_binomial_phi'])
        filename = get_file_name_suffix(p, 'ONS-UK', 'model-' + model_i + '_full-fit-True', params_opt_names[model_i])
    else:
        params_opt_names[model_i+'_fixEta'] = parameter_to_optimise_list(False, True, model_i,['rho', 'Iinit1', 'theta', 'xi'])
        filename = get_file_name_suffix_anymodel(p, 'ONS-UK', '','', model_i , params_opt_names[model_i+'_fixEta'])
        filename = filename + '_fixedEta-' + str(0.1923)
        params_opt_names[model_i+'_fixEta'].extend(['negative_binomial_phi'])
    saveas = os.path.join(folder_fits, filename)
    filename_MAPS = saveas + '-chain-'+ str(default_chain) + '-MAPS.npy'
    MAPS_model_i = np.load(filename_MAPS)
    print([*params_opt_names][i] + ' MAP loaded')
    p_dic_model_i = dict(zip(params_opt_names[[*params_opt_names][i]], MAPS_model_i))
    
    
    theta = p_dic_model_i.get('theta',p.theta)
    eta = p_dic_model_i.get('eta',p.eta)
    xi = p_dic_model_i.get('xi',p.xi)
    DeltaD = p_dic_model_i.get('DeltaD',p.DeltaD)

    p.theta = theta
    p.eta = eta
    p.xi = xi
    p.DeltaD = DeltaD

    betaj, pzetaj = MODEL_Profile_FUNCTIONS[model_i](p, tmax)
    mean_beta[i] = np.average(range(tmax+1), weights = betaj)
    mean_pzeta[i] = np.average(range(tmax + 1 + int(p.DeltaD)), weights = pzetaj)
    
    axD2.plot(days_infect, betaj[:len(days_infect)], Model_line[i], color=colors[i+1] ,label = Model_label[i+1])
    axD3.plot(days_D, pzetaj[:len(days_D)], color=colors[i+1], label = Model_label[i+1])

axD2.set_xlabel(r'Day, $j$')
axD2.set_ylabel(r'Probability, $\beta_j$')
axD3.set_xlabel(r'Day, $j$')
axD3.set_ylabel(r'Probability, $\zeta_j$')


print('Beta means at MAPs:')
print(mean_beta)
print('Pzeta means at MAPs:')
print(mean_pzeta)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, Figure_name), bbox_inches='tight')

