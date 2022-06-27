import os
import argparse
import glob
import shutil
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


# Synthetic data file
data_filename = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy'
Figure_name = 'Figure3B.png'
plt.rcParams.update({'font.size': 10})
# Definiiton of parameters and  fits
folder_fits = os.path.join(MODULE_DIR, 'out-mcmc-table2B')
folder_data = os.path.join(MODULE_DIR, 'out_SIRvsAGEfits')
default_chain = 1
burn_in = 25000
Model_name = ['SItD', 'SIR', 'SIRDeltaD', 'SIUR', 'SEIUR',]
Model_label = [r'$SI_tD$', r'$SIRD$', r'$SIRD_{\Delta_D}$', r'$SIURD$', r'$SE^2I^2U^2RD$']

# PARAMETERS
maxtime_fit = 150 # Days to simulate
travel_data = True


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

p.flat_priors = False
p.day_1st_death_after_150220 = 22



# Get Age data from file
print('Getting simulated data...')
data = np.load(os.path.join(folder_data, data_filename))
data_S = data[:,0]
data_Itot = data[:,1]
data_R = data[:,2]
data_Dreal = data[:,3]
data_D = data[:,4] # noise data
data_I = data[:,5:].T # transpose to get exactly the same shape as other code

# Get correct length of plots:
if len(data_R) < maxtime_fit:
    p.maxtime = len(data_R) -1
    maxtime_fit = len(data_R) -1
else:
    p.maxtime = maxtime_fit
# cut the data to the maxtime lenght:
data_D = data_D[:p.maxtime+1]
data_Dreal = data_Dreal[:p.maxtime+1]
data_S_long = data_S
data_S = data_S[:p.maxtime+1]
data_Itot = data_Itot[:p.maxtime+1]
data_R = data_R[:p.maxtime+1]
data_I = data_I[:,:p.maxtime+1]
# to get the same data and fit lenghts as in Data_loader
p.maxtime = p.maxtime + p.numeric_max_age + p.extra_days_to_simulate

# True simulation paramters:
p_dict_data = dict(zip(['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_offset', 'negative_binomial_phi'], [3.203, 860, 0.2814, 31.57, 0.002]))

store_rate_vectors(p_dict_data,p)
p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_data)
R_eff_data = calculate_R_instantaneous(p, data_S_long, p_dict_data)
R_0_data = R_eff_data[0]

# Initialize table + data entry
table_2_params_labels = ['$I_0$', '$\\rho$', '$t^*$' , '$\\alpha_b$', '$\\theta$', '$\eta$', '$\\xi$', '$\\Delta D$', '$\\phi$' ]
table_2_derived_labels = ['$\mathcal{R}_0$','$\min_i \{\mathcal{R}_i\}$','argmin$_i \{\mathcal{R}_i<1\}$','$\max_i \{S_i - S_{i+1} \} (\\times 1000)', '$S_{end}/N$']
table_2_params = ['Iinit1', 'rho', 'lockdown_offset' , 'lockdown_baseline', 'theta', 'eta', 'xi', 'DeltaD', 'negative_binomial_phi']
table_2_MAPS = np.zeros([6,9])
for i, par in enumerate(table_2_params):
    table_2_MAPS[0,i] = p_dict_data.get(par, -150)
table_2_derived = np.zeros([6,5])
table_2_derived[0,:] = [round(R_0_data,2),round(min(R_eff_data),2), np.where(R_eff_data<1)[0][0], round(np.max(data_I[0,:])/1000,1), round(data_S[-1]/p.N, 3)]

# Initialize plot
Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
date_list = [Feb15 + timedelta(days=x) for x in range(maxtime_fit+1)]
# time
t = np.linspace(0, maxtime_fit-1, maxtime_fit)
fig = plt.figure(figsize=(8.0, 11), dpi=200)
grid = gridspec.GridSpec(2,1)
grid0 =grid[0].subgridspec(3, 1,wspace=0.2, hspace=0.05)
grid1 =grid[1].subgridspec(2, 1, hspace=0.35)
ax = fig.add_subplot(grid0[1])
ax2 = fig.add_subplot(grid0[0])
ax4 =  fig.add_subplot(grid0[2])
ax.plot(t, data_I[0,:maxtime_fit],'b', label = 'Synt data')
ax2.plot(t[p.day_1st_death_after_150220:], data_D[p.day_1st_death_after_150220:maxtime_fit],'b.', label = 'Synt data')
ax4.plot(R_eff_data[:maxtime_fit], 'b', label = r'Synt data $R_0 =$ ' + str( round(R_0_data, 2)))

params_opt_names = {}
for i, model_i in enumerate(Model_name): 


    if model_i == 'SIRDeltaD':
        p.DeltaD = 18.69 - 1 - 5.2
    else:
        p.DeltaD = 0

    p.alpha = np.ones(p.maxtime)
    
    if model_i == 'SEIUR':
        p.theta = 0.3362
        p.eta = 0.3362
        p.xi = 1 / 11.744308
        fixed_params_tag = '_fixedEta-' + str(p.eta)  + '_fixedTheta-' + str(p.theta)
    elif model_i != 'SItD':
        p.xi = 1 / (18.69 - 1 - 5.2)
        p.theta = 0.1923
        fixed_params_tag = '_fixedTheta-' + str(p.theta)
    else:
        fixed_params_tag = ''
    if model_i == 'SItD':
        params_opt_names[model_i] = parameter_to_optimise_list(True, True, model_i, [])
        params_opt_names[model_i].extend(['negative_binomial_phi'])
        filename = get_file_name_suffix(p, 'SimSItD-1_rho_0-2', 'NBphi_2e-3_model-' + model_i + '_full-fit-True', params_opt_names[model_i])
    else:
        params_opt_names[model_i] = parameter_to_optimise_list(False, True, model_i, ['rho', 'Iinit1'])
        filename = get_file_name_suffix_anymodel(p, 'SimSItD-1', '_rho_0-2', 'NBphi_2e-3_', model_i , params_opt_names[model_i])
        filename = filename + fixed_params_tag
        params_opt_names[model_i].extend(['negative_binomial_phi'])
    saveas = os.path.join(folder_fits, filename)
    filename_MAPS = saveas + '-chain-'+ str(default_chain) + '-MAPS.npy'
    if os.path.exists(filename_MAPS):
        MAPS_model_i = np.load(filename_MAPS)
        print(model_i + ' MAP loaded')
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
        print(model_i + ' MAP and index after burn-in: ' + str([logpdfs[MAP_idx, 0],MAP_idx]))
        print(model_i + ' MAP log-likelihood: ' + str(logpdfs[MAP_idx, 1]))
        MAPS_model_i = chains[MAP_idx,:]
        save_np_to_file(MAPS_model_i, filename_MAPS)
        print(model_i + ' MAP saved')
    p_dic_model_i = dict(zip(params_opt_names[model_i], MAPS_model_i))
    for j, par in enumerate(table_2_params):
        table_2_MAPS[i+1,j] = p_dic_model_i.get(par, -150)
    
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
        theta_fit = p_dic_model_i.get('theta', p.theta)
        R_0 = (rho_fit * 1) / theta_fit
        if model_i == 'SEIUR':
            S, E1, E2, New_I, I1 ,I2, Inew, U1, U2, R, D = solve_SEIUR_difference_equations(p, p_dic_model_i, travel_data)
            I = I1 + I2
        elif model_i == 'SIUR':
            S, I, New_I, U, R, D = solve_SIUR_difference_equations(p, p_dic_model_i, travel_data)
        else:
            S, I, New_I, R, D = solve_SIR_difference_equations(p, p_dic_model_i, travel_data)
        R_eff = ((rho_fit * alpha) / theta_fit) * (S / p.N)
    table_2_derived[i+1,:] = [round(R_0,2),round(min(R_eff),2), np.where(R_eff<1)[0][0], round(np.max(New_I[:maxtime_fit])/1000,1), round(S[-1]/p.N,3)]

    ax.plot(t, New_I[:maxtime_fit], label = Model_label[i])
    ax2.plot(t, D[:maxtime_fit], label = Model_label[i])
    ax4.plot(R_eff[:maxtime_fit]  , label = Model_label[i] + r' $R_0 =$ ' + str( round(R_0, 2)))


ax.set_ylabel('Daily new infections')
ax.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
ax.grid(True)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_title(), visible=False)
ax2.legend(bbox_to_anchor=(-0.05, 1.02), loc='lower left', ncol = len(Model_label) + 1)
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
Derived_data = [R_0_data, min(R_eff_data), np.where(R_eff_data<1)[0][0], np.max(data_I[0,:])]

params_opt_names = {}
R0 = []
Remin =[]
New_I =[]
tstart =[]

burn_in = 25000
thinning_factor = 10
for i, model_i in enumerate(Model_name): 
    if model_i == 'SIRDeltaD':
        p.DeltaD = 18.69 - 1 - 5.2
    else:
        p.DeltaD = 0
    p.alpha = np.ones(p.maxtime)
    if model_i == 'SEIUR':
        p.theta = 0.3362
        p.eta = 0.3362
        p.xi = 1 / 11.744308
        fixed_params_tag = '_fixedEta-' + str(p.eta)  + '_fixedTheta-' + str(p.theta)
    elif model_i != 'SItD':
        p.xi = 1 / (18.69 - 1 - 5.2)
        p.theta = 0.1923
        fixed_params_tag = '_fixedTheta-' + str(p.theta)
    else:
        fixed_params_tag = ''
        
    if model_i == 'SItD':
        params_opt_names[model_i] = parameter_to_optimise_list(True, True, model_i, [])
        params_opt_names[model_i].extend(['negative_binomial_phi'])
        filename = get_file_name_suffix(p, 'SimSItD-1_rho_0-2', 'NBphi_2e-3_model-' + model_i + '_full-fit-True', params_opt_names[model_i])
        
    else:
        params_opt_names[model_i] = parameter_to_optimise_list(False, True, model_i, ['rho', 'Iinit1'])
        filename = get_file_name_suffix_anymodel(p, 'SimSItD-1', '_rho_0-2', 'NBphi_2e-3_', model_i , params_opt_names[model_i])
        filename = filename + fixed_params_tag
        params_opt_names[model_i].extend(['negative_binomial_phi'])
    filename_epiparams = os.path.join(folder_fits, filename + '_epiParams_chain_' + str(default_chain) + '_burnIn_' + str(burn_in) + '_thinning_' + str(thinning_factor) + '.npy')
    Epipar_model_i = np.load(filename_epiparams)
    print(model_i + ' epi pars loaded')
    R0.append(Epipar_model_i[0,:])
    Remin.append(Epipar_model_i[1,:])
    tstart.append(Epipar_model_i[2,:])
    New_I.append(Epipar_model_i[3,:])
    

colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']#, u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
gridR =grid1[0].subgridspec(1, 4,wspace=0.05)

axR = fig.add_subplot(gridR[0])
axR2 = fig.add_subplot(gridR[1])
axR3 = fig.add_subplot(gridR[2])
axR4 =  fig.add_subplot(gridR[3])

# Create an axes instance
R0vp = axR.violinplot(R0,showmeans = True, vert = False, positions = [5,4,3,2,1])
axR.set_yticks(np.arange(1, len(Model_label) + 1))
axR.set_yticklabels(reversed(Model_label))
#ax.set_xlim([0,18])
Reminvp = axR2.violinplot(Remin, showmeans = True, vert = False, positions = [5,4,3,2,1])
axR2.set_yticks(np.arange(1, len(Model_label) + 1))
axR2.set_yticklabels([])
tstartvp = axR3.violinplot(tstart, showmeans = True, vert = False, positions = [5,4,3,2,1], bw_method = 0.5)
axR3.set_yticks(np.arange(1, len(Model_label) + 1))
axR3.set_yticklabels([])
NewIvp = axR4.violinplot(New_I, showmeans = True, vert = False, positions = [5,4,3,2,1])
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
    axname.axvline(x=Derived_data[i], color='b', linestyle='--')



Model_name = ['SIR', 'SIRDeltaD', 'SIUR', 'SEIUR']
Model_line = ['-','-.', ':','-']
Model_label = [r'$SI_tD$', r'$SIRD$', r'$SIRD_{\Delta_D}$', r'$SIURD$', r'$SE^2I^2U^2RD$']

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
gridD =grid1[1].subgridspec(1,2,wspace=0.25, hspace=0.2)
axD2 = fig.add_subplot(gridD[0])
axD3 = fig.add_subplot(gridD[1])
#plt.subplots_adjust(wspace=0, hspace=0)
axD2.grid(True)
#axD2.set_title('Infectiousness profile')
axD2.bar(days_infect, gamma.pdf(days_infect, ka, loc=0, scale=ksc), linewidth=2, color='blue', alpha=0.6, label = 'Synt data')
axD2.plot(days_infect, gamma.pdf(days_infect, ka, loc=0, scale=ksc), color = colors[0],  label = Model_label[0])
axD3.grid(True)
#axD3.set_title('Infection-to-death distribution')
axD3.bar(days_D, nbinom.pmf(days_D, kdeath_N_NB, kdeath_p_NB), linewidth=2, color='blue', alpha=0.6, label = 'Synt data')
axD3.plot(days_D, nbinom.pmf(days_D, kdeath_N_NB, kdeath_p_NB),  color = colors[0],  label = Model_label[0])


tmax = 500
mean_beta = np.zeros(len(Model_name))
mean_pzeta = np.zeros(len(Model_name))

for i, model_i in enumerate(Model_name): 
    p.theta = 0.1923
    p.DeltaD = 0

    if model_i == 'SIRDeltaD':
        p.DeltaD = 18.69 - 1 - 5.2

    if model_i == 'SEIUR':
        p.theta = 1 / 2.974653
        p.eta = 1 / 2.974653
        p.xi = 1 /  11.744308
        #2.974653 11.744308
        # 0.2605 0.4285 1 / 11.505735
        T = T_matrix_SEIUR(p)
        betaj, pzetaj = profiles_SEIUR(p, tmax)
        mean_beta[i] = np.average(range(tmax+1),weights=betaj)
        mean_pzeta[i] = np.average(range(tmax+1),weights=pzetaj)
        
    elif model_i == 'SIUR':
        p.xi = 1 / (18.69 - 1 - 5.2)
        betaj, pzetaj = profiles_SIUR(p, tmax)
        mean_beta[i] = np.average(range(tmax+1),weights=betaj)
        mean_pzeta[i] = np.average(range(tmax+1),weights=pzetaj)

        
    else:
        betaj, pzetaj = profiles_SIR_DeltaD(p, tmax)
        mean_beta[i] = np.average(range(tmax + 1), weights=betaj)
        mean_pzeta[i] = np.average(range(tmax + 1 + int(p.DeltaD)), weights=pzetaj)
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
plt.savefig(Figure_name, bbox_inches='tight')

fig_path = os.path.join(MODULE_DIR, 'figures', 'saved-plots')
if not os.path.exists(fig_path):
   os.mkdir(fig_path)

files = glob.glob('./*.png')
for f in files:
    shutil.move(f, fig_path)
