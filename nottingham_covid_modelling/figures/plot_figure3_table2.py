import os
import argparse
import glob
import shutil
import cma
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from datetime import datetime, timedelta, date
import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pints
from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import NOISE_MODEL_MAPPING, POPULATION
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, solve_SIR_difference_equations, store_rate_vectors, solve_SIUR_difference_equations, step, solve_SEIUR_difference_equations
from nottingham_covid_modelling.lib.settings import Params, get_file_name_suffix
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE
from nottingham_covid_modelling.lib.ratefunctions import calculate_R_instantaneous

def parameter_to_optimize_list(FitFull, FitStep, model_name):
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


# Synthetic data file
data_filename = 'SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy'
Figure_name = 'Figure3_testSEIUR.png'
# Definiiton of parameters and  fits
folder_fits = os.path.join(MODULE_DIR, 'cmaes_fits_SIR')
folder_data = os.path.join(MODULE_DIR, 'out_SIRvsAGEfits')

filename_SIR = 'Data_SimSItD-1_rho_0-2_noise-model_NBphi_2e-3_model-SIR_full-fit-True_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset.txt'
filename_SIRDeltaD ='Data_SimSItD-1_rho_0-2_noise-model_NBphi_2e-3_model-SIRDeltaD_full-fit-True_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset.txt'
filename_SIUR ='Data_SimSItD-1_rho_0-2_noise-model_NBphi_2e-3_model-SIUR_full-fit-True_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset.txt'
filename_SEIUR = 'Data_SimSItD-1_rho_0-2_noise-model_NBphi_2e-3_model-SEIUR_full-fit-True_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset.txt'
filename_AGE ='Data_SimSItD-1_rho_0-2_noise-model_NBphi_2e-3_model-SItD_full-fit-True_square-lockdown_rho_Init1_lockdown-baseline_lockdown-offset.txt'

parameters_to_optimise_SIUR = parameter_to_optimize_list(True, True, 'SIUR')
parameters_to_optimise_SEIUR = parameter_to_optimize_list(True, True, 'SEIUR')
parameters_to_optimise_SIR = parameter_to_optimize_list(True, True, 'SIR')
parameters_to_optimise_SIRDeltaD = parameter_to_optimize_list(True, True, 'SIRDeltaD')
parameters_to_optimise = parameter_to_optimize_list(True, True, 'SItD')
# Load parameters
obtained_parameters_SIR = np.loadtxt(os.path.join(folder_fits, filename_SIR))
p_dict_SIR = dict(zip(parameters_to_optimise_SIR, obtained_parameters_SIR))
obtained_parameters_SIRDeltaD = np.loadtxt(os.path.join(folder_fits, filename_SIRDeltaD))
p_dict_SIRDeltaD = dict(zip(parameters_to_optimise_SIRDeltaD, obtained_parameters_SIRDeltaD))
obtained_parameters_SIUR = np.loadtxt(os.path.join(folder_fits, filename_SIUR))
p_dict_SIUR = dict(zip(parameters_to_optimise_SIUR, obtained_parameters_SIUR))
obtained_parameters_SEIUR = np.loadtxt(os.path.join(folder_fits, filename_SEIUR))
p_dict_SEIUR = dict(zip(parameters_to_optimise_SEIUR, obtained_parameters_SEIUR))
obtained_parameters = np.loadtxt(os.path.join(folder_fits, filename_AGE))
p_dict_SItRD = dict(zip(parameters_to_optimise, obtained_parameters))

# True simulation paramters:
p_dict_data = dict(zip(parameters_to_optimise, [3.203, 860, 0.2814, 31.57, 0.002]))

# PARAMETERS
# labels for saving/plotting
Model2_label = r'SI$_t$D'
maxtime_fit = 150 # Days to simulate
travel_data = True


p = Params()
p.N = 59.1e6
p.numeric_max_age = 35
p.extra_days_to_simulate = 10
p.IFR = 0.00724 # UK time
p.square_lockdown = True
# parameters based on UK google and ONS data
p.alpha = np.ones(p.maxtime)
p.lockdown_baseline = 0.2814
p.lockdown_offset = 31.57
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

# Dummy parameters for the forward model format
beta_SIR = 1
theta_SIR = 1 / p.beta_mean
theta_SIUR = 1 / p.beta_mean
DeltaD_SIR = int(p.death_mean - p.beta_mean)
xi_SIUR = 1 / (p.death_mean -p.beta_mean)
eta_SEIUR = 1 / p.beta_mean

# SIMULATIONS

#('-------  SIRD model -----------')
# Update dummy params and alpha
p.beta = beta_SIR
p.DeltaD = 0
p.theta = theta_SIR
if p.square_lockdown:
    alpha_SIR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIR)
    alpha_SIR = alpha_SIR[:-p.extra_days_to_simulate]
else:
    alpha_SIR = np.ones(p.maxtime+1)
# Run forward model
S_s, I_s, Inew_s, R_s, D_s = solve_SIR_difference_equations(p, p_dict_SIR, travel_data)
rho_SIR = p_dict_SIR.get('rho', p.rho)
Iinit_SIR = p_dict_SIR.get('Iinit1', p.Iinit1)
theta_fit_SIR = p_dict_SIR.get('theta',theta_SIR)
R_0_s = (rho_SIR * p.beta * 1) / theta_fit_SIR
R_eff_s = ((rho_SIR * p.beta * alpha_SIR) / theta_fit_SIR) * (S_s / p.N)


# ('------- SIRD-DeltaD model -----------')
# Update dummy params and alpha
p.beta = beta_SIR
p.DeltaD = DeltaD_SIR
p.theta = theta_SIR
if p.square_lockdown:
    alpha_SIRDeltaD = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIRDeltaD)
    alpha_SIRDeltaD = alpha_SIRDeltaD[:-p.extra_days_to_simulate]
else:
    alpha_SIRDeltaD = np.ones(p.maxtime+1)
# Run forward model
S_sD, I_sD, Inew_sD, R_sD, D_sD = solve_SIR_difference_equations(p, p_dict_SIRDeltaD, travel_data)
rho_SIRDeltaD = p_dict_SIRDeltaD.get('rho', p.rho)
Iinit_SIRDeltaD = p_dict_SIRDeltaD.get('Iinit1', p.Iinit1)
theta_fit_SIRDeltaD = p_dict_SIRDeltaD.get('theta',theta_SIR)
R_0_sD = (rho_SIRDeltaD * p.beta * 1) / theta_fit_SIRDeltaD
R_eff_sD = ((rho_SIRDeltaD * p.beta * alpha_SIRDeltaD) / theta_fit_SIRDeltaD) * (S_sD / p.N)


# ('------- SIURD model -----------')
# Update dummy params and alpha
p.theta = theta_SIUR
p.xi = xi_SIUR
if p.square_lockdown:
    alpha_SIUR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SIUR)
else:
    alpha_SIUR = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
# Run forward model
S_u, I_u, Inew_u, N_u, R_u, D_u = solve_SIUR_difference_equations(p, p_dict_SIUR, travel_data)
rho_SIUR = p_dict_SIUR.get('rho', p.rho)
Iinit_SIUR = p_dict_SIUR.get('Iinit1', p.Iinit1)
theta_fit_SIUR = p_dict_SIUR.get('theta', theta_SIUR)
R_0_u = (rho_SIUR * p.beta * 1) / theta_fit_SIUR
R_eff_u = ((rho_SIUR * p.beta * alpha_SIUR[:-p.extra_days_to_simulate]) / theta_fit_SIUR) * (S_u / p.N)

# ('------- SEIURD model -----------')
# Update dummy params and alpha
p.theta = theta_SIUR
p.xi = xi_SIUR
p.eta = eta_SEIUR
if p.square_lockdown:
    alpha_SEIUR = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SEIUR)
else:
    alpha_SEIUR = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
# Run forward model
S_e, E1_e, E2_e, Enew_e, I1_e ,I2_e, Inew_e, U1_e, U2_e, R_e, D_e = solve_SEIUR_difference_equations(p, p_dict_SEIUR, travel_data)
rho_SEIUR = p_dict_SEIUR.get('rho', p.rho)
Iinit_SEIUR = p_dict_SEIUR.get('Iinit1', p.Iinit1)
theta_fit_SEIUR = p_dict_SEIUR.get('theta', theta_SIUR)
R_0_e = (rho_SEIUR * p.beta * 1) / theta_fit_SEIUR
R_eff_e = ((rho_SEIUR * p.beta * alpha_SEIUR[:-p.extra_days_to_simulate]) / theta_fit_SEIUR) * (S_e / p.N)



# ('------- SItD model -----------')
# Update params
store_rate_vectors(p_dict_SItRD,p)
S_a, Iday_a, R_a, D_a, Itot_a = solve_difference_equations(p, p_dict_SItRD, travel_data)
# For R_0 and R_eff
if p.square_lockdown:
    p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_SItRD)
else:
    p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
R_eff_a = calculate_R_instantaneous(p, S_a, p_dict_SItRD)
R_0_a = R_eff_a[0]

# ('----------- data Reff -----------')
# Update params
store_rate_vectors(p_dict_data,p)
if p.square_lockdown:
    p.alpha = step(p, lgoog_data = p.maxtime + 1  - p.numeric_max_age, parameters_dictionary = p_dict_data)
else:
    p.alpha = np.ones(p.maxtime + 1 + p.extra_days_to_simulate)
R_eff_data = calculate_R_instantaneous(p, data_S_long, p_dict_data)
R_0_data = R_eff_data[0]



print('TABLE 2 ...')
print('Par          syn data   SItD    SIRD    SIRD_DeltaD SIURD    SEIURD')
print('I_0          ' + str(p_dict_data.get('Iinit1')) + '      ' + str(round(p_dict_SItRD.get('Iinit1'),2)) + '   ' + str(round(p_dict_SIR.get('Iinit1'),2)) + '       ' + str(round(p_dict_SIRDeltaD.get('Iinit1'),2)) + '   ' + str(round(p_dict_SIUR.get('Iinit1'),2)) + '   ' + str(round(p_dict_SEIUR.get('Iinit1'),2)))
print('rho          ' + str(p_dict_data.get('rho')) + '      ' + str(round(p_dict_SItRD.get('rho'),2)) + '   ' + str(round(p_dict_SIR.get('rho'),3)) + '       ' + str(round(p_dict_SIRDeltaD.get('rho'),3)) + '   ' + str(round(p_dict_SIUR.get('rho'),2))+ '   ' + str(round(p_dict_SEIUR.get('rho'),2)))
print('t^*          ' + str(p_dict_data.get('lockdown_offset')) + '      ' + str(round(p_dict_SItRD.get('lockdown_offset'),1)) + '   ' + str(round(p_dict_SIR.get('lockdown_offset'),2)) + '       ' + str(round(p_dict_SIRDeltaD.get('lockdown_offset'),2)) + '   ' + str(round(p_dict_SIUR.get('lockdown_offset'),2))+ '   ' + str(round(p_dict_SEIUR.get('lockdown_offset'),2)))
print('alpha_b      ' + str(p_dict_data.get('lockdown_baseline'))  + '    ' + str(round(p_dict_SItRD.get('lockdown_baseline'),4)) + '   ' + str(round(p_dict_SIR.get('lockdown_baseline'),3)) + '       ' + str(round(p_dict_SIRDeltaD.get('lockdown_baseline'),3)) + '   ' + str(round(p_dict_SIUR.get('lockdown_baseline'),3)) + '   ' + str(round(p_dict_SEIUR.get('lockdown_baseline'),3)))
print('theta        -          -        '+ str(round(p_dict_SIR.get('theta'),3)) + '       ' + str(round(p_dict_SIRDeltaD.get('theta'),3)) + '   ' + str(round(p_dict_SIUR.get('theta'),3)) + '   ' + str(round(p_dict_SEIUR.get('theta'),3)))
print('eta           -          -        -          -       -       ' + str(round(p_dict_SEIUR.get('eta'),3)))
print('xi           -          -        -          -      ' + str(round(p_dict_SIUR.get('xi'),3)) + '   ' + str(round(p_dict_SEIUR.get('xi'),3)))
print('Delta D      -          -        -          '+ str(round(p_dict_SIRDeltaD.get('DeltaD'),2)) +'     -     -')
print('phi          ' + str(p_dict_data.get('negative_binomial_phi')) + '      ' + str(round(p_dict_SItRD.get('negative_binomial_phi'),4)) + '   ' + str(round(p_dict_SIR.get('negative_binomial_phi'),3)) + '       ' + str(round(p_dict_SIRDeltaD.get('negative_binomial_phi'),3)) + '   ' + str(round(p_dict_SIUR.get('negative_binomial_phi'),3))+ '   ' + str(round(p_dict_SEIUR.get('negative_binomial_phi'),3)))
print('Derived parameters')
print('R_0          ' + str(round(R_0_data,2)) + '      ' + str(round(R_0_a,2)) + '     ' + str(round(R_0_s,2)) + '       ' + str(round(R_0_sD,2)) + '     ' + str(round(R_0_u,2))+ '     ' + str(round(R_0_e,2)))
print('min{R_i}     ' + str(round(min(R_eff_data),2)) + '      ' + str(round(min(R_eff_a),2)) + '     ' + str(round(min(R_eff_s),2)) + '       ' + str(round(min(R_eff_sD),2)) + '   ' + str(round(min(R_eff_u),2))+ '   ' + str(round(min(R_eff_e),2)))
print('argmin{Ri<1} ' + str(np.where(R_eff_data<1)[0][0]) + '      ' + str(np.where(R_eff_a<1)[0][0])  + '     ' + str(np.where(R_eff_s<1)[0][0])  + '       ' + str(np.where(R_eff_sD<1)[0][0])  + '   ' + str(np.where(R_eff_u<1)[0][0]) + '   ' + str(np.where(R_eff_e<1)[0][0]) )
print('max{I_{i,1}x1000  ' + str(round(np.max(data_I[0,:])/1000,1)) + '      ' + str(round(np.max(Iday_a[0,: -(p.numeric_max_age + p.extra_days_to_simulate)])/1000,1)) + '     ' + str(round(np.max(Inew_s[: -(p.numeric_max_age + p.extra_days_to_simulate)])/1000,1)) + '       ' + str(round(np.max(Inew_sD[: -(p.numeric_max_age + p.extra_days_to_simulate)])/1000,1)) + '   ' + str(round(np.max(Inew_u[: -(p.numeric_max_age + p.extra_days_to_simulate)])/1000,1)) + '   ' + str(round(np.max(Inew_e[: -(p.numeric_max_age + p.extra_days_to_simulate)])/1000,1)))

# figure with R_eff
print('Ploting figure 3 ...')

# xticks:
Feb15 = datetime.strptime("15-02-2020", "%d-%m-%Y").date()
date_list = [Feb15 + timedelta(days=x) for x in range(maxtime_fit+1)]
# time
t = np.linspace(0, maxtime_fit-1, maxtime_fit)
fig, (ax2, ax, ax4) = plt.subplots(3, 1, figsize=(8.0, 6), dpi=300)
ax.plot(t, data_I[0,:maxtime_fit],'b', label = 'Synt data')
ax.plot(t, Iday_a[0,:maxtime_fit], label = Model2_label)
ax.plot(t, Inew_s[:maxtime_fit], label='SIRD')
ax.plot(t, Inew_sD[:maxtime_fit], label=r'SIRD$_{\Delta D}$')
ax.plot(t, Inew_u[:maxtime_fit], label='SIURD')
ax.plot(t, Inew_e[:maxtime_fit], label='SEIURD')
ax.legend()
ax.set_title('Daily new infections')
ax.set_ylabel('Number')
ax.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
ax.grid(True)
ax.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
ax2.plot(t[p.day_1st_death_after_150220:], data_D[p.day_1st_death_after_150220:maxtime_fit],'b.', label = 'Synt data')
ax2.plot(t, D_a[:maxtime_fit], label = Model2_label)
ax2.plot(t, D_s[:maxtime_fit], label='SIRD')
ax2.plot(t, D_sD[:maxtime_fit], label=r'SIRD$_{\Delta D}$')
ax2.plot(t, D_u[:maxtime_fit], label='SIURD')
ax2.plot(t, D_e[:maxtime_fit], label='SEIURD')
ax2.legend()
ax2.set_title('Daily deaths')
ax2.set_ylabel('Number')
ax2.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
ax2.grid(True)
ax2.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
ax4.plot(R_eff_data[:maxtime_fit], 'b', label = r'Synt data $R_0 =$ ' + str( round(R_0_data, 2)))
ax4.plot(R_eff_a[:maxtime_fit]  , label = Model2_label + r' $R_0 =$ ' + str( round(R_0_a, 2)))
ax4.plot(R_eff_s[:maxtime_fit], label = r'SIRD $R_0 = $' + str( round(R_0_s, 2)))
ax4.plot(R_eff_sD[:maxtime_fit], label = r'SIRD$_{\Delta D}$ $R_0 =$ ' + str( round(R_0_sD, 2)))
ax4.plot(R_eff_u[:maxtime_fit], label = r'SIURD $R_0 = $' + str( round(R_0_u, 2)))
ax4.plot(R_eff_e[:maxtime_fit], label = r'SEIURD $R_0 = $' + str( round(R_0_e, 2)))
ax4.legend()
ax4.set_xlabel('Date')
ax4.set_title(r'$\mathcal{R}$')
ax4.set_xticks([x for x in (0, 40, 80, 120) if x < len(date_list)])
ax4.set_xticklabels([date_list[x] for x in (0, 40, 80, 120) if x < len(date_list)])
ax4.grid(True)
plt.tight_layout()
plt.savefig(Figure_name)

fig_path = os.path.join(MODULE_DIR, 'figures', 'saved-plots')
if not os.path.exists(fig_path):
   os.mkdir(fig_path)

files = glob.glob('./*.png')
for f in files:
    shutil.move(f, fig_path)
    
    
