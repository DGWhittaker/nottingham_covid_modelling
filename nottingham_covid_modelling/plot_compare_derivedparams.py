import os
from nottingham_covid_modelling import MODULE_DIR
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
from scipy.stats import gaussian_kde

file_appends = ['SItD', 'SIR', 'SIRDeltaD', 'SIUR', 'SEIUR']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fixeta = True
syn_data_values = [3.2, 0.79, 32, 295.8*1000]


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title(r'$\mathcal{R}_0$')
ax2 = fig.add_subplot(222)
ax2.set_title(r'min$_i\{\mathcal{R}_i\}$')
ax3 = fig.add_subplot(223)
ax3.set_title(r'argmin$_i\{\mathcal{R}_i<1\}$')
ax4 = fig.add_subplot(224)
ax4.set_title(r'max$_i\{I_{i,1}\}$')

R0 = []
Rmin = []
Rl1 = []
maxI = []
for n, f in enumerate(file_appends):

    if f == 'SEIUR' and fixeta:
        filename = 'Data_SimSItD-1_rho_0-2_NBphi_2e-3_model-SEIUR_square-lockdown_flat-priors_parameters-estimated__rho_Iinit1_theta_xi_lockdown_baseline_lockdown_offset'
    else:
        filename = 'Data_SimSItD-1_rho_0-2_noise-model_NBphi_2e-3_model-' + f + '_full-fit-True_square-lockdown_flat-priors_rho_Init1_lockdown-baseline_lockdown-offset'
    filename_epiparams = os.path.join(MODULE_DIR, 'out-mcmc', filename + '_epiParams_chain_1_burnIn_25000_thinning_10.npy')

    epiparams = np.load(filename_epiparams) 
    R0.append(epiparams[0,:].tolist())
    Rmin.append(epiparams[1,:].tolist())
    Rl1.append(epiparams[2,:].tolist())
    maxI.append(epiparams[3,:].tolist())


    R0_min, R0_max = np.min(R0[n]), np.max(R0[n])
    R0x = np.linspace(R0_min, R0_max, 100)

    Rmin_min, Rmin_max = np.min(Rmin[n]), np.max(Rmin[n])
    Rminx = np.linspace(Rmin_min, Rmin_max, 100)

    Rl1_min, Rl1_max = np.min(Rl1[n]), np.max(Rl1[n])
    Rl1x = np.linspace(Rl1_min, Rl1_max, 6)

    maxI_min, maxI_max = np.min(maxI[n]), np.max(maxI[n])
    maxIx = np.linspace(maxI_min, maxI_max, 100)

    if f in {'SIR', 'SIRDeltaD', 'SIUR', 'SItD'}:
        nbins = 2
    else:
        nbins = 3

    ax1.hist(R0[n], bins=25, density=True, alpha=0.2, color=colors[n], label=f)
    ax1.plot(R0x, gaussian_kde(R0[n])(R0x), color=colors[n])
    ax2.hist(Rmin[n], bins=25, density=True, alpha=0.2, color=colors[n])
    ax2.plot(Rminx, gaussian_kde(Rmin[n])(Rminx), color=colors[n])
    ax3.hist(Rl1[n], bins=nbins, density=True, alpha=0.2, color=colors[n])
    ax4.hist(maxI[n], bins=25, density=True, alpha=0.2, color=colors[n])
    ax4.plot(maxIx, gaussian_kde(maxI[n])(maxIx), color=colors[n])

ax1.legend()
plt.tight_layout()




fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title(r'$\mathcal{R}_0$')
ax2 = fig.add_subplot(222)
ax2.set_title(r'min$_i\{\mathcal{R}_i\}$')
ax3 = fig.add_subplot(223)
ax3.set_title(r'argmin$_i\{\mathcal{R}_i<1\}$')
ax4 = fig.add_subplot(224)
ax4.set_title(r'max$_i\{I_{i,1}\}$')
ax1.axhline(y=syn_data_values[0],  linestyle=':')
ax1.boxplot(R0, labels = file_appends)#, bins=25, density=True, alpha=0.2, label=f)
ax2.axhline(y=syn_data_values[1],  linestyle=':')
ax2.boxplot(Rmin,labels = file_appends)#, bins=25, density=True, alpha=0.2) 
ax3.axhline(y=syn_data_values[2],  linestyle=':')
ax3.boxplot(Rl1, labels = file_appends)#, bins=nbins, density=True, alpha=0.2)
ax4.axhline(y=syn_data_values[3],  linestyle=':')
ax4.boxplot(maxI, labels = file_appends)#, bins=25, density=True, alpha=0.2)
plt.tight_layout()
plt.show()

