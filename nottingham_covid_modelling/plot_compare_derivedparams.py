import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
from scipy.stats import gaussian_kde

file_appends = ['SIR', 'SItD', 'SIUR', 'SEIUR', 'SIRDeltaD']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title(r'$\mathcal{R}_0$')
ax2 = fig.add_subplot(222)
ax2.set_title(r'min$_i\{\mathcal{R}_i\}$')
ax3 = fig.add_subplot(223)
ax3.set_title(r'argmin$_i\{\mathcal{R}_i<1\}$')
ax4 = fig.add_subplot(224)
ax4.set_title(r'max$_i\{I_{i,1}\}$')

for n, f in enumerate(file_appends):

    R0 = np.load('../R0_samples_' + f + '.npy')
    Rmin = np.load('../Rmin_samples_' + f + '.npy')
    Rl1 = np.load('../Rl1_samples_' + f + '.npy')
    maxI = np.load('../maxI_samples_' + f + '.npy')

    R0_min, R0_max = np.min(R0), np.max(R0)
    R0x = np.linspace(R0_min, R0_max, 100)

    Rmin_min, Rmin_max = np.min(Rmin), np.max(Rmin)
    Rminx = np.linspace(Rmin_min, Rmin_max, 100)

    Rl1_min, Rl1_max = np.min(Rl1), np.max(Rl1)
    Rl1x = np.linspace(Rl1_min, Rl1_max, 6)

    maxI_min, maxI_max = np.min(maxI), np.max(maxI)
    maxIx = np.linspace(maxI_min, maxI_max, 100)

    if f in {'SIR', 'SIRDeltaD', 'SIUR', 'SItD'}:
        nbins = 2
    else:
        nbins = 3

    ax1.hist(R0, bins=25, density=True, alpha=0.2, color=colors[n], label=f)
    ax1.plot(R0x, gaussian_kde(R0)(R0x), color=colors[n])
    ax2.hist(Rmin, bins=25, density=True, alpha=0.2, color=colors[n])
    ax2.plot(Rminx, gaussian_kde(Rmin)(Rminx), color=colors[n])
    ax3.hist(Rl1, bins=nbins, density=True, alpha=0.2, color=colors[n])
    ax4.hist(maxI, bins=25, density=True, alpha=0.2, color=colors[n])
    ax4.plot(maxIx, gaussian_kde(maxI)(maxIx), color=colors[n])

ax1.legend()
plt.tight_layout()
plt.show()

