import os

import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR

def plot_figure6A(p, data, filename, plot=False):
    import matplotlib as mpl
    label_size = 24
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size 
    
    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = 50000
    chains = chains[:, burn_in:, :]

    # Show pairwise plots
    parameter_names = [r'$\rho$', r'$I_0$', r'$\alpha_b$', r'$t^{*}$', r'$\beta_{\mu}$', r'$\beta_{\sigma^2}$', r'$\zeta_{\mu}$', r'$\zeta_{\phi}$', r'$\delta$', r'$\phi$']

    # Apply thinning
    chains = chains[:, ::10, :]

    pints.plot.pairwise(chains[0], kde=True, n_percentiles=99, parameter_names=parameter_names)

    # Show graphs
    if plot:
        plt.show()
    else:
        plt.savefig('Figure6A.png')
        