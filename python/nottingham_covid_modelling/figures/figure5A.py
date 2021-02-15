import os

import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.io
import pints.plot
from nottingham_covid_modelling import MODULE_DIR

def plot_figure5A(p, data, filename, plot=False):
    saveas = os.path.join(MODULE_DIR, 'out-mcmc', filename)
    chains = pints.io.load_samples(saveas + '-chain.csv', 3)

    chains = np.array(chains)

    niter = len(chains[1])

    # Discard burn in
    burn_in = 25000
    chains = chains[:, burn_in:, :]

    # Show pairwise plots
    parameter_names = [r'$\rho$', r'$I_0$', r'$\alpha_b$', r'$t^{*}$', r'$\delta$', r'$\phi$']

    # Apply thinning
    chains = chains[:, ::10, :]

    pints.plot.pairwise(chains[0], kde=True, n_percentiles=99, parameter_names=parameter_names)

    # Show graphs
    if plot:
        plt.show()
    else:
        plt.savefig('Figure5A.png')
        