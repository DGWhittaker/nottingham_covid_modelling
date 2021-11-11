import matplotlib.pyplot as plt
import numpy as np
from nottingham_covid_modelling.lib.settings import Params
from scipy.stats import gamma, nbinom


DEBUG = False


def discretise_gamma_distribution(mean, var, timestep, max_infected_age):

    '''Calculates probability mass function (pmf), cumulative distribution function (cdf)
    and survival function of a discretised gamma distribution
    for a given mean, variance, over an interval of [0, max_infected_age] with intervals of 'timestep'.  '''
    if timestep != 1:
        raise NotImplementedError("Current implementation expects a timestep of 1!")

    sc = var / mean  # scale parameter
    a = mean / sc    # shape parameter

    breaks = np.linspace(timestep / 2, max_infected_age + timestep / 2, int(np.rint(max_infected_age / timestep) + 1))

    # this mean the first element of discretised pmf integrates from zero to 3/2 *timestep,
    # consistent with https://arxiv.org/pdf/2004.11342.pdf [middle of p4]
    breaks[0] = 0

    Finc0 = gamma.cdf(breaks, a, loc=0, scale=sc)  # calculates cdf at break points

    f = np.diff(Finc0)   # calculates pmf from the cdf
    F = Finc0[1:]     # adjusts cdf to remove first element
    G = 1 - Finc0[:-1]  # calculates survival function G_j = (1 - F_{j-1}) see Overleaf document

    return f, F, G  # returns pmf, cdf, and survival function

def negative_binomial_distribution(N, p, max_infected_age):
    '''Calculates probability mass function (pmf), cumulative distribution function (cdf)
    and survival function of a negative binomial distribution for a given N and p, [0, max_infected_age]  '''
    
    breaks = np.linspace(0, max_infected_age, max_infected_age+ 1)
    
    f = nbinom.pmf(breaks[:-1], N, p)   # calculates pmf
    F = nbinom.cdf(breaks[:-1], N, p)     # cdf
    G = 1 - np.insert(F[:-1],0,0)  # calculates survival function G_j = (1 - F_{j-1}) starting with G(0), see Overleaf document

    return f, F, G  # returns pmf, cdf, and survival function



def make_rate_vectors(parameters_dictionary, params=Params()):

    ''' Produces rate vectors (i.e. lambda, zeta and gamma) assuming the equivalent
    continuous distribution is a gamma distributions with specified means and variances'''
    if params.timestep != 1:
        raise NotImplementedError("Current implementation expects a timestep of 1!")

    beta_mean = parameters_dictionary.get('beta_mean', params.beta_mean)
    beta_var = parameters_dictionary.get('beta_var', params.beta_var)
    death_mean = parameters_dictionary.get('death_mean', params.death_mean)
    death_dispersion = parameters_dictionary.get('death_dispersion', params.death_dispersion)
    recovery_mean = parameters_dictionary.get('recovery_mean', params.recovery_mean)
    recovery_dispersion = parameters_dictionary.get('recovery_dispersion', params.recovery_dispersion)
    IFR = parameters_dictionary.get('IFR', params.IFR)

    death_N_NB = 1 / death_dispersion
    death_p_NB = 1 / (1 + death_mean * death_dispersion)
    recovery_N_NB = 1 / recovery_dispersion
    recovery_p_NB = 1 / (1 + recovery_mean * recovery_dispersion)

    # Calculate pmf, cdf and survival function for rates:
    betaf, betaF, betaG = discretise_gamma_distribution(beta_mean, beta_var, params.timestep,
                                                           params.max_infected_age)

    deathf, deathF, deathG = negative_binomial_distribution(death_N_NB, death_p_NB, params.max_infected_age)

    recoveryf, recoveryF, recoveryG = negative_binomial_distribution(recovery_N_NB, recovery_p_NB,
                                                                    params.max_infected_age)

    # joint survival function (this is the survival function for remaining "I", i.e. not yet having become a "D" or "R"
    Gjoint = 1 - ((1 - IFR) * (1 - recoveryG) + IFR * (1 - deathG))

    beta = betaf

    gamma = (1 - IFR) * recoveryf / Gjoint  # recovery hazard
    zeta = IFR * deathf / Gjoint  # death hazard

    # amend values (which are of no importance anyway) that are caused by numerical error or divide-by-zero
    gamma[Gjoint < 1e-14] = 1 - IFR
    zeta[Gjoint < 1e-14] = IFR

    # Plots rates and joint survival function
    if DEBUG:
        fig = plt.figure(figsize=(8, 6))
        times = params.timestep * np.linspace(1, np.rint(params.max_infected_age / params.timestep),
                                              int(np.rint(params.max_infected_age / params.timestep)))
        ax1 = fig.add_subplot(321)
        plt.plot(times, beta, 'r-', lw=5, alpha=0.6)
        ax1.set_title(r'Infectiousness profile ($\beta$)')
        ax1.set_xlim([-1, 61])
        plt.grid(True)
        ax2 = fig.add_subplot(322)
        plt.plot(times, Gjoint, 'r-', lw=5, alpha=0.6)
        ax2.set_title(r'Survival function in $I$ compartment')
        ax2.set_xlim([-1, 61])
        plt.grid(True)
        ax3 = fig.add_subplot(323)
        plt.plot(times, deathf, 'r-', lw=5, alpha=0.6)
        ax3.set_title('Infection-to-death distribution')
        ax3.set_xlim([-1, 61])
        plt.grid(True)
        ax4 = fig.add_subplot(324)
        plt.plot(times, zeta, 'r-', lw=5, alpha=0.6)
        ax4.set_title(r'Death hazard ($\zeta$)')
        ax4.set_xlim([-1, 61])
        plt.grid(True)
        ax5 = fig.add_subplot(325)
        plt.plot(times, recoveryf, 'r-', lw=5, alpha=0.6)
        ax5.set_title('Infection-to-recovery distribution')
        ax5.set_xlim([-1, 61])
        ax5.set_xlabel('Time (days)')
        plt.grid(True)
        ax6 = fig.add_subplot(326)
        plt.plot(times, gamma, 'r-', lw=5, alpha=0.6)
        ax6.set_title(r'Recovery hazard ($\gamma$)')
        ax6.set_xlim([-1, 61])
        ax6.set_xlabel('Time (days)')
        plt.grid(True)
        plt.tight_layout()

    gamma = gamma.reshape(1, params.max_infected_age)
    zeta = zeta.reshape(1, params.max_infected_age)

    params.Gjoint = Gjoint

    beta = beta.reshape(1, params.max_infected_age)

    return beta, gamma, zeta


def calculate_R0(p, parameters_dictionary):

    summ = np.zeros(p.maxtime + 1)
    for i in range(p.maxtime - p.numeric_max_age + 1):
        fac = 0
        for j in range(p.numeric_max_age - 1):
            fac += p.Gjoint[j] * p.beta[0, j]
        summ[i] = fac
    rho = parameters_dictionary.get('rho', p.rho)
    if p.extra_days_to_simulate > 0:
        R0 = rho * p.alpha[:-p.extra_days_to_simulate] * summ
    else:
        R0 = rho * p.alpha * summ

    return R0[:-p.numeric_max_age]


def calculate_R_effective(p, susceptibles, parameters_dictionary):

    summ = np.zeros(p.maxtime + 1)
    for i in range(p.maxtime - p.numeric_max_age + 1):
        fac = 0
        for j in range(p.numeric_max_age - 1):
            fac += p.Gjoint[j] * p.beta[0, j] * susceptibles[i + j]
        summ[i] = fac
    rho = parameters_dictionary.get('rho', p.rho)
    if p.extra_days_to_simulate > 0:
        R_eff = rho * p.alpha[:-p.extra_days_to_simulate] * summ / p.N
    else:
        R_eff = rho * p.alpha * summ / p.N

    return R_eff[:-p.numeric_max_age]

def calculate_R_instantaneous(p, susceptibles, parameters_dictionary):
    summ = np.zeros(p.maxtime + 1)
    for i in range(p.maxtime - p.numeric_max_age + 1):
        fac = 0
        for j in range(p.numeric_max_age - 1):
            fac += p.Gjoint[j] * p.beta[0, j]
        summ[i] = fac * susceptibles[i]
    rho = parameters_dictionary.get('rho', p.rho)
    if p.extra_days_to_simulate > 0:
        R_eff = rho * p.alpha[:-p.extra_days_to_simulate] * summ / p.N
    else:
        R_eff = rho * p.alpha * summ / p.N

    return R_eff[:-p.numeric_max_age]
