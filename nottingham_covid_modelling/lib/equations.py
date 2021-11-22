import numpy as np
#import scipy.linalg as la
from scipy.optimize import fsolve
from nottingham_covid_modelling.lib.ratefunctions import make_rate_vectors


def solve_difference_equations(p, parameters_dictionary, travel_data):

    '''Solves difference equations for SIRD model'''
    
    assert not travel_data or hasattr(p, 'alpha'), "If using travel data, it must be loaded into p using DataLoader"
    
    assert hasattr(p, 'gamma'), "Need to precompute the age-depending rates and loaded into p using store_rate_vectors"

    rho = parameters_dictionary.get('rho', p.rho)
    Iinit1 = parameters_dictionary.get('Iinit1', p.Iinit1)

    time_vector = range(p.maxtime)

    old_max_infected_age = p.max_infected_age
    if p.simple:
        p.max_infected_age = p.c

    # Create arrays for susceptible (S), recovered (R), and deceased (D) individuals
    S, R, D = np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1)

    # Create arrays for daily (Iday) and total (Itot) infecteds
    Iday = np.zeros((p.max_infected_age, p.maxtime + 1))  # Get shape right
    Itot = np.zeros(p.maxtime + 1)

    # Initial number of recovereds and deceased, Rinit and Dinit
    Rinit, Dinit = 0, 0
    
    # Initialise Iday, Itot
    if p.simple:
        for j in range(p.max_infected_age):
            # Approximate distribution as linear over 21 days
            Iday[j, 0] = Iinit1 - j * (Iinit1 / (p.max_infected_age - 1))
    else:
        pseudo_dist = pseudo_IC_dist(p, rho)
        Iday[:, 0] = Iinit1 * pseudo_dist
        #Iday[0, 0] = Iinit1
    Itot[:] = 0
    Iinit = int(np.sum(Iday[:, 0]))

    # Initial number of susceptible (everyone that isn't already infected/recovered/dead)
    Sinit = p.N - (Iinit + Rinit + Dinit)

    # Initialise S, R, D
    S[0], R[0], D[0] = Sinit, Rinit, Dinit

    if travel_data:
        if p.square_lockdown:
            alpha = step(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
        else:
            alpha = tanh_spline(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)

    if p.calculate_rate_vectors:
        p.beta, p.gamma, p.zeta = make_rate_vectors(parameters_dictionary, p)
        # print('making rate vectors...')

    A = np.zeros((p.max_infected_age, p.max_infected_age))
    A[:, 0] = rho * p.beta[0,:]
    for i in range(p.max_infected_age - 1):
        A[ i, i + 1] = 1 - p.gamma[0, i] - p.zeta[0,i]


    for i in time_vector:
        if travel_data:
            A[:, 0] = alpha[i] * rho * p.beta[0,:]
        # Compute total number of infecteds
        Infecteds = Iday[:, i]
        Itot[i] = np.sum(Infecteds)
        Iday[:, i + 1] = Infecteds @ A
        Iday[0, i + 1] = (S[i] / p.N) * Iday[0, i + 1]
        S[i + 1] = S[i] -  Iday[0, i+1] #(S[i] / p.N) * pressure[i]
        R[i + 1] = R[i] + np.dot(p.gamma, Infecteds)
        # Compute death rate per day (number of deceased would be D[i+1] = D[i] + np.dot(zeta, Infecteds))
        D[i + 1] = np.dot(p.zeta, Infecteds)
    Itot[p.maxtime] = np.sum(Iday[:, p.maxtime])

    p.max_infected_age = old_max_infected_age
    
    return S, Iday, R, D, Itot


def get_model_solution(p, parameters_dictionary, travel_data = True):

    '''Returns model solution for given parameters'''

    _, _, _, D, _ = solve_difference_equations(p, parameters_dictionary=parameters_dictionary, travel_data = travel_data)
    return D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]


def tanh_spline(p, lgoog_data, parameters_dictionary, weekends=True):

    d_vec = np.linspace(0, lgoog_data - 1, lgoog_data)
    if not weekends:
        d_vec = [x for i, x in enumerate(d_vec) if not (
            (i % 7 == 0) or (i % 7 == 1))]
    for i in range(p.numeric_max_age + p.extra_days_to_simulate):
        d_vec = np.append(d_vec, lgoog_data + i)

    lockdown_baseline = parameters_dictionary.get('lockdown_baseline', p.lockdown_baseline)
    lockdown_fatigue = parameters_dictionary.get('lockdown_fatigue', p.lockdown_fatigue)
    lockdown_new_normal = parameters_dictionary.get('lockdown_new_normal', p.lockdown_new_normal)
    lockdown_offset = parameters_dictionary.get('lockdown_offset', p.lockdown_offset)
    lockdown_rate = parameters_dictionary.get('lockdown_rate', p.lockdown_rate)

    f = 1.0 - 0.5 * (1 - lockdown_baseline) * (np.tanh(lockdown_rate * (d_vec - lockdown_offset)) + 1)
    idx = np.where(np.heaviside(d_vec - lockdown_offset, 0) == 1)
    f[idx] = \
        f[idx] + (-lockdown_new_normal * (1.0 - lockdown_baseline)) * \
                  (np.exp(-(d_vec[idx] - lockdown_offset) * lockdown_fatigue) - 1.0)

    return f
    

def step(p, lgoog_data, parameters_dictionary):

    d_vec = np.linspace(0, lgoog_data + p.numeric_max_age + p.extra_days_to_simulate - 1, lgoog_data + \
        p.numeric_max_age + p.extra_days_to_simulate)

    lockdown_baseline = parameters_dictionary.get('lockdown_baseline', p.lockdown_baseline)
    lockdown_offset = parameters_dictionary.get('lockdown_offset', p.lockdown_offset)

    f = d_vec
    offset_round = int(lockdown_offset)
    normal_frac = lockdown_offset - offset_round
    for c, d in enumerate(d_vec):
        if d > lockdown_offset:
            f[c] = lockdown_baseline
        elif d == offset_round:
            f[c] = normal_frac + (1 - normal_frac) * lockdown_baseline
        else:
            f[c] = 1.0
    return f


def store_rate_vectors(parameters_dictionary, p):
    ''' Precompute the age-depending vector parameters and store them in p'''
    # Age depending parameters:
    # Infection rate, beta (/days)
    old_max_infected_age = p.max_infected_age
    IFR = parameters_dictionary.get('IFR', p.IFR)
    
    if p.simple:
        p.max_infected_age = p.c
        beta = (1 / p.M) * np.ones((1, p.max_infected_age))
        beta[:, :p.L] = 0
        beta[:, p.L + p.M:] = 0

        # Mean recovery rate, gamma (/days)
        gamma = np.zeros((1, p.max_infected_age))
        gamma[:, - 1] = (1 - IFR)

        # Death rate, zeta (/days)
        zeta = np.zeros((1, p.max_infected_age))
        zeta[:, - 1] = IFR
    else:
        beta, gamma, zeta = make_rate_vectors(parameters_dictionary, p)
  
    p.beta = beta
    p.gamma = gamma
    p.zeta = zeta
    p.max_infected_age = old_max_infected_age
    
    return

def pseudo_IC_dist(p,rho):
    ''' Returns the pseudo st st distribution of Infected individuals at <almost> early infection.
        It assumes that lambda = rho * beta, even for the mobility data, since at early times alpha = 1.
    '''
    #Leah and Simon Version:
    x0 = (1 / p.max_infected_age) * (1 + np.zeros(p.max_infected_age))  # guess of solution required for the solver.
    fparams = [p,rho]
    Pseudo_Leah = fsolve(pseudo_equation, x0, fparams)
    pseudo_dist = Pseudo_Leah / sum(Pseudo_Leah) #normalize solution
    
    # Frank's version : CURRENTLY NOT WORKING
    '''
    A = np.zeros((p.max_infected_age, p.max_infected_age))
    A[:, 0] = p.rho * p.beta[0,:]
    for i in range(p.max_infected_age - 1):
        A[ i, i + 1] = 1 - p.gamma[0, i] - p.zeta[0,i]
    EigVal, Vvectors, Wvectors = la.eig(A,right = True, left = True)
    pseudo_dist = Vvectors.real[:,1] / sum (Vvectors.real[:,1])
    '''
    return pseudo_dist

def pseudo_equation(x,fparams):

    p = fparams[0]
    rho = fparams[1]
    lamba = rho * p.beta[0, :]
    F = np.zeros(p.max_infected_age)
    for j in range(p.max_infected_age - 1):
        F[j] = (x[j + 1] / x[0]) - (x[j] * (1 - p.gamma[0, j] - p.zeta[0, j]) / np.dot(lamba, x))
    F[-1:] = sum(x) - 1

    return F


def solve_SIR_difference_equations(p, parameters_dictionary, travel_data):

    '''Solves difference equations for basic SIRD model without infection age distribution'''
    
    assert not travel_data or hasattr(p, 'alpha'), "If using travel data, it must be loaded into p using DataLoader"
    
    assert hasattr(p, 'theta'), "Need to pre-define the rates"
    assert np.isscalar(p.beta), "The rate beta needs to be a scalar for this model"
    assert np.isscalar(p.theta), "The rate theta needs to be a scalar for this model"

    rho = parameters_dictionary.get('rho', p.rho)
    Iinit = parameters_dictionary.get('Iinit1', p.Iinit1)
    theta = parameters_dictionary.get('theta',p.theta)
    beta = parameters_dictionary.get('beta',p.beta)
    DeltaD = parameters_dictionary.get('DeltaD',p.DeltaD)
    DeltaD = int(DeltaD)
    
    time_vector = range(p.maxtime)

    # Create arrays for susceptible (S), infected (I), new infections (Inew) recovered (R), and deceased (D) individuals
    S, I, Inew, R, D = np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1 + DeltaD)

    # Initial conditions
    I[0], R[0], D[0] = Iinit, 0, 0
    Inew[0] = Iinit
    S[0] = p.N - (I[0] + R[0] + D[0])
    
    # travel data
    if travel_data:
        if p.square_lockdown:
            alpha = step(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
        else:
            alpha = tanh_spline(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
    else:
        alpha = np.ones(p.maxtime)
        
        
    for i in time_vector:
        Inew[i + 1] = ((S[i] / p.N) * rho * alpha[i] * beta * I[i])
        I[i + 1] = I[i] + Inew[i + 1] - theta * I[i]
        S[i + 1] = S[i] - Inew[i + 1]
        # Compute death and recovery per day
        R[i + 1] = theta * (1 - p.IFR) * I[i]
        D[i + DeltaD + 1] = theta * p.IFR *I[i]
        if S[i+1]<0:
            S[i+1] =0
        
    return S, I, Inew, R, D


def solve_SIUR_difference_equations(p, parameters_dictionary, travel_data):

    '''Solves difference equations for SIURD model without infection age distribution'''

    assert not travel_data or hasattr(p, 'alpha'), "If using travel data, it must be loaded into p using DataLoader"

    assert hasattr(p, 'theta'), "Need to pre-define the rates"
    assert np.isscalar(p.beta), "The rate beta needs to be a scalar for this model"
    assert np.isscalar(p.theta), "The rate theta needs to be a scalar for this model"
    assert np.isscalar(p.xi), "The rate xi needs to be a scalar for this model"

    rho = parameters_dictionary.get('rho', p.rho)
    Iinit = parameters_dictionary.get('Iinit1', p.Iinit1)
    theta = parameters_dictionary.get('theta',p.theta)
    beta = parameters_dictionary.get('beta',p.beta)
    xi = parameters_dictionary.get('xi',p.xi)

    time_vector = range(p.maxtime)

    # Create arrays for susceptible (S), infectious (I) and new infections (Inew), non-infectious infected (N), recovered (R), and deceased (D) individuals
    S, I, Inew, U, R, D = np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1)

    # Initial conditions
    I[0], R[0], D[0], U[0] = Iinit, 0, 0, 0
    Inew[0] = Iinit
    S[0] = p.N - (I[0] + R[0] + D[0] + U[0])

    # travel data
    if travel_data:
        if p.square_lockdown:
            alpha = step(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
        else:
            alpha = tanh_spline(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
    else:
        alpha = np.ones(p.maxtime)
    
    
    for i in time_vector:
        Inew[i + 1] = ((S[i] / p.N) * rho * alpha[i] * beta * I[i])
        I[i + 1] = I[i] + Inew[i + 1] - theta * I[i]
        S[i + 1] = S[i] - Inew[i + 1]
        U[i + 1] = U[i] + theta * I[i] -  xi * U[i]
        # Compute death and recovery per day
        R[i + 1] = xi * (1 - p.IFR) * U[i]
        D[i + 1] = xi * p.IFR * U[i]
        if S[i+1]<0:
            S[i+1] =0
    
    return S, I,Inew, U, R, D


def solve_SEIUR_difference_equations(p, parameters_dictionary, travel_data):

    '''Solves difference equations for SEIURD model with E->I->U_> Erlang transtiions'''

    assert not travel_data or hasattr(p, 'alpha'), "If using travel data, it must be loaded into p using DataLoader"

    assert hasattr(p, 'theta'), "Need to pre-define the rates"
    assert np.isscalar(p.beta), "The rate beta needs to be a scalar for this model"
    assert np.isscalar(p.eta), "The rate eta needs to be a scalar for this model"
    assert np.isscalar(p.theta), "The rate theta needs to be a scalar for this model"
    assert np.isscalar(p.xi), "The rate xi needs to be a scalar for this model"
    

    rho = parameters_dictionary.get('rho', p.rho)
    Iinit = parameters_dictionary.get('Iinit1', p.Iinit1)
    theta = parameters_dictionary.get('theta',p.theta)
    beta = parameters_dictionary.get('beta',p.beta)
    eta = parameters_dictionary.get('eta',p.eta)
    xi = parameters_dictionary.get('xi',p.xi)

    time_vector = range(p.maxtime)

    # Create arrays for susceptible (S),  new infections (Inew),  recovered (R), and deceased (D) individuals
    S, Inew, Enew, R, D =  np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1)
    # Create arrays for the Erlang dist compartments: E1, E2 (expossed), I1, I2 (infected), U1, U2 (non infectious)
    E1, E2, I1, I2, U1, U2 = np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1), np.zeros(p.maxtime + 1)

    # Initial conditions
    I1[0], I2[0] = Iinit, 0
    E1[0], E2[0], U1[0], U2[0] =  0, 0, 0, 0
    R[0], D[0] =  0, 0
    Enew[0] = 0
    Inew[0] = Iinit
    S[0] = p.N - (I1[0] + I2[0] + E1[0] + E2[0] + R[0] + D[0] + U1[0] + U2[0])

    # travel data
    if travel_data:
        if p.square_lockdown:
            alpha = step(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
        else:
            alpha = tanh_spline(p, lgoog_data=len(p.alpha), parameters_dictionary=parameters_dictionary)
    else:
        alpha = np.ones(p.maxtime)
    
    
    for i in time_vector:
        Enew[i + 1] = ((S[i] / p.N) * rho * alpha[i] * beta * (I1[i] + I2[i]))
        S[i + 1] = S[i] - Enew[i + 1]
        E1[i + 1] = E1[i] + Enew[i + 1] - 2 * eta * E1[i]
        Inew[i + 1] = 2 * eta * E2[i]
        E2[i + 1] = E2[i] + 2 * eta * E1[i] - 2 * eta * E2[i]
        I1[i + 1] = I1[i] + 2 * eta * E2[i] - 2 * theta * I1[i]
        I2[i + 1] = I2[i] + 2 * theta * I1[i] - 2 * theta * I2[i]
        U1[i + 1] = U1[i] + 2 * theta * I2[i] -  2 * xi * U1[i]
        U2[i + 1] = U2[i] + 2 * xi * U1[i] -  2 * xi * U2[i]
        # Compute death and recovery per day
        R[i + 1] = 2 * xi * (1 - p.IFR) * U2[i]
        D[i + 1] = 2 * xi * p.IFR * U2[i]
        if S[i+1]<0:
            S[i+1] =0
    
    return S, E1, E2, Enew, I1, I2, Inew, U1, U2, R, D

def get_model_SIR_solution(p, parameters_dictionary, travel_data = True):

    '''Returns model solution for given parameters, including DeltaD for basic SIRD model without infection age distribution'''

    _, _, _, _, D = solve_SIR_difference_equations(p, parameters_dictionary=parameters_dictionary, travel_data=travel_data)
    DeltaD = parameters_dictionary.get('DeltaD', p.DeltaD)
    return D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate + int(DeltaD))]

def get_model_SIUR_solution(p, parameters_dictionary, travel_data = True):

    '''Returns model solution for SIUR model for given parameters'''

    _, _, _, _, _, D = solve_SIUR_difference_equations(p, parameters_dictionary=parameters_dictionary, travel_data = travel_data)
    return D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]

def get_model_SEIUR_solution(p, parameters_dictionary, travel_data = True):

    '''Returns model solution for SIUR model for given parameters'''

    _, _, _, _, _, _, _, _, _, _, D = solve_SEIUR_difference_equations(p, parameters_dictionary=parameters_dictionary, travel_data = travel_data)
    return D[p.day_1st_death_after_150220: -(p.numeric_max_age + p.extra_days_to_simulate)]
