
class Params(object):
    def __init__(self):

        '''Fixed parameter settings used for simulations'''

        # Time points (in days)
        # Simulate roughly 7 months. If Google data is being used this is overwritten in get_google_data function
        self.maxtime = 240

        self.c = 21  # Time after which individual recovers/dies
        self.day_1st_death_after_150220 = 0  # Default is 0 from global deaths csv
        self.extra_days_to_simulate = 0  # Extra time to simulate, must be >= 0
        self.numeric_max_age = 133
        
        # Google data transit row flag
        self.transit = True

        # Useful numbers
        self.N = 70000000  # Total population, N
        self.L = 4  # Latent period before individual becomes infectious
        self.M = 6  # Infectious for 6 days according to Ferguson SI fig.
        self.IFR = 0.01 #0.0114  # Calculated based on UK data

        # The following lines define the parameters, for rate vectors matching overleaf
        self.beta_mean = 5.2
        self.beta_var = 2.96
        self.death_mean = 18.69 #21.857
        self.death_dispersion = 0.0546 #0.11765
        self.recovery_mean = 150 #23.726 #28.435
        self.recovery_dispersion = 1e-12 #0.0342 #0.05988
        self.timestep = 1  # number of days between time step i+1 and i

        # Default parameter values
        self.rho = 2.4
        self.Iinit1 = 1000
        self.lockdown_offset = 15
        self.lockdown_rate = 0.15
        self.lockdown_fatigue = 1e-3
        self.lockdown_baseline = 0.5
        self.lockdown_new_normal = 1.0
        self.fixed_phi = 0.01
        self.fixed_sigma = 60

        # maximum number of days for which to compute the rates (should be sufficiently large
        # that the survival probability for remaining an "I" is pretty much zero after this number of days)
        self.max_infected_age = 133

        self.simple = False
        self.alpha1 = False
        self.square_lockdown = False
        self.calculate_rate_vectors = False
        self.gamma_dist_params = ['beta_mean', 'beta_var', 'death_mean', 'death_dispersion', \
        'recovery_mean', 'recovery_dispersion', 'IFR']
        self.n_days_to_simulate_after_150220 = 0
        self.five_param_spline = False
        self.fix_phi = False
        self.fix_sigma = False
        self.flat_priors = False

def get_file_name_suffix(p, country, noise, parameters_list):

    filename = 'Data_' + str(country) + '_noise-model_' + noise

    if p.simple:
        filename = filename + '_simple-rates'
    if p.fix_phi:
        filename = filename + '_fixed-phi-' + str(p.fixed_phi)
    if p.fix_sigma:
        filename = filename + '_fixed-sigma-' + str(p.fixed_sigma)
    if p.alpha1:
        filename = filename + '_alpha1'
    if p.square_lockdown:
        filename = filename + '_square-lockdown'
    if p.flat_priors:
        filename = filename + '_flat-priors'
    if p. five_param_spline:
        filename = filename + '_five-param-spline'
    if p.n_days_to_simulate_after_150220 > 0:
        filename = filename + '_n-days-after-150220-' + \
        str(p.n_days_to_simulate_after_150220) 
    
    for i in parameters_list:
        if i == 'rho':
            filename = filename + '_rho'
        if i == 'Iinit1':
            filename = filename + '_Init1'
        if i == 'lockdown_baseline':
            filename = filename + '_lockdown-baseline'
        if i == 'lockdown_offset':
            filename = filename + '_lockdown-offset'
        if i == 'beta_var':
            filename = filename + '_beta-var'
        if i == 'beta_mean':
            filename = filename + '_beta-mean'
        if i == 'death_mean':
            filename = filename + '_death-mean'
        if i == 'death_dispersion':
            filename = filename + '_death-dispersion'
        if i == 'recovery_mean':
            filename = filename + '_recovery-mean'
        if i == 'recovery_dispersion':
            filename = filename + '_recovery-dispersion'
        if i == 'IFR':
            filename = filename + '_IFR'
        if not p.square_lockdown:
            if i == 'lockdown_fatigue':
                filename = filename + '_lockdown-fatigue'
            if i == 'lockdown_new_normal':
                filename = filename + '_lockdown-new-normal'
            if i == 'lockdown_rate':
                filename = filename + '_lockdown-rate'

    return filename


