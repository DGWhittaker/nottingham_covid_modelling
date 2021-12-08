import numpy as np
import pints
from scipy.special import xlogy, gammaln
from scipy.stats import norm, gamma

DEBUG = False


class LogPrior(pints.LogPrior):
    """
    Boundary constraints on the parameters
    """
    def __init__(self, noise_model, upper_gaussian_noise_sigma, model_name = 'SItD'):
        super(LogPrior, self).__init__()
        assert model_name in ['SIR', 'SIRDeltaD', 'SItD', 'SIUR', 'SEIUR'], "Unknown model"
        self.noise_model = noise_model
        self.upper_gaussian_noise_sigma = upper_gaussian_noise_sigma

        self.lower, self.upper = [], []
        if model_name == 'SItD':
            self._lower_rho, self._upper_rho = 1, 10
            self._lower_Iinit1, self._upper_Iinit1 = 1, 5e7
        else:
            self._lower_rho, self._upper_rho = 0, 2
            self._lower_Iinit1, self._upper_Iinit1 = 1, 5e4
        self._lower_lockdown_baseline, self._upper_lockdown_baseline = 0, 1
        self._lower_lockdown_fatigue, self._upper_lockdown_fatigue = 0, 0.1
        self._lower_lockdown_offset, self._upper_lockdown_offset = 0, 100
        self._lower_lockdown_rate, self._upper_lockdown_rate = 0.01, 1
        self._lower_lockdown_new_normal, self._upper_lockdown_new_normal = 0, 1
        self._lower_beta_mean, self._upper_beta_mean = 3, 9
        self._lower_beta_var, self._upper_beta_var = 0.5, 18
        self._lower_death_mean, self._upper_death_mean = 5, 40
        self._lower_death_dispersion, self._upper_death_dispersion = 1e-12, 0.45
        self._lower_IFR, self._upper_IFR = 0, 1
        self._lower_gaussian_noise_sigma, self._upper_gaussian_noise_sigma = 0, self.upper_gaussian_noise_sigma
        self._lower_negative_binomial_phi, self._upper_negative_binomial_phi = 1e-12, 1
        self._lower_theta, self._upper_theta = 0,1
        self._lower_DeltaD, self._upper_DeltaD = 0, 50
        self._lower_xi, self._upper_xi = 0, 1
        if model_name == 'SEIUR':
            self._lower_eta, self._upper_eta = 0, .5
            self._lower_theta, self._upper_theta = 0, .5
            self._lower_xi, self._upper_xi = 0, .5

        for parameter_name in self.noise_model.parameter_labels:
            assert hasattr(self, '_lower_' + parameter_name), "Unknown parameter %s" % parameter_name
            self.lower = np.append(self.lower, getattr(self, '_lower_' + parameter_name))
            self.upper = np.append(self.upper, getattr(self, '_upper_' + parameter_name))

        # Deal with distribution parameter stuff
        self.threshold = 1e-3

        # Deal with gamma distribution prior for infectiousness profile
        if 'beta_mean' in self.noise_model.parameter_labels:
            self.beta_mean_shape = 16
            self.beta_mean_rate = 3
            self.beta_mean_constant = xlogy(self.beta_mean_shape, self.beta_mean_rate) - gammaln(self.beta_mean_shape)
        if 'beta_var' in self.noise_model.parameter_labels:
            self.beta_var_shape = 2
            self.beta_var_rate = 0.4545
            self.beta_var_constant = xlogy(self.beta_var_shape, self.beta_var_rate) - gammaln(self.beta_var_shape)

        # Deal with normal distribution priors
        self.IFR_mean, self.IFR_sd = 0.00724, 0.001
        self.lo_mean, self.lo_sd = 31.0, 3.0
        self.dm_mean, self.dm_sd = 20.0, 3.0
        self.dd_mean, self.dd_sd = 0.03, 0.025
        if 'IFR' in self.noise_model.parameter_labels:
            self.IFR_offset, self.IFR_factor = self.cache_normal_dist_constants(self.IFR_sd)
        if 'lockdown_offset' in self.noise_model.parameter_labels:
            self.lo_offset, self.lo_factor = self.cache_normal_dist_constants(self.lo_sd)
        if 'death_mean' in self.noise_model.parameter_labels:
            self.dm_offset, self.dm_factor = self.cache_normal_dist_constants(self.dm_sd)
        if 'death_dispersion' in self.noise_model.parameter_labels:
            self.dd_offset, self.dd_factor = self.cache_normal_dist_constants(self.dd_sd)
        
        self.n_params = len(self.lower)
        self.minf = -float('inf')
        self.flat_priors = self.noise_model.flat_priors

    def cache_normal_dist_constants(self, sd):

        # Cache constants
        offset = np.log(1 / np.sqrt(2 * np.pi * sd ** 2))
        factor = 1 / (2 * sd ** 2)
        return offset, factor

    def evaluate_normal_logpdf(self, offset, factor, mean, x):
        return offset - factor * (x - mean)**2

    def n_parameters(self):
        return self.n_params

    def __call__(self, parameters):

        self.prior_weight = 0

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if DEBUG:
                print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if DEBUG:
                print('Upper')
            return self.minf

        # Return 0 if using flat priors
        if self.flat_priors:
            return 0

        # Deal with normal distribution priors
        if 'IFR' in self.noise_model.parameter_labels:
            x = parameters[self.noise_model.parameter_labels.index('IFR')]
            self.prior_weight += self.evaluate_normal_logpdf(self.IFR_offset, self.IFR_factor, self.IFR_mean, x)
        if 'lockdown_offset' in self.noise_model.parameter_labels:
            x = parameters[self.noise_model.parameter_labels.index('lockdown_offset')]
            self.prior_weight += self.evaluate_normal_logpdf(self.lo_offset, self.lo_factor, self.lo_mean, x)
        if 'death_mean' in self.noise_model.parameter_labels:
            x = parameters[self.noise_model.parameter_labels.index('death_mean')]
            self.prior_weight += self.evaluate_normal_logpdf(self.dm_offset, self.dm_factor, self.dm_mean, x)
        if 'death_dispersion' in self.noise_model.parameter_labels:
            x = parameters[self.noise_model.parameter_labels.index('death_dispersion')]
            self.prior_weight += self.evaluate_normal_logpdf(self.dd_offset, self.dd_factor, self.dd_mean, x)

        # Deal with gamma distribution priors
        if 'beta_mean' in self.noise_model.parameter_labels:
            x = parameters[self.noise_model.parameter_labels.index('beta_mean')]
            self.prior_weight += (self.beta_mean_constant + xlogy(self.beta_mean_shape - 1., x) - self.beta_mean_rate * x)
        if 'beta_var' in self.noise_model.parameter_labels:
            x = parameters[self.noise_model.parameter_labels.index('beta_var')]
            self.prior_weight += (self.beta_var_constant + xlogy(self.beta_var_shape - 1., x) - self.beta_var_rate * x)

        return self.prior_weight

    def _sample_distribution(self, para1, para2):

        debug = False
        a1 = getattr(self, '_lower_' + para1)
        a2 = getattr(self, '_upper_' + para1)
        b1 = getattr(self, '_lower_' + para2)
        b2 = getattr(self, '_upper_' + para2)

        if debug:
            success_a, success_b, failure_a, failure_b = [], [], [], []

        for i in range(100):
            a = np.random.uniform(a1, a2)
            b = np.random.uniform(b1, b2)

            if self.acceptance_criterion(a, b) <= 1:
                if debug:
                    print('a = ' + str(a) + ', b = ' + str(b))
                    success_a.append(a)
                    success_b.append(b)
                else:
                    return a, b
            else:
                if debug:
                    failure_a.append(a)
                    failure_b.append(b)

        if debug:
            np.savetxt('samples_accepted.txt', np.c_[success_a, success_b])
            np.savetxt('samples_rejected.txt', np.c_[failure_a, failure_b])

        raise ValueError('Too many iterations failed acceptance criterion')

    def _debug_normal(self):

        a_s, b_s = [], []
        for i in range(10000):
            b = np.random.normal(self.IFR_mean, self.IFR_sd)
            a_s.append(b)
            b_s.append(self.IFR_offset - self.IFR_factor * (b - self.IFR_mean)**2)    

        np.savetxt('samples_gaussian.txt', np.c_[a_s, np.exp(b_s)])
        np.savetxt('samples_gaussian_log.txt', np.c_[a_s, b_s])

        raise ValueError('Too many iterations')

    def _sample_gamma_distribution(self, shape, scale, lower, upper):

        for i in range(100):
            b = np.random.gamma(shape, scale)
            if lower <= b <= upper:
                return b
        raise ValueError('Too many iterations of sampling gamma distribution')

    def sample(self, n=1):

        if n != 1:
            raise NotImplementedError

        p = []  # Sample parameters

        for parameter_name in self.noise_model.parameter_labels:
            if self.flat_priors:
                # Use uniform priors for every parameter
                p1 = getattr(self, '_lower_' + parameter_name)
                p2 = getattr(self, '_upper_' + parameter_name)
                p.append(np.random.uniform(p1, p2))
            else:
                # Deal first with uniform priors
                if parameter_name not in {'beta_mean', 'beta_var', 'death_mean', 'death_dispersion', \
                'IFR', 'lockdown_offset'}:
                    p1 = getattr(self, '_lower_' + parameter_name)
                    p2 = getattr(self, '_upper_' + parameter_name)
                    p.append(np.random.uniform(p1, p2))
                # Gamma distribution priors
                if parameter_name == 'beta_mean':
                    p.append(self._sample_gamma_distribution(self.beta_mean_shape, 1 / self.beta_mean_rate, \
                        self._lower_beta_mean, self._upper_beta_mean))
                if parameter_name == 'beta_var':
                    p.append(self._sample_gamma_distribution(self.beta_var_shape, 1 / self.beta_var_rate, \
                        self._lower_beta_var, self._upper_beta_var))
                # Normal distribution priors
                if parameter_name == 'death_mean':
                    p.append(np.random.normal(self.dm_mean, self.dm_sd))
                if parameter_name == 'death_dispersion':
                    p.append(np.random.normal(self.dd_mean, self.dd_sd))
                if parameter_name == 'IFR':
                    p.append(np.random.normal(self.IFR_mean, self.IFR_sd))
                if parameter_name == 'lockdown_offset':
                    p.append(np.random.normal(self.lo_mean, self.lo_sd))

        assert len(p) == len(self.noise_model.parameter_labels), ('Mismatch between parameter vector and label list '
                                                                  'length')
        p = np.array(p)

        # The Boundaries interface requires a matrix ``(n, n_parameters)``
        p.reshape(1, self.n_params)
        
        return p

    def acceptance_criterion(self, a, b):
        return 1 / (self.threshold**b) - a * b

    def get_priors(self, parameter_name):

        lower = getattr(self, '_lower_' + parameter_name)
        upper = getattr(self, '_upper_' + parameter_name)    

        return lower, upper

    def get_normal_prior(self, parameter_name):

        means = {'IFR': self.IFR_mean, 'death_mean': self.dm_mean, 'death_dispersion': self.dd_mean,\
            'lockdown_offset': self.lo_mean}
        sds = {'IFR': self.IFR_sd, 'death_mean': self.dm_sd, 'death_dispersion': self.dd_sd,\
            'lockdown_offset': self.lo_sd}

        lower, upper = self.get_priors(parameter_name)
        x = np.linspace(lower, upper, 100)
        pdf = norm.pdf(x, loc=means[parameter_name], scale=sds[parameter_name])

        return x, pdf

    def get_gamma_prior(self, parameter_name):

        lower, upper = self.get_priors(parameter_name)
        x = np.linspace(lower, upper, 100)
        if parameter_name == 'beta_mean':
            pdf = gamma.pdf(x, self.beta_mean_shape, loc=0, scale=1/self.beta_mean_rate)
        if parameter_name == 'beta_var':
            pdf = gamma.pdf(x, self.beta_var_shape, loc=0, scale=1/self.beta_var_rate)

        return x, pdf

def get_finite_LL_starting_point(log_prior, log_posterior):
    
    for j in range(100):
        x0 = log_prior.sample()
        if np.isfinite(log_posterior(x0)):
            return x0
    raise ValueError('Too many iterations')

def get_good_starting_point(log_prior, log_posterior, niterations):
    
    x0_best = -1e9
    print('Finding good starting point...')
    for j in range(niterations):
        x0 = log_prior.sample()
        score = log_posterior(x0)
        if np.isfinite(score):
            if score > x0_best:
                print('Best likelihood: ' + str(score))
                x0_best = score
                p0 = x0
    print('Starting point: ' + str(p0))
    return p0
