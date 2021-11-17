from time import monotonic
import numpy as np
import pints
from nottingham_covid_modelling.lib.equations import get_model_solution, store_rate_vectors
from scipy.stats import nbinom, norm, poisson

debug = False
fix_num = 1e-12

class _LogLikelihood(pints.LogPDF):
    def __init__(self, settings, deaths,
                 parameters_to_optimise=['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_fatigue'], 
                 travel_data=True, model_func = get_model_solution):
        self._p = settings
        self._y = deaths
        self._travel_data = travel_data
        self.parameter_labels = parameters_to_optimise
        self.flat_priors = self._p.flat_priors
        self.model_func = model_func

    def __call__(self, x):

        '''
        Preparation for returning log-likelihood

        :returns params_dict, m, y where param_dict is a dict of parameters, m the model solution, y the observed data
        '''
        assert len(x) == len(self.parameter_labels), 'Mismatch between parameter vector and label list length'

        param_dict = dict(zip(self.parameter_labels, x))
        if self.model_func == get_model_solution:
            store_rate_vectors(param_dict, self._p)
        m = self.model_func(self._p, param_dict, self._travel_data)
        assert len(m) == len(self._y), \
            "Mismatch between deaths and model_solution, have you used the DataLoader to load number of deaths?"
        return param_dict, m, self._y

    def n_parameters(self):
        return len(self.parameter_labels)


class Gauss_LogLikelihood(_LogLikelihood):
    def __init__(self, settings, deaths,
                 parameters_to_optimise=['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_fatigue'], 
                 travel_data = True, model_func = get_model_solution):
        super().__init__(settings, deaths, parameters_to_optimise + ['gaussian_noise_sigma'], travel_data, model_func)
        if self._p.fix_sigma:
            self.parameter_labels.remove('gaussian_noise_sigma')

    def __call__(self, x):

        '''
        Preparation for returning log-likelihood

        :returns params_dict, m where param_dict is a dict of parameters and m is the model solution
        '''
        param_dict, m, y = super().__call__(x)
        sigma = param_dict.get('gaussian_noise_sigma', self._p.fixed_sigma)
        f = np.sum(norm.logpdf(y - m, scale=sigma))

        return f


class NegBinom_LogLikelihood(_LogLikelihood):
    def __init__(self, settings, deaths,
                 parameters_to_optimise=['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_fatigue'], 
                 travel_data=True, model_func = get_model_solution):
        super().__init__(settings, deaths, parameters_to_optimise + ['negative_binomial_phi'], travel_data, model_func)
        if self._p.fix_phi:
            self.parameter_labels.remove('negative_binomial_phi')

    def __call__(self, x):

        '''
        Returns negative binomial log-likelihood

        m is model solution
        y is observed data
        phi is dispersion parameter to be optimised, such that variance = m + phi * m**2
        '''
        param_dict, m, y = super().__call__(x)
        f = 0

        fix_flag = False
        NB_phi = param_dict.get('negative_binomial_phi', self._p.fixed_phi)
        for i in range(len(m)):
            mu = m[i]
            if mu < fix_num and y[i] > 0:
                mu = fix_num
                if not fix_flag and debug:
                    print('WARNING: Numerical fix activated.\nModel solution prevented from going below ' + \
                        str(fix_num) + ' to avoid infinite log-likelihood.')
                fix_flag = True
            n = 1 / NB_phi
            p = 1 / (1 + mu * NB_phi)
            f += nbinom.logpmf(y[i], n, p)

        return f


class Poiss_LogLikelihood(_LogLikelihood):
    def __call__(self, x):

        '''
        Returns Poisson log-likelihood

        m is model solution
        y is observed data
        Poisson distribution assumes variance = m
        '''
        param_dict, m, y = super().__call__(x)

        fix_flag = False
        for i in range(len(m)):
            mu = m[i]
            if mu < fix_num and y[i] > 0:
                mu = fix_num
                if not fix_flag and debug:
                    print('WARNING: Numerical fix activated.\nModel solution prevented from going below ' + \
                        str(fix_num) + ' to avoid infinite log-likelihood.')
                fix_flag = True

        f = np.sum(poisson.logpmf(y, m))
        return f

