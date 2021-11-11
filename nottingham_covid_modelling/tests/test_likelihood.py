import os

import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood, Poiss_LogLikelihood
from nottingham_covid_modelling.lib.settings import Params


def setup_function(function):
    """ setup printing options.
    """
    np.set_printoptions(precision=5, suppress=True)  # reduce float precision for comparison across systems


@pytest.fixture(scope='session')
def reference_folder():
    """ Returns the path were the reference files for tests are stored in an OS independent way."""
    return os.path.join(MODULE_DIR, 'tests', 'reference_files')


@pytest.fixture(scope='session')
def settings():
    return Params()


@pytest.fixture(scope='session')
def data(reference_folder, settings):
    return DataLoader(True, settings, 'United Kingdom', data_dir=reference_folder)


@pytest.fixture(scope='session')
def parameter_labels():
    return ['R_0', 'Iinit1', 'lockdown_baseline', 'lockdown_fatigue', 'lockdown_offset', 'lockdown_rate']


@pytest.fixture(scope='session')
def parameter_values():
    return [2.4, 1000, 0.5, 1e-3, 15, 0.15]


def test_label_param_mismatch(settings, data, parameter_labels, parameter_values):
    likelihood = Poiss_LogLikelihood(settings, data.daily_deaths)
    with pytest.raises(AssertionError, match="Mismatch between parameter vector and label list length"):
        likelihood(parameter_values + [1, 2, 3, 4, 5])


def test_deaths_mismatch(settings, data, parameter_labels, parameter_values):
    likelihood = Poiss_LogLikelihood(settings, data.daily_deaths + [0])
    with pytest.raises(AssertionError, match=("Mismatch between parameter vector and label list length")):
        likelihood(parameter_values[:5])


def test_gauss_default(settings, data, parameter_values):
    likelihood = Gauss_LogLikelihood(settings, data.daily_deaths)
    assert likelihood.n_parameters() == 5
    value = np.array([likelihood(parameter_values[:4] + [0.23])])
    assert np.allclose(value, [-7.9832e+08], rtol=1e-2, atol=1e-2)
    

def test_negbinom_default(settings, data, parameter_values):
    likelihood = NegBinom_LogLikelihood(settings, data.daily_deaths)
    assert likelihood.n_parameters() == 5
    # Exact results differ on Windows and Linux. Wrapping the result in numpy array
    # so we can take advantage of precision limiting for output checking purposes
    value = np.array([likelihood(parameter_values[:4] + [0.23])])
    assert np.allclose(value, [-743.91954], rtol=1e-2, atol=1e-2)


def test_poiss_default(settings, data, parameter_values):
    likelihood = Poiss_LogLikelihood(settings, data.daily_deaths)
    assert likelihood.n_parameters() == 4
    value = np.array([likelihood(parameter_values[:4])])
    assert np.allclose(value, [-37232.2979], rtol=1e-2, atol=1e-2)


def test_gauss(settings, data, parameter_labels, parameter_values):
    likelihood = Gauss_LogLikelihood(settings, data.daily_deaths, parameter_labels)
    assert likelihood.n_parameters() == 7
    value = np.array([likelihood(parameter_values + [0.23])])
    assert np.allclose(value, [-3.11422e+08], rtol=1e-2, atol=1e-2)
    

def test_negbinom(settings, data, parameter_labels, parameter_values):
    likelihood = NegBinom_LogLikelihood(settings, data.daily_deaths, parameter_labels)
    assert likelihood.n_parameters() == 7
    value = np.array([likelihood(parameter_values + [0.23])])
    assert np.allclose(value, [-3097.57618], rtol=1e-2, atol=1e-2)


def test_poiss(settings, data, parameter_labels, parameter_values):
    likelihood = Poiss_LogLikelihood(settings, data.daily_deaths, parameter_labels)
    assert likelihood.n_parameters() == 6
    value = np.array([likelihood(parameter_values)])
    assert np.allclose(value, [-74598.73816], rtol=1e-2, atol=1e-2)
