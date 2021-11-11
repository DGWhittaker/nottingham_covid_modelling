import os

import nottingham_covid_modelling.lib.priors as priors
import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood, Poiss_LogLikelihood
from nottingham_covid_modelling.lib.settings import Params


def setup_function(function):
    """ setup printing options.
    """
    np.random.seed(100)
    priors.DEBUG = True


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
    return ['rho', 'Iinit1', 'lockdown_baseline', 'lockdown_fatigue', 'lockdown_offset', 'lockdown_rate']


def test_priors_Gauss_LogLikelihood(capsys, settings, data, parameter_labels):
    # Get likelihood function
    LL = Gauss_LogLikelihood(settings, data.daily_deaths, parameter_labels)

    upper_sigma = np.max(data.daily_deaths)
    log_prior = priors.LogPrior(LL, upper_sigma)
    assert log_prior.n_parameters() == 7
    assert np.allclose(log_prior.lower, np.array([[1, 1, 0, 0, 0, 0.01, 0]]), rtol=1e-2, atol=1e-2)
    assert np.allclose(log_prior.upper, np.array([[10, 5e7, 1, 0.1, 100, 1, upper_sigma]]), rtol=1e-2, atol=1e-2)
    sample = log_prior.sample()
    assert np.allclose(sample, np.array([5.89064448e+00, 1.39184700e+07, 4.24517591e-01, 8.44776132e-02,
                                        3.39439624e+01, 1.45339524e-01, 7.78101275e+02]), rtol=1e-2, atol=1e-2)
    assert log_prior(sample) == -2.499046065324075
    sample[0] = 99999
    assert log_prior(sample) == -float('inf')
    sample[0] = -99999
    assert log_prior(sample) == -float('inf')
    captured = capsys.readouterr()
    assert 'Upper' in captured.out
    assert 'Lower' in captured.out


def test_priors_NegBinom_LogLikelihood(capsys, settings, data, parameter_labels):
    # Get likelihood function
    LL = NegBinom_LogLikelihood(settings, data.daily_deaths, parameter_labels)

    upper_sigma = np.max(data.daily_deaths)
    log_prior = priors.LogPrior(LL, upper_sigma)
    assert log_prior.n_parameters() == 7
    assert np.allclose(log_prior.lower, np.array([[1, 1, 0, 0, 0, 0.01, 0]]), rtol=1e-2, atol=1e-2)
    assert np.allclose(log_prior.upper, np.array([[10, 5e7, 1, 0.1, 100, 1, 1]]), rtol=1e-2, atol=1e-2)
    sample = log_prior.sample()
    assert np.allclose(sample, np.array([5.89064448e+00, 1.39184700e+07, 4.24517591e-01, 8.44776132e-02,
                                        3.39439624e+01, 1.45339524e-01, 5.75093329e-01]), rtol=1e-2, atol=1e-2)
    assert log_prior(sample) == -2.499046065324075
    sample[0] = 99999
    assert log_prior(sample) == -float('inf')
    sample[0] = -99999
    assert log_prior(sample) == -float('inf')
    captured = capsys.readouterr()
    assert 'Upper' in captured.out
    assert 'Lower' in captured.out


def test_priors_Poiss_LogLikelihood(capsys, settings, data, parameter_labels):
    # Get likelihood function
    LL = Poiss_LogLikelihood(settings, data.daily_deaths, parameter_labels)

    upper_sigma = np.max(data.daily_deaths)
    log_prior = priors.LogPrior(LL, upper_sigma)
    assert log_prior.n_parameters() == 6
    assert np.allclose(log_prior.lower, np.array([[1, 1, 0, 0, 0, 0.01]]), rtol=1e-2, atol=1e-2)
    assert np.allclose(log_prior.upper, np.array([[10, 5e7, 1, 0.1, 100, 1]]), rtol=1e-2, atol=1e-2)
    sample = log_prior.sample()
    assert np.allclose(sample, np.array([5.89064448e+00, 1.39184700e+07, 4.24517591e-01, 8.44776132e-02,
                                        3.39439624e+01, 1.45339524e-01]), rtol=1e-2, atol=1e-2)
    assert log_prior(sample) == -2.499046065324075
    sample[0] = 99999
    assert log_prior(sample) == -float('inf')
    sample[0] = -99999
    assert log_prior(sample) == -float('inf')
    captured = capsys.readouterr()
    assert 'Upper' in captured.out
    assert 'Lower' in captured.out


def test_wrong_parameter_label(capsys, settings, data, parameter_labels):
    # Get likelihood function
    LL = Poiss_LogLikelihood(settings, data.daily_deaths, parameter_labels + ['wrog_name'])

    upper_sigma = np.max(data.daily_deaths)
    with pytest.raises(AssertionError, match=r"Unknown parameter wrog_name"):
        priors.LogPrior(LL, upper_sigma)


def test_wrong_samples(capsys, settings, data, parameter_labels):
    # Get likelihood function
    LL = Poiss_LogLikelihood(settings, data.daily_deaths, parameter_labels)

    upper_sigma = np.max(data.daily_deaths)
    log_prior = priors.LogPrior(LL, upper_sigma)
    with pytest.raises(NotImplementedError):
        log_prior.sample(5)
