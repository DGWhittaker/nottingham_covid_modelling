import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import nottingham_covid_modelling
import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
from nottingham_covid_modelling.lib.ratefunctions import discretise_gamma_distribution, negative_binomial_distribution, make_rate_vectors
from nottingham_covid_modelling.lib.settings import Params
from nottingham_covid_modelling.lib.data import DataLoader


def setup_function(function):
    """ setup printing options.
    """
    matplotlib.use('Agg')
    warnings.filterwarnings("ignore", message=("Matplotlib is currently using agg, which is a non-GUI backend, "
                                               "so cannot show the figure."))


@pytest.fixture(scope='session')
def reference_folder():
    """ Returns the path were the reference files for tests are stored in an OS independent way."""
    return os.path.join(MODULE_DIR, 'tests', 'reference_files')

@pytest.fixture(scope='session')
def settings_and_data(reference_folder):
    # We need to set up settings and load data together as data loading affects settings.alpha
    settings = Params()
    data = DataLoader(True, settings, 'United Kingdom', data_dir=reference_folder)
    return settings, data

@pytest.fixture(scope='session')
def settings(settings_and_data):
    settings, _ = settings_and_data
    return settings

@pytest.fixture(scope='session')
def data(settings_and_data):
    _, data = settings_and_data
    return data

@pytest.fixture(scope='session')
def param_dictionary(settings):
    return {'lockdown_offset': settings.lockdown_offset, 'lockdown_baseline': settings.lockdown_baseline,
            'lockdown_rate': settings.lockdown_rate, 'lockdown_fatigue': settings.lockdown_fatigue}

def test_discretise_gamma_distribution(reference_folder, tmp_path):
    """ Check discretise_gamma_distribution works as expected """
    p = Params()
    lamdaf, lamdaF, lamdaG = discretise_gamma_distribution(p.beta_mean, p.beta_var, p.timestep, p.max_infected_age)
    file_name = 'rate_functions_discretise_gamma_distribution_1.npy'
    
    gen_numbers_path = os.path.join(tmp_path, file_name)
    save_np_to_file(np.append(np.append(lamdaf, lamdaF), lamdaG), gen_numbers_path)
    
    generated = np.load(gen_numbers_path, allow_pickle=True)
    expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
    assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_discretise_gamma_distribution2(reference_folder, tmp_path):
    """ Check discretise_gamma_distribution works when we change parameters """
    p = Params()
    p.lamda_mean = 25
    p.lamda_var = 17
    p.max_infected_age = 1500
        
    lamdaf, lamdaF, lamdaG = discretise_gamma_distribution(p.lamda_mean, p.lamda_var, p.timestep, p.max_infected_age)
    file_name = 'rate_functions_discretise_gamma_distribution_2.npy'
    
    gen_numbers_path = os.path.join(tmp_path, file_name)
    save_np_to_file(np.append(np.append(lamdaf, lamdaF), lamdaG), gen_numbers_path)
    
    generated = np.load(gen_numbers_path, allow_pickle=True)
    expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
    assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_discretise_gamma_distribution3():
    """ Check changing settings works """
    p = Params()
    p.timestep = -6

    with pytest.raises(NotImplementedError, match="Current implementation expects a timestep of 1!"):
        lamdaf, lamdaF, lamdaG = discretise_gamma_distribution(p.beta_mean, p.beta_var, p.timestep, p.max_infected_age)


# def test_make_rate_vectors(reference_folder, tmp_path, param_dictionary):
#     """ Check make_rate_vectors works as expected """
#     lamda, gamma, zeta = make_rate_vectors(param_dictionary)
#     file_name = 'rate_functions_make_rate_vectors_1.npy'
    
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(np.append(np.append(lamda, gamma), zeta), gen_numbers_path)

#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


# def test_make_rate_vectors2(reference_folder, tmp_path, param_dictionary):
#     """ Check changing settings works """
#     p = Params()
#     p.beta_mean = 25
#     p.beta_var = 17
#     p.death_N_NB = 30
#     p.death_p_NB = .6
#     p.recovery_N_NB = 30
#     p.recovery_p_NB = .6
#     p.max_infected_age = 500

#     lamda, gamma, zeta = make_rate_vectors(param_dictionary, p)

#     file_name = 'rate_functions_make_rate_vectors_2.npy'
    
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(np.append(np.append(lamda, gamma), zeta), gen_numbers_path)

#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_make_rate_vectors4(param_dictionary):
    """ Check changing settings works """
    p = Params()
    p.timestep = 11 

    with pytest.raises(NotImplementedError, match="Current implementation expects a timestep of 1!"):
        make_rate_vectors(param_dictionary, p)


def test_make_rate_vectors5(param_dictionary):
    nottingham_covid_modelling.lib.ratefunctions.DEBUG = True
    p = Params()
    make_rate_vectors(param_dictionary, p)
    plt.show()


def test_wrong_timestep_gamma_distribution():
    p = Params()
    p.timestep = 11
    with pytest.raises(NotImplementedError, match="Current implementation expects a timestep of 1!"):
        lamdaf, lamdaF, lamdaG = discretise_gamma_distribution(p.beta_mean, p.beta_var, p.timestep, p.max_infected_age)


def test_wrong_timestep_ratevector(param_dictionary):
    p = Params()
    p.timestep = 11
    with pytest.raises(NotImplementedError, match="Current implementation expects a timestep of 1!"):
        make_rate_vectors(param_dictionary, p)
