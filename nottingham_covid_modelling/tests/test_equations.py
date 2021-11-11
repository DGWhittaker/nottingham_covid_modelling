import os

import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import get_model_solution, solve_difference_equations, tanh_spline, store_rate_vectors
from nottingham_covid_modelling.lib.settings import Params

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


# def test_solve_difference_equations(tmp_path, reference_folder, settings, param_dictionary):
#     store_rate_vectors(param_dictionary, settings)
#     S, Iday, R, D, Itot = solve_difference_equations(settings, param_dictionary, True)
#     result = np.append(np.append(np.append(np.append(S, Iday), R), D), Itot)
#     file_name = 'solve_difference_equations_1.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(result, gen_numbers_path)
    
#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


# def test_solve_difference_equations2(tmp_path, reference_folder, settings, param_dictionary):
#     # Same results but using parameters stored in settings
#     store_rate_vectors(param_dictionary, settings)
#     S, Iday, R, D, Itot = solve_difference_equations(settings, {}, True)
#     result = np.append(np.append(np.append(np.append(S, Iday), R), D), Itot)
#     file_name = 'solve_difference_equations_1.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(result, gen_numbers_path)
    
#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


# def test_solve_difference_equations3(tmp_path, reference_folder, settings, param_dictionary):
#     store_rate_vectors(param_dictionary, settings)
#     S, Iday, R, D, Itot = solve_difference_equations(settings, {}, False)
#     result = np.append(np.append(np.append(np.append(S, Iday), R), D), Itot)
#     file_name = 'solve_difference_equations_3.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(result, gen_numbers_path)
    
#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_solve_difference_equations4(tmp_path, reference_folder, settings, param_dictionary):
    settings.simple = True
    store_rate_vectors(param_dictionary, settings)
    S, Iday, R, D, Itot = solve_difference_equations(settings, {}, True)
    settings.simple = False
    result = np.append(np.append(np.append(np.append(S, Iday), R), D), Itot)
    file_name = 'solve_difference_equations_4.npy'
    gen_numbers_path = os.path.join(tmp_path, file_name)
    save_np_to_file(result, gen_numbers_path)
    
    generated = np.load(gen_numbers_path, allow_pickle=True)
    expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
    assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_solve_difference_equations5(tmp_path, reference_folder, settings, param_dictionary):
    settings.simple = True
    store_rate_vectors(param_dictionary, settings)
    S, Iday, R, D, Itot = solve_difference_equations(settings, {}, False)
    settings.simple = False
    result = np.append(np.append(np.append(np.append(S, Iday), R), D), Itot)
    file_name = 'solve_difference_equations_5.npy'
    gen_numbers_path = os.path.join(tmp_path, file_name)
    save_np_to_file(result, gen_numbers_path)
    
    generated = np.load(gen_numbers_path, allow_pickle=True)
    expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
    assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_solve_difference_equations_wrong_params1(tmp_path, reference_folder, settings, param_dictionary):
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        store_rate_vectors(param_dictionary, settings)
        solve_difference_equations(settings, None, False)


def test_store_rate_vectors_wrong_params1(tmp_path, reference_folder, settings, param_dictionary):
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'max_infected_age'"):
        store_rate_vectors(param_dictionary, None)
        #solve_difference_equations(None, {}, False)


# def test_get_model_solution(tmp_path, reference_folder, settings, param_dictionary):
#     store_rate_vectors(param_dictionary, settings)
#     result = get_model_solution(settings, param_dictionary)

#     file_name = 'get_model_solution_1.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(result, gen_numbers_path)
    
#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


# def test_get_model_solution2(tmp_path, reference_folder, settings, param_dictionary):
#     # Same results but using parameters stored in settings
#     store_rate_vectors(param_dictionary, settings)
#     result = get_model_solution(settings, {})

#     file_name = 'get_model_solution_1.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     save_np_to_file(result, gen_numbers_path)
    
#     generated = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
#     assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_get_model_solution_wrong_attr1(reference_folder, settings, param_dictionary):
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        get_model_solution(settings, None)


def test_get_model_solution_wrong_attr2(reference_folder, settings, param_dictionary):
    with pytest.raises(AssertionError, match="If using travel data, it must be loaded into p using DataLoader"):
        get_model_solution(None, {})


def test_tanh_spline(tmp_path, reference_folder, settings, data, param_dictionary):
    result = tanh_spline(settings, data.lgoog_data, param_dictionary)

    file_name = 'tanh_spline_1.npy'
    gen_numbers_path = os.path.join(tmp_path, file_name)
    save_np_to_file(result, gen_numbers_path)
    
    generated = np.load(gen_numbers_path, allow_pickle=True)
    expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
    assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_tanh_spline2(tmp_path, reference_folder, settings, data):
    # Same results but using parameters stored in settings
    result = tanh_spline(settings, data.lgoog_data, {})
    
    file_name = 'tanh_spline_1.npy'
    gen_numbers_path = os.path.join(tmp_path, file_name)
    save_np_to_file(result, gen_numbers_path)
    
    generated = np.load(gen_numbers_path, allow_pickle=True)
    expected = np.load(os.path.join(reference_folder, file_name), allow_pickle=True)
    assert np.allclose(generated, expected, rtol=1e-2, atol=1e-2)


def test_tanh_spline_wrong_attr1(reference_folder, settings):
    with pytest.raises(ValueError, match="Number of samples, -44, must be non-negative."):
        tanh_spline(settings, -44, {})


def test_tanh_spline_wrong_attr2(reference_folder, settings):
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        tanh_spline(settings, 10, None)


def test_tanh_spline_wrong_attr3(reference_folder, settings):
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'numeric_max_age'"):
        tanh_spline(None, 10, {})
