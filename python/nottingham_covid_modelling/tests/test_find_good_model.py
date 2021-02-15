import os
import sys
import warnings
from unittest import mock

import matplotlib
import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.find_good_model import find_good_model


def setup_function(function):
    """ setupprinting options.
    """
    matplotlib.use('Agg')
    np.set_printoptions(threshold=np.inf)  # print everything
    np.random.seed(100)
    warnings.filterwarnings("ignore", message=("Matplotlib is currently using agg, which is a non-GUI backend, "
                                               "so cannot show the figure."))


@pytest.fixture(scope='session')
def reference_folder():
    """ Returns the path were the reference files for tests are stored in an OS independent way."""
    return os.path.join(MODULE_DIR, 'tests', 'reference_files')


def test_help(capsys, reference_folder):
    """ Check script usage"""
    testargs = ["find_good_model", "-h"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            find_good_model()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.out)
    print(output)
    expected = open(os.path.join(reference_folder, 'find_good_model_help.txt'), 'r').read()
    assert output == expected


def test_too_few_models(capsys, reference_folder):
    """ Check error for too few models"""
    testargs = ["find_good_model", "--nmodels", "9"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            find_good_model()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.err)
    print(output)
    expected = open(os.path.join(reference_folder, 'find_good_model_too_few_models.txt'), 'r').read()
    assert output == expected


def test_ons_wrong_country(capsys, reference_folder):
    """ Check error for too few models"""
    testargs = ["find_good_model", "--ons_data", "-c", "Sweden"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            find_good_model()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.err)
    print(output)
    expected = open(os.path.join(reference_folder, 'find_good_model_ons_wrong_country.txt'), 'r').read()
    assert output == expected


# def test_defaults_ons(reference_folder, tmp_path):
#     """ Check script with default settings"""
#     file_name = 'find_good_model_numbers_defaults.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     expected_numbers_path = os.path.join(reference_folder, file_name)
#     testargs = ["find_good_model", "--nmodels", "10", "--datafolder", reference_folder,
#                 "--outputnumbers", gen_numbers_path, "-nm", "Norm", "--ons_data", "--simulate_full"]
#     with mock.patch.object(sys, 'argv', testargs):
#         find_good_model()

#     expected = np.load(expected_numbers_path, allow_pickle=True)
#     output = np.load(gen_numbers_path, allow_pickle=True)
#     assert np.allclose(output, expected, rtol=1e-1, atol=1e-2)


# def test_poiss(reference_folder, tmp_path):
#     """ Check script with Poiss Canada"""
#     file_name = 'find_good_model_numbers_Poiss_canada.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     expected_numbers_path = os.path.join(reference_folder, file_name)
#     testargs = ["find_good_model", "--nmodels", "10", "--datafolder", reference_folder,
#                 "--outputnumbers", gen_numbers_path, "-nm", "Poiss", "-c", "Canada", 
#                 "--simulate_full"]
#     with mock.patch.object(sys, 'argv', testargs):
#         find_good_model()

#     output = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(expected_numbers_path, allow_pickle=True)
#     assert np.allclose(output, expected, rtol=1e-2, atol=1e-2)


# def test_negbinom(reference_folder, tmp_path):
#     """ Check script with NegBinom Australia"""
#     file_name = 'find_good_model_numbers_NegBinom_australia.npy'
#     gen_numbers_path = os.path.join(tmp_path, file_name)
#     expected_numbers_path = os.path.join(reference_folder, file_name)
#     testargs = ["find_good_model", "--nmodels", "10", "--datafolder", reference_folder,
#                 "--outputnumbers", gen_numbers_path, "-nm", "NegBinom", "-c", "Australia", 
#                 "--simulate_full"]
#     with mock.patch.object(sys, 'argv', testargs):
#         find_good_model()

#     output = np.load(gen_numbers_path, allow_pickle=True)
#     expected = np.load(expected_numbers_path, allow_pickle=True)
#     assert np.allclose(output, expected, rtol=1e-2, atol=1e-2)


# def test_defaults_ons_current_data():
#     """ Check script works with current data and default settings"""
#     testargs = ["find_good_model", "--nmodels", "10", "-nm", "Norm", "--ons_data"]
#     with mock.patch.object(sys, 'argv', testargs):
#         find_good_model()


# def test_poiss_current_data():
#     """ Check script works with current data and Poiss Canada"""
#     testargs = ["find_good_model", "--nmodels", "10", "-nm", "Poiss", "-c", "Canada"]
#     with mock.patch.object(sys, 'argv', testargs):
#         find_good_model()


# def test_negbinom_current_data():
#     """ Check script works with current data and NegBinom Australia"""
#     testargs = ["find_good_model", "--nmodels", "10", "-nm", "NegBinom", "-c", "Australia"]
#     with mock.patch.object(sys, 'argv', testargs):
#         find_good_model()
