import os
import sys
import warnings
from unittest import mock

import matplotlib
import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.optimise_model import run_optimise

def setup_function(function):
    """ setupprinting options.
    """
    matplotlib.use('Agg')
    np.random.seed(100)
    warnings.filterwarnings("ignore", message=("Matplotlib is currently using agg, which is a non-GUI backend, "
                                               "so cannot show the figure."))

@pytest.fixture(scope='session')
def reference_folder():
    """ Returns the path were the reference files for tests are stored in an OS independent way."""
    return os.path.join(MODULE_DIR, 'tests', 'reference_files')


def test_help(capsys, reference_folder):
    testargs = ["run_optimise", "-h"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            run_optimise()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.out)
    print(output)
    expected = open(os.path.join(reference_folder, 'optimise_model_help.txt'), 'r').read()
    assert output == expected


def test_ons_wrong_country(capsys, reference_folder):
    testargs = ["run_optimise", "--ons_data", "-c", "Sweden"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            run_optimise()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.err)
    print(output)
    expected = open(os.path.join(reference_folder, 'optimise_model_ons_wrong_country.txt'), 'r').read()
    assert output == expected


# def test_defaults_ons(reference_folder, tmp_path):
#     # default noise model: NegBinom_LogLikelihood
#     file_prefix = 'Data_England+Wales_noise-model_NegBinom_rho_Init1_lockdown-baseline_death-mean_death-dispersion_recovery-mean_recovery-dispersion'

#     gen_fit_path = os.path.join(tmp_path, file_prefix + '.txt')
#     gen_err_path = os.path.join(tmp_path, file_prefix + '-errors.txt')

#     expected_fit_path = os.path.join(reference_folder, file_prefix + '.txt')
#     expected_err_path = os.path.join(reference_folder, file_prefix + '-errors.txt')

#     testargs = ["run_optimise", "--limit_pints_iterations", "2", "--repeats", "1", "--datafolder", reference_folder,
#                 "--cmaes_fits", str(tmp_path), "--ons_data", "--sample_from_priors", "--simulate_full"]

#     with mock.patch.object(sys, 'argv', testargs):
#         run_optimise()

#     gen_fit = np.loadtxt(gen_fit_path)
#     gen_err = np.loadtxt(gen_err_path)
#     expected_fit = np.loadtxt(expected_fit_path)
#     expected_err = np.loadtxt(expected_err_path)
#     assert np.allclose(gen_fit, expected_fit, rtol=1e-2, atol=1e-2)
#     assert np.allclose(gen_err, expected_err, rtol=1e-1, atol=1e-2)


# def test_Gauss(reference_folder, tmp_path):
#     file_prefix = 'Data_United-Kingdom_noise-model_Norm_rho_Init1_lockdown-baseline_death-mean_death-dispersion_recovery-mean_recovery-dispersion'

#     gen_fit_path = os.path.join(tmp_path, file_prefix + '.txt')
#     gen_err_path = os.path.join(tmp_path, file_prefix + '-errors.txt')

#     expected_fit_path = os.path.join(reference_folder, file_prefix + '.txt')
#     expected_err_path = os.path.join(reference_folder, file_prefix + '-errors.txt')
    
#     testargs = ["run_optimise", "--limit_pints_iterations", "2", "--repeats", "1", "--datafolder", reference_folder,
#                 "--cmaes_fits", str(tmp_path), "-nm", "Norm", "--sample_from_priors", "--simulate_full"]

#     with mock.patch.object(sys, 'argv', testargs):
#         run_optimise()

#     gen_fit = np.loadtxt(gen_fit_path)
#     gen_err = np.loadtxt(gen_err_path)
#     expected_fit = np.loadtxt(expected_fit_path)
#     expected_err = np.loadtxt(expected_err_path)
#     assert np.allclose(gen_fit, expected_fit, rtol=1e-2, atol=1e-2)
#     assert np.allclose(gen_err, expected_err, rtol=1e-1, atol=1e-2)


# def test_Poiss(reference_folder, tmp_path):
#     file_prefix = 'Data_Sweden_noise-model_Poiss_rho_Init1_lockdown-baseline_death-mean_death-dispersion_recovery-mean_recovery-dispersion'

#     gen_fit_path = os.path.join(tmp_path, file_prefix + '.txt')
#     gen_err_path = os.path.join(tmp_path, file_prefix + '-errors.txt')
#     gen_param1_path = os.path.join(tmp_path, file_prefix + '-parameters-1.txt')
#     gen_log_path = os.path.join(tmp_path, file_prefix + '-log-0.txt')

#     expected_fit_path = os.path.join(reference_folder, file_prefix + '.txt')
#     expected_err_path = os.path.join(reference_folder, file_prefix + '-errors.txt')
#     expected_param1_path = os.path.join(reference_folder, file_prefix + '-parameters-1.txt')
#     expected_log_path = os.path.join(reference_folder, file_prefix + '-log-0.txt')

#     testargs = ["run_optimise", "--limit_pints_iterations", "2", "--repeats", "1", "--datafolder", reference_folder,
#                 "--cmaes_fits", str(tmp_path), "-nm", "Poiss", "--detailed_output", "-c", "Sweden",
#                 "--sample_from_priors", "--simulate_full"]
#     with mock.patch.object(sys, 'argv', testargs):
#         run_optimise()

#     gen_fit = np.loadtxt(gen_fit_path)
#     gen_err = np.loadtxt(gen_err_path)
#     gen_param1 = np.loadtxt(gen_param1_path)
#     expected_fit = np.loadtxt(expected_fit_path)
#     expected_err = np.loadtxt(expected_err_path)
#     expected_param1 = np.loadtxt(expected_param1_path)
#     assert np.allclose(gen_fit, expected_fit, rtol=1e-2, atol=1e-2)
#     assert np.allclose(gen_err, expected_err, rtol=1e-2, atol=1e-2)
#     assert np.allclose(gen_param1, expected_param1, rtol=1e-2, atol=1e-2)
    
#     # go through the log files line by line comapring, butting off the time (last item) as that will differ
#     with open(gen_log_path, "r") as f:
#         gen_log = f.readlines()
#     with open(expected_log_path, "r") as f:
#         expected_log = f.readlines()
#     for gen, exp in zip(gen_log, expected_log):
#         gen = gen.split()
#         exp = exp.split()
#         print(gen, exp)
#         assert gen[:-1] == gen[:-1]
