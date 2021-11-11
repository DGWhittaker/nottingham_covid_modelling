import os
import sys
from unittest import mock

import matplotlib
import numpy as np
import pytest
from nottingham_covid_modelling import MODULE_DIR
from nottingham_covid_modelling.forward_model import forward_model


def setup_function(function):
    """ setup printing options.
    """
    matplotlib.use('Agg')
    np.set_printoptions(threshold=np.inf)  # print everything
    np.set_printoptions(precision=5)  # reduce float precision for comparison across systems


@pytest.fixture(scope='session')
def reference_folder():
    """ Returns the path were the reference files for tests are stored in an OS independent way."""
    return os.path.join(MODULE_DIR, 'tests', 'reference_files')


def test_help(capsys, reference_folder):
    """ Check script usage"""
    testargs = ["forward_model", "-h"]
    with mock.patch.object(sys, 'argv', testargs):
        try:
            forward_model()
        except SystemExit:
            pass  # We expect this to print usage and exit
    captured = capsys.readouterr()
    output = str(captured.out)
    print(output)
    expected = open(os.path.join(reference_folder, 'forward_model_help.txt'), 'r').read()
    assert output == expected


# def test_defaults(reference_folder, tmp_path):
#     """ Check script usage"""
#     gen_graph_path = os.path.join(tmp_path, 'forward_model_defaults.png')
#     gen_numbers_path = os.path.join(tmp_path, 'forward_model_defaults_nubers.npy')
#     testargs = ["forward_model", "--outputgraph", gen_graph_path, "--outputnumbers", gen_numbers_path]
#     with mock.patch.object(sys, 'argv', testargs):
#         forward_model()

#     output = np.load(gen_numbers_path)
#     expected = np.load(os.path.join(reference_folder, 'forward_model_defaults_nubers.npy'))
#     assert np.allclose(output, expected, rtol=1e-2, atol=1e-2)

#     # we won't compare images as that's overly complicated but let's check it's there and is the same size
#     assert os.path.exists(gen_graph_path)
#     assert abs(os.path.getsize(gen_graph_path) -
#                os.path.getsize(os.path.join(reference_folder, 'forward_model_defaults.png'))) < 10000


# def test_different_parameters(reference_folder, tmp_path):
#     """ Check script usage"""
#     gen_numbers_path = os.path.join(tmp_path, 'forward_model_other_params_numbers.npy')
#     testargs = ["forward_model", "--outputnumbers", gen_numbers_path,
#                 "--maxtime", "90", "--startmonth", "Mar20", "-I0", "100", "-rho", "1.4"]
#     with mock.patch.object(sys, 'argv', testargs):
#         forward_model()

#     output = np.load(gen_numbers_path)
#     expected = np.load(os.path.join(reference_folder, 'forward_model_other_params_numbers.npy'))
#     assert np.allclose(output, expected, rtol=1e-2, atol=1e-2)
