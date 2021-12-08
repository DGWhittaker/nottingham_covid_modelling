import argparse
import os

from nottingham_covid_modelling import MODULE_DIR
# Load project modules
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood, Poiss_LogLikelihood


NOISE_MODEL_MAPPING = {'Norm': Gauss_LogLikelihood, 'NegBinom': NegBinom_LogLikelihood, 'Poiss': Poiss_LogLikelihood}

POPULATION = {'Australia': 25.0e6, 'Canada': 37.6e6, 'France': 67.0e6, 'Germany': 83.0e6,
              'India': 1.35e9, 'Italy': 60.4e6, 'New Zealand': 4.89e6, 'United Kingdom': 66.7e6,
              'United States': 328e6, 'South Korea': 51.6e6, 'Spain': 46.9e6, 'Sweden': 10.2e6}

IFR_dict = {'Canada': 0.00655, 'France': 0.008, 'Germany': 0.00897, 'Italy': 0.00945, \
       'United Kingdom': 0.00724, 'United States': 0.00606, 'South Korea': 0.00558, \
       'Spain': 0.00812, 'Sweden': 0.00764}


def get_parser(skip_data_folder=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("-nm", "--noise_model", type=str, help="which noise model to use",
                        choices=NOISE_MODEL_MAPPING.keys(), default='NegBinom')
    parser.add_argument("-o", "--ons_data", action='store_true', help="whether to use ONS England data or not",
                        default=False)
    parser.add_argument("-s", "--simple", action='store_true',
                        help="whether to use simple model or not (uses rate distributions)",
                        default=False)
    parser.add_argument("--square", action='store_true',
                        help="whether to use square lockdown",
                        default=False)
    parser.add_argument("--fix_phi", action='store_true',
                        help="whether to fix NB dispersion or not",
                        default=False)
    parser.add_argument("--fix_sigma", action='store_true',
                        help="whether to fix gaussian sigma or not",
                        default=False)
    parser.add_argument("--fixed_phi", type=float, help="value of phi", default=0.002)
    parser.add_argument("--flat_priors", action='store_true',
                        help="whether to use flat priors",
                        default=False)
    if not skip_data_folder:
        parser.add_argument("--datafolder", default=os.path.join(MODULE_DIR, '..', 'data', 'archive', 'current'), type=str,
                            help="full path of the folder where the data is located (default: ../data/archive/current)")
    return parser
