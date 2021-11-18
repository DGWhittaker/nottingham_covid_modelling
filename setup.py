#
# setuptools script
#
import os

from setuptools import find_packages, setup


# Load text for description
with open('README.md') as f:
    readme = f.read()

# Load version number
with open(os.path.join('nottingham_covid_modelling', 'version.txt'), 'r') as f:
    version = f.read()

setup(
    # Module name (lowercase)
    name='nottingham_covid_modelling',

    version=version,
    description='Scripts for Covid-19 modelling at The University of Nottingham School of Mathematical Sciences',
    long_description=readme,
    license='BSD 3-clause license',
    author='Dominic Whittaker, Maurice Hendrix, Gary Mirams',
    author_email='Dominic.Whittaker@nottingham.ac.uk, Maurice.Hendrix@nottingham.ac.uk, Gary.Mirams@nottingham.ac.uk',
    url='https://bitbucket.org/pmzspp/covid19-sp/src/master/python/',
    zip_safe=False,

    # Packages to include
    packages=find_packages(include=('nottingham_covid_modelling', 'nottingham_covid_modelling.*')),

    # Include non-python files (via MANIFEST.in)
    include_package_data=True,

    # Required Python version
    python_requires='>=3.5',

    # List of dependencies
    install_requires=[
        'imageio>=2.8',
        'requests>=2',
        'numpy>=1.18',
        'matplotlib>=3.2',
        'pytest>=5.4',
        'openpyxl>=3.0',
        'scipy>=1.4.1',
        'flake8>=3.8',
        'isort>=4.3',
        'requests-mock>=1.8',
        'pytest-cov>=2.9',
    ],
    entry_points={
        'console_scripts': [
            'download_ons_data='
            'nottingham_covid_modelling.download_ons_data:download_and_process_ons',
            'find_good_model='
            'nottingham_covid_modelling.find_good_model:find_good_model',
            'forward_model='
            'nottingham_covid_modelling.forward_model:forward_model',
            'plot_mcmc='
            'nottingham_covid_modelling.plot_MCMC:plot_mcmc',
            'plot_mcmc_series='
            'nottingham_covid_modelling.plot_MCMC_series:plot_mcmc_series',
            'plot_mcmc_series_map='
            'nottingham_covid_modelling.plot_MCMC_series_MAP:plot_mcmc_series_map',
            'plot_mcmc_series_nb_ll='
            'nottingham_covid_modelling.plot_MCMC_series_NB_LL:plot_mcmc_series_nb_ll',
            'plot_mcmc_nb_distributions='
            'nottingham_covid_modelling.plot_MCMC_NB_distributions:plot_mcmc_nb_distributions',
            'plot_mcmc_likelihoods='
            'nottingham_covid_modelling.plot_MCMC_likelihoods:plot_mcmc_likelihoods',
            'optimise_model='
            'nottingham_covid_modelling.optimise_model:run_optimise',
            'mcmc='
            'nottingham_covid_modelling.MCMC:run_mcmc',
            'run_UK_cmaes='
            'nottingham_covid_modelling.run_UK_cmaes:run_UK_cmaes',
            'run_countries_cmaes='
            'nottingham_covid_modelling.run_countries_cmaes:run_countries_cmaes',
            'run_all_mcmc='
            'nottingham_covid_modelling.run_all_MCMC:run_all_mcmc',
            'plot_cma_fit='
            'nottingham_covid_modelling.plot_CMA_fit:plot_cma',
            'plot_google_data='
            'nottingham_covid_modelling.plot_google_data:plot_google_data',
            'plot_counterfactuals='
            'nottingham_covid_modelling.plot_counterfactuals:plot_counterfactuals',
            'SIR_forward_model='
            'nottingham_covid_modelling.SIR_forward_model:SIR_forward_model',
            'hazards_exploration='
            'nottingham_covid_modelling.hazards_exploration:hazards_exploration',
            'forward_model_forSIRtravel='
            'nottingham_covid_modelling.forward_model_forSIRtravel:forward_model_forSIRtravel',
            'SIR_SINR_fit_AGEdata='
            'nottingham_covid_modelling.SIR_SINR_fit_AGEdata:run_optimise',
            'SIR_SINR_fit_AGEdataNOnoise='
            'nottingham_covid_modelling.SIR_SINR_fit_AGEdataNOnoise:run_optimise',
            'SIR_SINR_fit_ONSdata='
            'nottingham_covid_modelling.SIR_SINR_fit_DATA:run_optimise',
            'SIR_SINR_AGE_model_default='
            'nottingham_covid_modelling.SIR_SINR_AGE_models:SIR_SINR_AGE_model_default',
            'plot_SIR_fits_fig2='
                'nottingham_covid_modelling.plot_fig1_SIR_SINR_fit_AGEdata:plot_SIR_fits_fig2',
            'SIR_fit_debug='
            'nottingham_covid_modelling.optimise_likelihood_anyModel:run_optimise',
        ],
    },
)
