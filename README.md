# SItD model which uses Google mobility data

This folder contains a python implementation of a SItD (susceptible-infected-deceased) compartmental epidemiological model structured by infectious age.
The code herein is associated with the paper:

***"Uncertainty and error in SARS-CoV-2 epidemiological parameters inferred from population-level epidemic models"*** (Currently under review). Whittaker, D. G.\*, Herrera-Reyes, A. D.\*, Hendrix, M., Owen, M. R., Band, L. R., Mirams, G. R., Bolton, K. J., Preston, S. P.

\* joint first authors

# Code overview
## Installation

We *strongly* recommend installing and running the scripts in a virtual environment to avoid version conflicts and make sure all paths work as expected. In order to do this, follow these steps:

- `virtualenv folder_name` or if you have both python 2 and 3: `virtualenv --python=python3 folder_name`. Should `virtualenv` not be recognised you may need to call it as `python -m virtualenv folder_name`. If that doesn't work you may need to install virtualenv first `pip install virtualenv`.
- go into the virtual environment `cd folder_name` folder
- and activate `source ./bin/activate` (or `Scripts\activate` on Windows)
- now get the source code from git: `git clone https://github.com/DGWhittaker/nottingham_covid_modelling.git`
- install the required packages by typing `pip install -r requirements.txt`
- Now `pip install -e .` to install the *nottingham_covid_modelling* package itself

## Running
- Now you have a number of different scripts available, which can be run from any directory once the *nottingham_covid_modelling* package has been installed. Each of these can be run with the `-h`  flag to get more information about available command line arguments.

To run the basic forward model without Google mobility data, type:

`forward_model`.

- In order to find good parameters of the Google mobility data forward model, type:

`find_good_model -c "country_str"` where `"country_str"` is one of `{Australia,Canada,France,Germany,India,Italy,New Zealand,United Kingdom,United States,South Korea,Spain,Sweden}`.

- In order to optimise parameters of the Google mobility data forward model to UK death data using different noise models (default is negative binomial), type:

`optimise_model`.

- Following this, once best fits from CMA-ES have been generated, run:

`mcmc` (takes a while, so using a multi-core server recommended). Note that this can ONLY be run after `optimise_model`, as best fits from CMA-ES are used as inputs. 

- To see diagnostic plots from the MCMC, type:

`plot_mcmc` or `plot_mcmc_series`. For all of the above, the `-o` argument can again be added to use UK ONS deaths data.

The data used are processed from the [ONS website](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/weeklyprovisionalfiguresondeathsregisteredinenglandandwales).

**Please Note:** If your folder names have spaces this might cause issues depending on the platform you're using. e.g. `users/user_name/OneDrive - The University of Nottingham` if things don't run check you don't have spaces in your folder names.

## Folder structure
- The `nottingham_covid_modelling/lib` folder contains modules used by the forward model: `data.py` retrieves stored Google mobility data and daily deaths, `equations.py` solves the SItD model difference equations, `error_measures.py` contains functions to calculate different error measures, `likelihood.py` and `priors.py` contain log-likelihood functions and priors, respectively, for different noise models, and `settings.py` contains default fixed parameter settings for the model.
- Google mobility data are retrieved from [here](https://www.google.com/covid19/mobility/).
- GOV.UK daily death data are retrieved from [here](https://coronavirus.data.gov.uk).
- Automatic tests for our code are in `nottingham_covid_modelling/tests`


## Contributing new scripts

The way it works is:

- You make a script as per usual
- Except you put all code (apart from the imports) in a function
- In `setup.py` in the python folder you see `entry_points={ 'console_scripts': [ ...` here the entry point for all console scripts are defined. e.g. The follow means the rommand `run_me` will wil run `my_function` from `my_script_file.py` in the `nottingham_covid_modelling` folder (package) `'run_me=nottingham_covid_modelling.my_script_file:my_function'`
- You may need to uninstall the module `pip uninstall nottingham_covid_modelling` and re-install it `pip install -e .` in the `python/` directory for this change to take effect

## Running tests
To run the tests:

- Make sure you have the virtual environment activated (see above).
- Navigate to the source folder inside the virtual environment (i.e. covid19-sp)
- Run `pytest`

To run a specific test:
`pytest` <full path to specific test> e.g. `(covid) C:\Users\uczmh2\Desktop\covid\covid19-sp>pytest python\nottingham_covid_modelling\tests\test_csv_utils.py`

**Please Note:** you can actually be anywhere inside the source folder, but avoid being in the main virtual environment folder (in the above example that would be `C:\Users\uczmh2\Desktop\covid`) as this would make pytest try to run tests of the various other installed python modules. 
This is both a waste of time and it will probably fail (some modules require additional packages to be installed for the tests to work).

# Reproducing paper results
## Running simulations

### Figure 3
In order to generate and save synthetic data for Figure 3, type the following:

- `SIR_SINR_AGE_model_default -travel -step --outputnumbers FILENAME_WITH_FULL_PATH` 

This will simulate the SItD model and some configuration of the simple models. The observation model will be simulated with a negative binomial distribution and saved as part of the SItD simulation. The exact data for Figure 3 can be found in `SItRDmodel_ONSparams_noise_NB_NO-R_travel_TRUE_step_TRUE.npy` in [nottingham_covid_modelling/out_SIRvsAGEfits](python/nottingham_covid_modelling/out_SIRvsAGEfits/).

To fit the current synthetic data to the three simple models, and the SItD model, type the following:

- `SIR_SINR_fit_AGEdata -r 10 -age -travel -step -fitstep` 

This code authomatically saves the results in [nottingham_covid_modelling/out_SIRvsAGEfits](python/nottingham_covid_modelling/out_SIRvsAGEfits/).
It will run 10 repeats of the CMA-ES optimization routine. You can change the number of repetitions by "-r N" where N is your desired repeats.
It saves a PNG figure equivalent to Figure 3 in the paper (also in [nottingham_covid_modelling/out_SIRvsAGEfits](python/nottingham_covid_modelling/out_SIRvsAGEfits/)).


### Figure 4
In order to generate the data used in Figure 4, type the following:

- `optimise_model --optimise_likelihood --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR` for the step change in infectivity case
- `optimise_model --optimise_likelihood --ons_data --square --alpha1 --params_to_optimise rho Iinit1 IFR` for the constant infectivity (herd immunity) case.

These commands will run 5 repeats of the CMA-ES optimisation routine by default, and deposit the results in [nottingham_covid_modelling/cmaes_fits](python/nottingham_covid_modelling/cmaes_fits/). To quickly visualise SItD model output using the found parameters, simply type:

- `plot_cma_fit --plot_fit --optimise_likelihood --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR` for the step change in infectivity case
- `plot_cma_fit --plot_fit --optimise_likelihood --ons_data --square --alpha1 -pto rho Iinit1 IFR` for the constant infectivity (herd immunity) case

or without the `--plot_fit` input argument to save a PNG figure (also in [nottingham_covid_modelling/cmaes_fits](python/nottingham_covid_modelling/cmaes_fits/)). For convenience, we saved the model outputs in `.npy` files which are accessed in the plotting script for Figure 4.

### Figure 5
The results for Figure 5 rely on first obtaining an estimate of the maximum a posteriori probability (MAP) using CMA-ES optimisation, then using this as a starting point for Bayesian inference using MCMC. To generate the necessary CMA-ES files, type:

- `optimise_model --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR`

followed by

- `mcmc --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR`

to run the Markov chain Monte Carlo (MCMC) sampling. Note that each of these commands could take several hours to run, so it is advised to use an HPC resource, from which the code can take advantage of parallel capabilities. Our favoured method is to use a window manager such as `screen` or `tmux` to keep a session active even if disconnected, and to run the commands in detached mode, e.g. running

- `mcmc --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR &> mcmc_figure5.out &`

will print the console output to `mcmc_figure5.out` rather than to screen. Typing `jobs` in the terminal will show the status, i.e. `Running` or `Done` (or `Exited` if there is some error).

### Figure 6
Similarly to Figure 5, the results for Figure 6 rely on first obtaining the MAP estimate using CMA-ES. To generate the necessary CMA-ES files, type:

- `optimise_model --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset beta_mean beta_var death_mean death_var IFR --repeats 10`

followed by

- `mcmc --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset beta_mean beta_var death_mean death_var IFR --niter 200000`

to run the Markov chain Monte Carlo (MCMC) sampling. Note here that we opted to run more repeats of the CMA-ES optimisation and more iterations of the MCMC due to the higher dimensionality of the problem.

## Generating figures

Whereas Figure 2 is a schematic drawing, all other figures can be generated by typing `python plot_figure[figure number].py` inside the [nottingham_covid_modelling/figures](python/nottingham_covid_modelling/figures/) directory. For example, running

- `python plot_figure6.py`

in the aforementioned directory will plot and save separately the constituent panels of Figure 6.


# Acknowledging this work

If you publish any work based on the contents of this repository please cite:

[PLACEHOLDER]

