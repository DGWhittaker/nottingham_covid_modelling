# SItD model which uses Google mobility data

This folder contains a python implementation of a SItD (susceptible-infected-deceased) compartmental epidemiological model structured by infectious age.

It also contains a script to download ONS deaths statistics (see the `../data` folder for the latest download).

# Code overview
## Installation

We *strongly* recommend installing and running the scripts in a virtual environment to avoid version conflicts and make sure all paths work as expected. In order to do this, follow these steps:

- `virtualenv folder_name` or if you have both python 2 and 3: `virtualenv --python=python3 folder_name`. Should `virtualenv` not be recognised you may need to call it as `python -m virtualenv folder_name` or (`python -m virtualenv folder_name`). If that doesn't work you may need to install virtualenv first `pip install virtualenv`.
- go into the virtual environment `cd folder_name` folder
- and activate `source ./bin/activate` (or `Scripts\activate` on Windows)
- now get the source code from git: `git clone --recurse git@bitbucket.org:pmzspp/covid19-sp.git`
- install the required packages by typing `pip install -r requirements.txt` in the `python/` directory to install the required packages
- Now `pip install -e .` in the `python/` directory to install the *nottingham_covid_modelling* package itself

## Running
- Now you have a number of different scripts available, which can be run from any directory once the *nottingham_covid_modelling* package has been installed. Each of these can be run with the `-h`  flag to get more information about available command line arguments.

To run the basic forward model without Google mobility data, type:

`forward_model`.

- In order to find good parameters of the Google mobility data forward model, type:

`find_good_model -c "country_str"` where `"country_str"` is one of `{Australia,Canada,France,Germany,India,Italy,New Zealand,United Kingdom,United States,South Korea,Spain,Sweden}`.

- In order to optimise parameters of the Google mobility data forward model to UK death data using different noise models (default is negative binomial), type:

`optimise_model` or `run_all_cmaes` to generate all fits at once (UK ONS data only).

- Following this, once best fits from CMA-ES have been generated, run:

`mcmc` or `run_all_mcmc` to run all MCMC routines at once (takes a while, so using a multi-core server recommended). Note that this can ONLY be run after `optimise_model`, as best fits from CMA-ES are used as inputs. 

- To see diagnostic plots from the MCMC, type:

`plot_mcmc` or `plot_mcmc_series`. For all of the above, the `-o` argument can again be added to use UK ONS deaths data.

- To download the latest deaths numbers from the ONS website run:

`download_ons_data`

By default the file `ONS_daily_deaths.csv` will end up in `../data`. The data is processed from the [ONS website](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/deaths/datasets/weeklyprovisionalfiguresondeathsregisteredinenglandandwales).

**Please Note:** If your folder names have spaces this might cause issues depending on the platform you're using. e.g. `users/user_name/OneDrive - The University of Nottingham` if things don't run check you don't have spaces in your folder names.

## Folder structure
- The `nottingham_covid_modelling/lib` folder contains modules used by the forward model: `data.py` retrieves stored Google mobility data and daily deaths, `equations.py` solves the SIRD model difference equations, `error_measures.py` contains functions to calculate different error measures, `likelihood.py` and `priors.py` contain log-likelihood functions and priors, respectively, for different noise models, and `settings.py` contains default fixed parameter settings for the model.
- Google mobility data are now taken from `../data` folder and retrieved from [here](https://www.google.com/covid19/mobility/).
- GOV.UK daily death data are now taken from `../data` and retrieved from [here](https://coronavirus.data.gov.uk).
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

### Figure 4
In order to generate the data used in Figure 4, type the following:

- `optimise_model --optimise_likelihood --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR` for the step change in infectivity case
- `optimise_model --optimise_likelihood --ons_data --square --alpha1 --params_to_optimise rho Iinit1 IFR` for the constant infectivity (herd immunity) case.

These commands will run 5 repeats of the CMA-ES optimisation routine by default, and deposit the results in [nottingham_covid_modelling/cmaes_fits](https://bitbucket.org/MHendrix/covid19-mh/src/master/python/nottingham_covid_modelling/cmaes_fits/). To quickly visualise SItD model output using the found parameters, simply type:

- `plot_cma_fit --plot_fit --optimise_likelihood --ons_data --square --params_to_optimise rho Iinit1 lockdown_baseline lockdown_offset IFR` for the step change in infectivity case
- `plot_cma_fit --plot_fit --optimise_likelihood --ons_data --square --alpha1 -pto rho Iinit1 IFR` for the constant infectivity (herd immunity) case

or without the `--plot_fit` input argument to save a PNG figure (also in [nottingham_covid_modelling/cmaes_fits](https://bitbucket.org/MHendrix/covid19-mh/src/master/python/nottingham_covid_modelling/cmaes_fits/)). For convenience, we saved the model outputs in `.npy` files which are accessed in the plotting script for Figure 4.

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

Whereas Figure 2 is a schematic drawing, all other figures can be generated by typing `python plot_figure[figure number].py` inside the [nottingham_covid_modelling/figures](https://bitbucket.org/MHendrix/covid19-mh/src/master/python/nottingham_covid_modelling/figures/) directory. For example, running

- `python plot_figure6.py`

in the aforementioned directory will plot and save separately the constituent panels of Figure 6.

