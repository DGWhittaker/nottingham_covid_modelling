import matplotlib.pyplot as plt
import pints
import numpy as np
# Load project modules
from nottingham_covid_modelling.lib._command_line_args import IFR_dict, NOISE_MODEL_MAPPING, POPULATION, get_parser
from nottingham_covid_modelling.lib._save_to_file import save_np_to_file
from nottingham_covid_modelling.lib.data import DataLoader
from nottingham_covid_modelling.lib.equations import solve_difference_equations, tanh_spline, store_rate_vectors
from nottingham_covid_modelling.lib.error_measures import calculate_RMSE
from nottingham_covid_modelling.lib.settings import Params
import nottingham_covid_modelling.lib.priors as priors
from nottingham_covid_modelling.lib.likelihood import Gauss_LogLikelihood, NegBinom_LogLikelihood


def find_good_model():
    save_data = np.array([])
    parser = get_parser()

    parser.add_argument("-c", "--country_str", type=str, help="which country to use (default: United Kingdom)",
                        choices=POPULATION.keys(), default='United Kingdom')
    parser.add_argument("--nmodels", help="number of models to simulate (at least 10, default: 10000)",
                        default=10000, type=int)
    parser.add_argument("--outputnumbers", type=str, default=None, help='Output path for saving the numbers')
    parser.add_argument("--simulate_full", action='store_true',
                        help="whether to use all Google data (default is 150 days)", default=False)
    parser.add_argument("-pto", "--params_to_optimise", nargs='+', type=str, required=True, \
                        help="which parameters to optimise, e.g. -pto rho Iinit1 lockdown_baseline")

    args = parser.parse_args()

    if args.nmodels < 10:
        parser.error('Invalid number of models, --nmodels should be at least 10')

    if args.ons_data and args.country_str != 'United Kingdom':
        parser.error('Can only use ONS data in combination with country United Kingdom')

    # Get parameters, p
    p = Params()
    if args.country_str in IFR_dict:
        p.IFR = IFR_dict[args.country_str]

    # Simulate days after end of Google mobility data
    p.extra_days_to_simulate = 10
    if not args.simulate_full:
        p.n_days_to_simulate_after_150220 = 150

    # Get Google travel and deaths data
    print('Getting data...')
    data = DataLoader(args.ons_data, p, args.country_str, data_dir=args.datafolder)

    parameters_to_optimise = args.params_to_optimise
    if 'lockdown_new_normal' in parameters_to_optimise and not p.five_param_spline:
        raise ValueError("Cannot optimise lockdown_new_normal with 4 parameter spline")
    for gdp in p.gamma_dist_params:
        if gdp in parameters_to_optimise:
            p.calculate_rate_vectors = True
            break

    # Get noise model
    noise_str = args.noise_model
    noise_model = NOISE_MODEL_MAPPING[noise_str]

    # Time points (in days)
    t = np.linspace(0, p.maxtime, p.maxtime + 1)
    t_daily = np.linspace(p.day_1st_death_after_150220, p.maxtime - (p.numeric_max_age + p.extra_days_to_simulate), \
        (p.maxtime - p.day_1st_death_after_150220 - (p.numeric_max_age + p.extra_days_to_simulate) + 1))

    # Simulate lots of models
    ten_percent = int(args.nmodels / 10)
    j = 0
    nbest = 5

    np.random.seed(100)
    scores, params = [], []

    # Get likelihood function
    LL = noise_model(p, data.daily_deaths, parameters_to_optimise)

    upper_sigma = np.max(data.daily_deaths)
    log_prior = priors.LogPrior(LL, upper_sigma)
    log_posterior = pints.LogPosterior(LL, log_prior)

    print('Creating ' + str(args.nmodels) + ' random models...')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(True)
    for i in range(args.nmodels):
        x0 = priors.get_finite_LL_starting_point(log_prior, log_posterior)
        score = log_posterior(x0)
        params.append(x0)
        scores.append(-score)
        # Display progress
        if (i + 1) % ten_percent == 0:
            j += 10
            print(str(j) + '% completed...')

    # Sort according to best score
    order = np.argsort(scores)
    scores = np.asarray(scores)[order]
    params = np.asarray(params)[order]

    for i in range(nbest):
        input_params = params[i]
        if noise_model == Gauss_LogLikelihood:
            param_dict = dict(zip(parameters_to_optimise + ['gaussian_noise_sigma'], input_params))
        elif noise_model == NegBinom_LogLikelihood:
            param_dict = dict(zip(parameters_to_optimise + ['negative_binomial_phi'], input_params))
        else:
            param_dict = dict(zip(parameters_to_optimise, input_params))
        print('Best model no. ' + str(i + 1) + ' parameters: ')
        print(param_dict)
        print('Best model no. ' + str(i + 1) + ' log-likelihood: ' + str(-scores[i]))
        _, _, _, D, _ = solve_difference_equations(p, param_dict, travel_data=True)
        save_data = np.append(save_data, D)
        ax1.plot(t[:-p.numeric_max_age], D[:-p.numeric_max_age])
    ax1.scatter(t_daily, data.daily_deaths, edgecolor='black', facecolor='None',
             label='Observed data (' + data.country_display + ')')
    ax1.legend()
    ax1.set_ylabel('Daily deaths')
    ax1.set_title('Best ' + str(nbest) + ' models')
    ax1.set_xticks([x for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax1.set_xticklabels([data.google_data[x] for x in (0, 80, 160, 240) if x < len(data.google_data)])
    ax1.grid(True)
    plt.tight_layout()

    # Add final bits of data and save (if path was provided)
    if args.outputnumbers:
        save_data = np.append(save_data, scores)
        save_data = np.append(save_data, params)
        save_np_to_file(save_data, args.outputnumbers)
    
    plt.show()
