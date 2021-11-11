from nottingham_covid_modelling.lib.settings import Params


def test_settings():
    """ Tests settings contains exactly the expected parameters and the defaults are as expected.
    
    This test is to guard against accidental modification as settings affects most other things in the code.
    """
    p = Params()
    assert p.maxtime == 240
    assert p.c == 21
    assert p.day_1st_death_after_150220 == 0
    assert p.extra_days_to_simulate == 0
    assert p.numeric_max_age == 133
    assert p.transit == True
    assert p.N == 70000000
    assert p.L == 4
    assert p.M == 6
    assert p.IFR == 0.01
    assert p.beta_mean == 5.2
    assert p.beta_var == 2.96
    assert p.death_mean == 18.69
    assert p.death_dispersion == 0.0546
    assert p.recovery_mean == 150
    assert p.recovery_dispersion == 1e-12
    assert p.timestep == 1
    assert p.rho == 2.4
    assert p.Iinit1 == 1000
    assert p.lockdown_offset == 15
    assert p.lockdown_rate == 0.15
    assert p.lockdown_fatigue == 1e-3
    assert p.lockdown_baseline == 0.5
    assert p.lockdown_new_normal == 1.0
    assert p.max_infected_age == 133
    assert p.simple == False
    assert p.square_lockdown == False

    attributes = sorted(filter(lambda a: not a.startswith('__'), dir(p)))
    print(str(attributes))
    assert str(attributes) == ("['IFR', 'Iinit1', 'L', 'M', 'N', 'alpha1', 'beta_mean', 'beta_var', 'c', "
        "'calculate_rate_vectors', 'day_1st_death_after_150220', 'death_dispersion', 'death_mean', "
        "'extra_days_to_simulate', 'five_param_spline', 'fix_phi', 'fix_sigma', 'fixed_phi', 'fixed_sigma', "
        "'flat_priors', 'gamma_dist_params', 'lockdown_baseline', 'lockdown_fatigue', 'lockdown_new_normal', "
        "'lockdown_offset', 'lockdown_rate', 'max_infected_age', 'maxtime', 'n_days_to_simulate_after_150220', "
        "'numeric_max_age', 'recovery_dispersion', 'recovery_mean', 'rho', 'simple', 'square_lockdown', 'timestep', 'transit']")

