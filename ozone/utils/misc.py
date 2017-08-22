import numpy as np
from six import iteritems


def ODEIntegrator(ode_function, integrator_name, scheme_name,
        initial_conditions=None, static_parameters=None, dynamic_parameters=None,
        initial_time=None, final_time=None, normalized_times=None, times=None,
        **kwargs):
    scheme = get_scheme(scheme_name)
    explicit = scheme.explicit
    integrator_class = get_integrator(integrator_name, explicit)

    # ------------------------------------------------------------------------------------
    # time-related option
    assert normalized_times is not None or times is not None, \
        'Either normalized_times or times must be provided'

    if normalized_times is not None:
        assert isinstance(normalized_times, np.ndarray) and len(normalized_times) == 1, \
            'normalized_times must be a 1-D array'

    if times is not None:
        assert isinstance(times, np.ndarray) and len(times.shape) == 1, \
            'times must be a 1-D array'

        assert initial_time is None and final_time is None and normalized_times is None, \
            'If times is provided: initial_time, final_time, and normalized_times cannot be'

        initial_time = times[0]
        final_time = times[-1]
        normalized_times = (times - times[0]) / (times[-1] - times[0])

    # ------------------------------------------------------------------------------------
    # Ensure that all initial_conditions are valid
    if initial_conditions is not None:
        for state_name, value in iteritems(initial_conditions):
            assert state_name in ode_function._states, \
                'Initial condition (%s) was not declared in ODEFunction' % state_name

            assert isinstance(value, np.ndarray) or np.isscalar(value), \
                'The initial condition for state %s must be an ndarray or a scalar' % state_name

            assert np.atleast_1d(value).shape == ode_function._states[state_name]['shape'], \
                'The initial condition for state %s has the wrong shape' % state_name

            initial_conditions[state_name] = np.atleast_1d(value)

    # ------------------------------------------------------------------------------------
    # Ensure that all static parameters are valid
    if static_parameters is not None:
        for parameter_name, value in iteritems(static_parameters):
            assert parameter_name in ode_function._static_parameters, \
                'Static parameter (%s) was not declared in ODEFunction' % parameter_name

            assert isinstance(value, np.ndarray) or np.isscalar(value), \
                'Static parameter (%s) must be an ndarray or a scalar' % parameter_name

            shape = ode_function._static_parameters[parameter_name]['shape']
            assert np.atleast_1d(value).shape == shape, \
                'Static parameter (%s) has the wrong shape' % parameter_name

            static_parameters[parameter_name] = np.atleast_1d(value)

    # ------------------------------------------------------------------------------------
    # Ensure that all dynamic parameters are valid
    if dynamic_parameters is not None:
        num_time_steps = len(normalized_times)

        for parameter_name, value in iteritems(dynamic_parameters):
            assert parameter_name in ode_function._dynamic_parameters, \
                'Dynamic parameter (%s) was not declared in ODEFunction' % parameter_name

            assert isinstance(value, np.ndarray), \
                'Dynamic parameter %s must be an ndarray' % parameter_name

            shape = ode_function._dynamic_parameters[parameter_name]['shape']
            assert value.shape == (num_time_steps,) + shape, \
                'Dynamic parameter %s has the wrong shape' % state_name

    # ------------------------------------------------------------------------------------

    if integrator_name == 'SAND' or integrator_name == 'MDF':
        kwargs['formulation'] = integrator_name

    integrator = integrator_class(ode_function=ode_function, scheme=scheme,
        initial_conditions=initial_conditions,
        static_parameters=static_parameters, dynamic_parameters=dynamic_parameters,
        initial_time=initial_time, final_time=final_time, normalized_times=normalized_times,
        all_norm_times=normalized_times,
        **kwargs)

    return integrator


def _get_class(name, classes, label):
    if name not in classes:
        msg = '%s name %s is invalid. Valid options are:\n' % (label, name)
        for tmp_name in classes:
            msg += '   %s\n' % tmp_name
        raise ValueError(msg)
    else:
        return classes[name]


def get_scheme(scheme_name):
    from ozone.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
        ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod, RK4ST, GaussLegendre, \
        LobattoIIIA, Radau, TrapezoidalRule, RK6, RK6ST, KuttaThirdOrderST, ExplicitMidpointST
    from ozone.schemes.multistep import AdamsPEC, AdamsPECE, AB, AM, ABalt, AMalt, BDF

    scheme_classes = {
        # First-order methods
        'ForwardEuler': ForwardEuler(),
        'BackwardEuler': BackwardEuler(),
        # Runge--Kutta methods
        'ExplicitMidpoint': ExplicitMidpoint(),
        'ImplicitMidpoint': ImplicitMidpoint(),
        'KuttaThirdOrder': KuttaThirdOrder(),
        'RK4': RK4(),
        'RK6': RK6(),
        'RalstonsMethod': RalstonsMethod(),
        'HeunsMethod': HeunsMethod(),
        'GaussLegendre2': GaussLegendre(2),
        'GaussLegendre4': GaussLegendre(4),
        'GaussLegendre6': GaussLegendre(6),
        'Lobatto2': LobattoIIIA(2),
        'Lobatto4': LobattoIIIA(4),
        'RadauI3': Radau('I', 3),
        'RadauI5': Radau('I', 5),
        'RadauII3': Radau('II', 3),
        'RadauII5': Radau('II', 5),
        'Trapezoidal': TrapezoidalRule(),
        # Adams--Bashforth family
        'AB1': ForwardEuler(),
        'AB2': AB(2),
        'AB3': AB(3),
        'AB4': AB(4),
        'AB5': AB(5),
        'ABalt2': ABalt(2),
        'ABalt3': ABalt(3),
        'ABalt4': ABalt(4),
        'ABalt5': ABalt(5),
        # Adams--Moulton family
        'AM1': BackwardEuler(),
        'AM2': AM(2),
        'AM3': AM(3),
        'AM4': AM(4),
        'AM5': AM(5),
        'AMalt2': AMalt(2),
        'AMalt3': AMalt(3),
        'AMalt4': AMalt(4),
        'AMalt5': AMalt(5),
        # Predictor-corrector methods,
        'AdamsPEC2': AdamsPEC(2),
        'AdamsPEC3': AdamsPEC(3),
        'AdamsPEC4': AdamsPEC(4),
        'AdamsPEC5': AdamsPEC(5),
        'AdamsPECE2': AdamsPECE(2),
        'AdamsPECE3': AdamsPECE(3),
        'AdamsPECE4': AdamsPECE(4),
        'AdamsPECE5': AdamsPECE(5),
        # Backwards differentiation formula family
        'BDF1': BackwardEuler(),
        'BDF2': BDF(2),
        'BDF3': BDF(3),
        'BDF4': BDF(4),
        'BDF5': BDF(5),
        'BDF6': BDF(6),
        # Starting methods with derivatives
        'ExplicitMidpointST': ExplicitMidpointST(),
        'KuttaThirdOrderST': KuttaThirdOrderST(),
        'RK4ST': RK4ST(),
        'RK6ST': RK6ST(),
    }
    return _get_class(scheme_name, scheme_classes, 'Scheme')


def get_integrator(integrator_name, explicit):
    from ozone.integrators.explicit_tm_integrator import ExplicitTMIntegrator
    from ozone.integrators.implicit_tm_integrator import ImplicitTMIntegrator
    from ozone.integrators.vectorized_integrator import VectorizedIntegrator

    integrator_classes = {
        'SAND': VectorizedIntegrator,
        'MDF': VectorizedIntegrator,
        'TM': ExplicitTMIntegrator if explicit else ImplicitTMIntegrator,
    }
    return _get_class(integrator_name, integrator_classes, 'Integrator')


def get_scheme_families():
    scheme_families = {}
    scheme_families['basic'] = [
        ('ForwardEuler', 1),
        ('BackwardEuler', 1),
        ('ExplicitMidpoint', 2),
        ('ImplicitMidpoint', 2),
    ]
    scheme_families['GaussLegendre'] = [
        ('GaussLegendre2', 2),
        ('GaussLegendre4', 4),
        ('GaussLegendre6', 6),
    ]
    scheme_families['Lobatto'] = [
        ('Lobatto2', 2),
        ('Lobatto4', 4),
    ]
    scheme_families['Radau'] = [
        ('RadauI3', 3),
        ('RadauI5', 5),
        ('RadauII3', 3),
        ('RadauII5', 5),
    ]
    scheme_families['AB'] = [
        ('AB1', 1),
        ('AB2', 2),
        ('AB3', 3),
        ('AB4', 4),
        ('AB5', 5),
    ]
    scheme_families['AM'] = [
        ('AM1', 1),
        ('AM2', 2),
        ('AM3', 3),
        ('AM4', 4),
        ('AM5', 5),
    ]
    scheme_families['BDF'] = [
        ('BDF1', 1),
        ('BDF2', 2),
        ('BDF3', 3),
        ('BDF4', 4),
        ('BDF5', 5),
        ('BDF6', 6),
    ]
    scheme_families['AdamsPEC'] = [
        ('AdamsPEC2', 2),
        ('AdamsPEC3', 3),
        ('AdamsPEC4', 4),
        ('AdamsPEC5', 5),
    ]
    scheme_families['AdamsPECE'] = [
        ('AdamsPECE2', 2),
        ('AdamsPECE3', 3),
        ('AdamsPECE4', 4),
        ('AdamsPECE5', 5),
    ]
    return scheme_families
