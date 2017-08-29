import numpy as np
from six import iteritems

from ozone.utils.misc import _get_class
from ozone.methods_list import get_method


def ODEIntegrator(ode_function, integrator_name, method_name,
        initial_conditions=None, static_parameters=None, dynamic_parameters=None,
        initial_time=None, final_time=None, normalized_times=None, times=None,
        **kwargs):
    method = get_method(method_name)
    explicit = method.explicit
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
        num_times = len(normalized_times)

        for parameter_name, value in iteritems(dynamic_parameters):
            assert parameter_name in ode_function._dynamic_parameters, \
                'Dynamic parameter (%s) was not declared in ODEFunction' % parameter_name

            assert isinstance(value, np.ndarray), \
                'Dynamic parameter %s must be an ndarray' % parameter_name

            shape = ode_function._dynamic_parameters[parameter_name]['shape']
            assert value.shape == (num_times,) + shape, \
                'Dynamic parameter %s has the wrong shape' % state_name

    # ------------------------------------------------------------------------------------

    if integrator_name == 'optimizer-based' or integrator_name == 'solver-based':
        kwargs['formulation'] = integrator_name

    integrator = integrator_class(ode_function=ode_function, method=method,
        initial_conditions=initial_conditions,
        static_parameters=static_parameters, dynamic_parameters=dynamic_parameters,
        initial_time=initial_time, final_time=final_time, normalized_times=normalized_times,
        all_norm_times=normalized_times,
        **kwargs)

    return integrator


def get_integrator(integrator_name, explicit):
    from ozone.integrators.explicit_tm_integrator import ExplicitTMIntegrator
    from ozone.integrators.implicit_tm_integrator import ImplicitTMIntegrator
    from ozone.integrators.vectorized_integrator import VectorizedIntegrator

    integrator_classes = {
        'optimizer-based': VectorizedIntegrator,
        'solver-based': VectorizedIntegrator,
        'time-marching': ExplicitTMIntegrator if explicit else ImplicitTMIntegrator,
    }
    return _get_class(integrator_name, integrator_classes, 'Integrator')
