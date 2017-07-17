from openode.ode_function import ODEFunction
from openode.integrators.explicit_tm_integrator import ExplicitTMIntegrator
from openode.integrators.implicit_tm_integrator import ImplicitTMIntegrator
from openode.integrators.vectorized_integrator import VectorizedIntegrator
from openode.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
    ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod


def _get_class(name, classes, label):
    if name not in classes:
        msg = '%s name %s is invalid. Valid options are:\n' % (label, name)
        for tmp_name in classes:
            msg += '   %s\n' % tmp_name
        raise ValueError(msg)
    else:
        return classes[name]


def ode_integrator_group(ode_function, integrator_name, scheme_name, **kwargs):
    scheme_classes = {
        'forward Euler': ForwardEuler,
        'backward Euler': BackwardEuler,
        'explicit midpoint': ExplicitMidpoint,
        'implicit midpoint': ImplicitMidpoint,
        'Kutta third order': KuttaThirdOrder,
        'RK4': RK4,
        'Ralstons method': RalstonsMethod,
        'Heuns method': HeunsMethod,
    }
    scheme_class = _get_class(scheme_name, scheme_classes, 'Scheme')
    explicit = scheme_class().explicit

    integrator_classes = {
        'SAND': VectorizedIntegrator,
        'MDF': VectorizedIntegrator,
        'TM': ExplicitTMIntegrator if explicit else ImplicitTMIntegrator,
    }
    integrator_class = _get_class(integrator_name, integrator_classes, 'Integrator')

    if integrator_name == 'SAND' or integrator_name == 'MDF':
        kwargs['formulation'] = integrator_name

    integrator = integrator_class(ode_function=ode_function, scheme=scheme_class(), **kwargs)

    return integrator
