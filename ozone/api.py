from ozone.ode_function import ODEFunction
from ozone.integrators.explicit_tm_integrator import ExplicitTMIntegrator
from ozone.integrators.implicit_tm_integrator import ImplicitTMIntegrator
from ozone.integrators.vectorized_integrator import VectorizedIntegrator
from ozone.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
    ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod
from ozone.utils.misc import get_scheme, get_integrator


def ODEIntegrator(ode_function, integrator_name, scheme_name, **kwargs):
    scheme = get_scheme(scheme_name)
    explicit = scheme.explicit
    integrator_class = get_integrator(integrator_name, explicit)

    if integrator_name == 'SAND' or integrator_name == 'MDF':
        kwargs['formulation'] = integrator_name

    integrator = integrator_class(ode_function=ode_function, scheme=scheme, **kwargs)

    return integrator
