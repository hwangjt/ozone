from openode.ode_function import ODEFunction
from openode.integrators.explicit_tm_integrator import ExplicitTMIntegrator
from openode.integrators.implicit_tm_integrator import ImplicitTMIntegrator
from openode.integrators.vectorized_integrator import VectorizedIntegrator
from openode.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
    ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod
from openode.utils.misc import get_scheme, get_integrator


def ODEIntegrator(ode_function, integrator_name, scheme_name, **kwargs):
    scheme_class = get_scheme(scheme_name)
    explicit = scheme_class().explicit
    integrator_class = get_integrator(integrator_name, explicit)

    if integrator_name == 'SAND' or integrator_name == 'MDF':
        kwargs['formulation'] = integrator_name

    integrator = integrator_class(ode_function=ode_function, scheme=scheme_class(), **kwargs)

    return integrator
