from openode.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
    ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod
from openode.schemes.bdf import BDF2


def _get_class(name, classes, label):
    if name not in classes:
        msg = '%s name %s is invalid. Valid options are:\n' % (label, name)
        for tmp_name in classes:
            msg += '   %s\n' % tmp_name
        raise ValueError(msg)
    else:
        return classes[name]


def get_scheme(scheme_name):
    scheme_classes = {
        'ForwardEuler': ForwardEuler,
        'BackwardEuler': BackwardEuler,
        'ExplicitMidpoint': ExplicitMidpoint,
        'ImplicitMidpoint': ImplicitMidpoint,
        'KuttaThirdOrder': KuttaThirdOrder,
        'RK4': RK4,
        'RalstonsMethod': RalstonsMethod,
        'HeunsMethod': HeunsMethod,
        'BDF2': BDF2,
    }
    return _get_class(scheme_name, scheme_classes, 'Scheme')


def get_integrator(integrator_name, explicit):
    from openode.integrators.explicit_tm_integrator import ExplicitTMIntegrator
    from openode.integrators.implicit_tm_integrator import ImplicitTMIntegrator
    from openode.integrators.vectorized_integrator import VectorizedIntegrator
    
    integrator_classes = {
        'SAND': VectorizedIntegrator,
        'MDF': VectorizedIntegrator,
        'TM': ExplicitTMIntegrator if explicit else ImplicitTMIntegrator,
    }
    return _get_class(integrator_name, integrator_classes, 'Integrator')
