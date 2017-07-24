from openode.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
    ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod, RK4ST
from openode.schemes.bdf import BDF
from openode.schemes.ab import AB
from openode.schemes.am import AM


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
        'ForwardEuler': ForwardEuler(),
        'BackwardEuler': BackwardEuler(),
        'ExplicitMidpoint': ExplicitMidpoint(),
        'ImplicitMidpoint': ImplicitMidpoint(),
        'KuttaThirdOrder': KuttaThirdOrder(),
        'RK4': RK4(),
        'RalstonsMethod': RalstonsMethod(),
        'HeunsMethod': HeunsMethod(),
        'AB1': ForwardEuler(),
        'AB2': AB(2),
        'AB3': AB(3),
        'AB4': AB(4),
        'AB5': AB(5),
        'AM1': BackwardEuler(),
        'AM2': AM(2),
        'AM3': AM(3),
        'AM4': AM(4),
        'BDF1': BackwardEuler(),
        'BDF2': BDF(2),
        'BDF3': BDF(3),
        'BDF4': BDF(4),
        'BDF5': BDF(5),
        'BDF6': BDF(6),
        'RK4ST': RK4ST(),
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
