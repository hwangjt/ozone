from openode.schemes.runge_kutta import ForwardEuler, BackwardEuler, ExplicitMidpoint, \
    ImplicitMidpoint, KuttaThirdOrder, RK4, RalstonsMethod, HeunsMethod, RK4ST, GaussLegendre, \
    LobattoIIIA, RadauI
from openode.schemes.bdf import BDF
from openode.schemes.adams import AB, ABalt, AM, AMalt, AdamsPEC, AdamsPECE


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
        # First-order methods
        'ForwardEuler': ForwardEuler(),
        'BackwardEuler': BackwardEuler(),
        # Runge--Kutta methods
        'ExplicitMidpoint': ExplicitMidpoint(),
        'ImplicitMidpoint': ImplicitMidpoint(),
        'KuttaThirdOrder': KuttaThirdOrder(),
        'RK4': RK4(),
        'RalstonsMethod': RalstonsMethod(),
        'HeunsMethod': HeunsMethod(),
        'GaussLegendre2': GaussLegendre(2),
        'GaussLegendre4': GaussLegendre(4),
        'GaussLegendre6': GaussLegendre(6),
        'Lobatto2': LobattoIIIA(2),
        'Lobatto4': LobattoIIIA(4),
        'Radau3': RadauI(3),
        'Radau5': RadauI(5),
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
