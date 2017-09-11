from ozone.methods.runge_kutta.explicit_runge_kutta import \
    ForwardEuler, ExplicitMidpoint, ExplicitMidpointST, HeunsMethod, RalstonsMethod, \
    KuttaThirdOrder, KuttaThirdOrderST, RK4, RK4ST, RK6, RK6ST
from ozone.methods.runge_kutta.implicit_runge_kutta import \
    BackwardEuler, ImplicitMidpoint, TrapezoidalRule
from ozone.methods.runge_kutta.gauss_legendre import GaussLegendre
from ozone.methods.runge_kutta.lobatto import LobattoIIIA
from ozone.methods.runge_kutta.radau import Radau
from ozone.methods.linear_multistep.adams import AB, AM
from ozone.methods.linear_multistep.adams_alt import ABalt, AMalt
from ozone.methods.linear_multistep.bdf import BDF
from ozone.methods.linear_multistep.predictor_corrector import AdamsPEC, AdamsPECE

from ozone.utils.misc import _get_class


method_classes = {
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


family_names = [
    'ExplicitRungeKutta',
    'ImplicitRungeKutta',
    'GaussLegendre',
    'Lobatto',
    'Radau',
    'AB',
    'AM',
    'ABalt',
    'AMalt',
    'BDF',
    'AdamsPEC',
    'AdamsPECE',
]


method_families = {}
method_families['ExplicitRungeKutta'] = [
    'ForwardEuler',
    'ExplicitMidpoint',
    'KuttaThirdOrder',
    'RK4',
    'RK6',
]
method_families['ImplicitRungeKutta'] = [
    'BackwardEuler',
    'ImplicitMidpoint',
    'Trapezoidal',
]
method_families['GaussLegendre'] = [
    'GaussLegendre2',
    'GaussLegendre4',
    'GaussLegendre6',
]
method_families['Lobatto'] = [
    'Lobatto2',
    'Lobatto4',
]
method_families['Radau'] = [
    'RadauI3',
    'RadauI5',
    'RadauII3',
    'RadauII5',
]
method_families['AB'] = [
    'AB2',
    'AB3',
    'AB4',
    'AB5',
]
method_families['AM'] = [
    'AM2',
    'AM3',
    'AM4',
    'AM5',
]
method_families['ABalt'] = [
    'ABalt2',
    'ABalt3',
    'ABalt4',
    'ABalt5',
]
method_families['AMalt'] = [
    'AMalt3',
    'AMalt4',
    'AMalt5',
]
method_families['BDF'] = [
    'BDF1',
    'BDF2',
    'BDF3',
    'BDF4',
    'BDF5',
    'BDF6',
]
method_families['AdamsPEC'] = [
    'AdamsPEC2',
    'AdamsPEC3',
    'AdamsPEC4',
    'AdamsPEC5',
]
method_families['AdamsPECE'] = [
    'AdamsPECE2',
    'AdamsPECE3',
    'AdamsPECE4',
    'AdamsPECE5',
]


def get_method(method_name):
    return _get_class(method_name, method_classes, 'Method')
