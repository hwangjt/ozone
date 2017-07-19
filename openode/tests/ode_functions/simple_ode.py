import numpy as np

from openmdao.api import ExplicitComponent

from openode.api import ODEFunction


class Comp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=np.eye(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: e^t + sin(2*pi*t)
        two_pi_t = 2 * np.pi * inputs['t']
        outputs['dy_dt'] = inputs['y'] + 2 * np.pi * np.cos(two_pi_t) - np.sin(two_pi_t)

    def compute_partials(self, inputs, outputs, partials):
        two_pi_t = 2 * np.pi * inputs['t']
        partials['dy_dt', 't'] = self.eye \
            * (-(2 * np.pi) ** 2 * np.sin(two_pi_t) - 2 * np.pi * np.cos(two_pi_t))


class SimpleODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(Comp)
        self.declare_state('y', rate_target='dy_dt', state_targets='y')
        self.declare_time('t')


def simple_ode_func(y, t):
    two_pi_t = 2 * np.pi * t
    dy_dt = y + 2 * np.pi * np.cos(two_pi_t) - np.sin(two_pi_t)

    return dy_dt

def simple_ode_sol(t):
    return np.exp(t) + np.sin(2*np.pi*t)

def simple_ode_dfunc(y, t):
    return np.ones((1, 1))
