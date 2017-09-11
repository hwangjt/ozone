import numpy as np

from openmdao.api import ExplicitComponent

from ozone.api import ODEFunction


class SimpleLinearODESystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=np.eye(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: e^t + sin(2*pi*t)
        two_pi_t = 2 * np.pi * inputs['t']
        outputs['dy_dt'][:, 0] = inputs['y'][:, 0] + 2 * np.pi * np.cos(two_pi_t) - np.sin(two_pi_t)

    def compute_partials(self, inputs, partials):
        two_pi_t = 2 * np.pi * inputs['t']
        partials['dy_dt', 't'] = self.eye \
            * (-(2 * np.pi) ** 2 * np.sin(two_pi_t) - 2 * np.pi * np.cos(two_pi_t))


class SimpleLinearODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(SimpleLinearODESystem)
        self.declare_state('y', rate_path='dy_dt', paths='y')
        self.declare_time('t')

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        state_names = ['y']
        return t0, t1, initial_conditions, state_names

    def get_exact_solution(self, initial_conditions, t0, t):
        # True solution: C e^t + sin(2*pi*t)
        # outputs['dy_dt'] = inputs['y'] + 2 * np.pi * np.cos(two_pi_t) - np.sin(two_pi_t)

        y0 = initial_conditions['y']
        C = (y0 - np.sin(2 * np.pi * t0)) / np.exp(t0)
        return {'y': C * np.exp(t) + np.sin(2 * np.pi * t)}
