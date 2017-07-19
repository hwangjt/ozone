import numpy as np

from openmdao.api import ExplicitComponent

from openode.api import ODEFunction

class NonlinearODE(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))

        # self.declare_partials('dy_dt', 'y', val=np.eye(num))
        self.declare_partials('dy_dt', 't', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'y', rows=np.arange(num), cols=np.arange(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: 2 / (2*C1 - x^2)
        outputs['dy_dt'] = inputs['t'] * np.square(inputs['y'])

    def compute_partials(self, inputs, outputs, partials):
        partials['dy_dt', 'y'] = (2*inputs['t']*inputs['y']).squeeze()
        partials['dy_dt', 't'] = np.square(inputs['y']).squeeze()


class LinearODE(ExplicitComponent):

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


class LinearODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(LinearODE)
        self.declare_state('y', rate_target='dy_dt', state_targets='y')
        self.declare_time('t')

class NonlinearODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(NonlinearODE)
        self.declare_state('y', rate_target='dy_dt', state_targets='y')
        self.declare_time('t')
