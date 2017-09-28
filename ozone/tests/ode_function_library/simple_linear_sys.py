import numpy as np

from openmdao.api import ExplicitComponent


class SimpleLinearODESystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', default=1, type_=int)

    def setup(self):
        num = self.metadata['num_nodes']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=np.eye(num))
        self.declare_partials('dy_dt', 't')

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: e^t + sin(2*pi*t)
        two_pi_t = 2 * np.pi * inputs['t']
        outputs['dy_dt'][:, 0] = inputs['y'][:, 0] + 2 * np.pi * np.cos(two_pi_t) - np.sin(two_pi_t)

    def compute_partials(self, inputs, partials):
        two_pi_t = 2 * np.pi * inputs['t']
        partials['dy_dt', 't'] = self.eye \
            * (-(2 * np.pi) ** 2 * np.sin(two_pi_t) - 2 * np.pi * np.cos(two_pi_t))
