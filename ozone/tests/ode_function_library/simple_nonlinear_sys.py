import numpy as np

from openmdao.api import ExplicitComponent


class SimpleNonlinearODESystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', default=1, types=int)

    def setup(self):
        num = self.metadata['num_nodes']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        # self.declare_partials('dy_dt', 'y', val=np.eye(num))
        self.declare_partials('dy_dt', 't', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'y', rows=np.arange(num), cols=np.arange(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: 2 / (2*C1 - x^2)
        outputs['dy_dt'][:, 0] = inputs['t'] * np.square(inputs['y'][:, 0])

    def compute_partials(self, inputs, partials):
        partials['dy_dt', 'y'] = (2*inputs['t']*inputs['y'][:, 0]).squeeze()
        partials['dy_dt', 't'] = np.square(inputs['y'][:, 0]).squeeze()
