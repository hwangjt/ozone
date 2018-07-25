import numpy as np

from openmdao.api import ExplicitComponent


class SimpleHomogeneousODESystem(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('a', default=1., types=(int, float))

    def setup(self):
        num = self.options['num_nodes']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=self.options['a'] * np.eye(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        outputs['dy_dt'] = self.options['a'] * inputs['y']
