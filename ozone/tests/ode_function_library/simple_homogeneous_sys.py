import numpy as np

from openmdao.api import ExplicitComponent


class SimpleHomogeneousODESystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)
        self.metadata.declare('a', default=1., type_=(int, float))

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=self.metadata['a'] * np.eye(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        outputs['dy_dt'] = self.metadata['a'] * inputs['y']
