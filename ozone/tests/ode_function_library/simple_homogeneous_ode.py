import numpy as np

from openmdao.api import ExplicitComponent

from ozone.api import ODEFunction


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


class SimpleHomogeneousODEFunction(ODEFunction):

    def initialize(self, system_init_kwargs=None):
        self.set_system(SimpleHomogeneousODESystem, system_init_kwargs=system_init_kwargs)
        self.declare_state('y', rate_path='dy_dt', paths='y')
        self.declare_time('t')

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        state_names = ['y']
        return t0, t1, initial_conditions, state_names

    def get_exact_solution(self, initial_conditions, t0, t):
        a = 1.0 if 'a' not in self._system_init_kwargs else self._system_init_kwargs['a']
        y0 = initial_conditions['y']
        C = y0 / np.exp(a * t0)
        return {'y': C * np.exp(a * t)}
