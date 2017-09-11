import numpy as np

from ozone.api import ODEFunction
from ozone.tests.ode_function_library.simple_homogeneous_sys import SimpleHomogeneousODESystem


class SimpleHomogeneousODEFunction(ODEFunction):

    def initialize(self, system_init_kwargs=None):
        self.set_system(SimpleHomogeneousODESystem, system_init_kwargs=system_init_kwargs)
        self.declare_state('y', rate_path='dy_dt', paths='y')
        self.declare_time('t')

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        return initial_conditions, t0, t1

    def get_exact_solution(self, initial_conditions, t0, t):
        a = 1.0 if 'a' not in self._system_init_kwargs else self._system_init_kwargs['a']
        y0 = initial_conditions['y']
        C = y0 / np.exp(a * t0)
        return {'y': C * np.exp(a * t)}
