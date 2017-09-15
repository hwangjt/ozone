import numpy as np

from ozone.api import ODEFunction
from ozone.tests.ode_function_library.simple_linear_sys import SimpleLinearODESystem


class SimpleLinearODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(SimpleLinearODESystem)
        self.declare_state('y', 'dy_dt', targets='y')
        self.declare_time(targets='t')

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        return initial_conditions, t0, t1

    def get_exact_solution(self, initial_conditions, t0, t):
        # True solution: C e^t + sin(2*pi*t)
        # outputs['dy_dt'] = inputs['y'] + 2 * np.pi * np.cos(two_pi_t) - np.sin(two_pi_t)

        y0 = initial_conditions['y']
        C = (y0 - np.sin(2 * np.pi * t0)) / np.exp(t0)
        return {'y': C * np.exp(t) + np.sin(2 * np.pi * t)}
