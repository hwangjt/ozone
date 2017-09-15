import numpy as np

from ozone.api import ODEFunction
from ozone.tests.ode_function_library.simple_nonlinear_sys import SimpleNonlinearODESystem


class SimpleNonlinearODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(SimpleNonlinearODESystem)
        self.declare_state('y', 'dy_dt', targets='y')
        self.declare_time(targets='t')

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        return initial_conditions, t0, t1

    def get_exact_solution(self, initial_conditions, t0, t):
        # True solution: 2 / (2*C - t^2)
        # outputs['dy_dt'] = inputs['t'] * np.square(inputs['y'])

        y0 = initial_conditions['y']
        C = (2. / y0 + t0 ** 2) / 2.
        return {'y': 2. / (2. * C - t ** 2)}
