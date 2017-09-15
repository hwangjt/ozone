import numpy as np
import time

from ozone.api import ODEFunction
from ozone.tests.ode_function_library.projectile_dynamics_sys import ProjectileSystem


class ProjectileFunction(ODEFunction):

    def initialize(self, system_init_kwargs=None):
        self.set_system(ProjectileSystem, system_init_kwargs)

        self.declare_state('x', 'dx_dt', shape=1)
        self.declare_state('y', 'dy_dt', shape=1)
        self.declare_state('vx', 'dvx_dt', shape=1, targets=['vx'])
        self.declare_state('vy', 'dvy_dt', shape=1, targets=['vy'])

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {
            'x': 0.,
            'y': 0.,
            'vx': 1.,
            'vy': 1.,
        }
        return initial_conditions, t0, t1

    def get_exact_solution(self, initial_conditions, t0, t):
        g = -9.81

        x0 = initial_conditions['x']
        y0 = initial_conditions['y']
        vx0 = initial_conditions['vx']
        vy0 = initial_conditions['vy']

        x = x0 + vx0 * (t - t0)
        y = y0 + vy0 * (t - t0) + 0.5 * g * (t - t0) ** 2
        vx = vx0
        vy = vy0 + g * (t - t0)
        return {'x': x, 'y': y, 'vx': vx, 'vy': vy}
