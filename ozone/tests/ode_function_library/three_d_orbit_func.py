import numpy as np

from ozone.api import ODEFunction
from ozone.tests.ode_function_library.three_d_orbit_sys import ThreeDOrbitSystem


class ThreeDOrbitFunction(ODEFunction):

    def initialize(self, system_init_kwargs=None):
        self.set_system(ThreeDOrbitSystem, system_init_kwargs=system_init_kwargs)

        self.declare_state('r', 'r_dot', paths='r', shape=3)
        self.declare_state('v', 'v_dot', paths='v', shape=3)
        self.declare_state('m', 'm_dot', paths='m', shape=1)

        self.declare_dynamic_parameter('d', 'd', shape=1)
        self.declare_dynamic_parameter('a', 'a', shape=1)
        self.declare_dynamic_parameter('b', 'b', shape=1)

    def get_default_parameters(self):
        r_scal = 1e12 if 'a' not in self._system_init_kwargs else self._system_init_kwargs['r_scal']
        v_scal = 1e3  if 'a' not in self._system_init_kwargs else self._system_init_kwargs['v_scal']

        t0 = 0.
        t1 = 348.795 * 24 * 36001

        initial_conditions = {
            'r': np.array([ -140699693 , -51614428 , 980 ]) * 1e3 / r_scal,
            'v': np.array([ 9.774596 , -28.07828 , 4.337725e-4 ]) * 1e3 / v_scal,
            'm': 1000.
        }
        return initial_conditions, t0, t1
