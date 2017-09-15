from ozone.api import ODEFunction
from ozone.tests.ode_function_library.getting_started_oc_sys import GettingStartedOCSystem


class GettingStartedOCFunction(ODEFunction):

    def initialize(self, system_init_kwargs=None):
        self.set_system(GettingStartedOCSystem, system_init_kwargs)

        # We have 3 states: x, y, v
        self.declare_state('x', 'dx_dt', shape=1, targets=['x'])
        self.declare_state('y', 'dy_dt', shape=1, targets=['y'])
        self.declare_state('v', 'dv_dt', shape=1, targets=['v'])

        # We declare theta as a dynamic parameter as we will declare it as a control later.
        self.declare_dynamic_parameter('theta', 'theta', shape=1)
