from ozone.api import ODEFunction
from ozone.tests.ode_function_library.getting_started_ode_sys import GettingStartedODESystem


class GettingStartedODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(GettingStartedODESystem)

        # Here, we declare that we have one state variable called 'y', which has shape 1.
        # We also specify the name/path for the 'f' for 'y', which is 'dy_dt'
        # and the name/path for the input to 'f' for 'y', which is 'y'.
        self.declare_state('y', 'dy_dt', shape=1, targets=['y'])
