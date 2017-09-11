from ozone.utils.test_utils import OzoneODETestCase
from ozone.tests.ode_function_library.simple_nonlinear_ode import SimpleNonlinearODEFunction


class TestCase(OzoneODETestCase):

    ode_function_class = SimpleNonlinearODEFunction

    def test(self):
        self.run_error_test()
        self.run_partials_test()
