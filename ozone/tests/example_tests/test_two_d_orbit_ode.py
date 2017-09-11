from ozone.utils.test_utils import OzoneODETestCase
from ozone.tests.ode_function_library.two_d_orbit_func import TwoDOrbitFunction


class TestCase(OzoneODETestCase):

    ode_function_class = TwoDOrbitFunction

    def test(self):
        self.run_error_test()
        self.run_partials_test()
