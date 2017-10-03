from ozone.utils.test_utils import OzoneODETestCase
from ozone.tests.ode_function_library.two_d_orbit_func import TwoDOrbitFunction


class TestCase(OzoneODETestCase):

    ode_function_class = TwoDOrbitFunction

    def test(self):
        self.run_error_test()
        self.run_partials_test()

    def test_doc(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from openmdao.api import Problem
        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.two_d_orbit_func import TwoDOrbitFunction

        ode_function = TwoDOrbitFunction()

        ecc = 1. / 2.
        initial_conditions = {
            'position': np.array([1 - ecc, 0]),
            'velocity': np.array([0, np.sqrt((1+ecc) / (1 - ecc))])
        }
        t0 = 0. * np.pi
        t1 = 1. * np.pi

        num = 100

        times = np.linspace(t0, t1, num)

        method_name = 'RK4'
        formulation = 'solver-based'

        integrator = ODEIntegrator(ode_function, formulation, method_name,
            times=times, initial_conditions=initial_conditions,
        )

        prob = Problem(integrator)
        prob.setup()
        prob.run_model()

        plt.plot(prob['state:position'][:, 0], prob['state:position'][:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
