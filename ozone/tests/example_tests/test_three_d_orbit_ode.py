from ozone.utils.test_utils import OzoneODETestCase
from ozone.tests.ode_function_library.three_d_orbit_func import ThreeDOrbitFunction


class TestCase(OzoneODETestCase):

    ode_function_class = ThreeDOrbitFunction

    def test(self):
        self.run_partials_test()

    def test_doc(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from openmdao.api import Problem
        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.three_d_orbit_func import ThreeDOrbitFunction

        r_scal = 1e12
        v_scal = 1e3

        ode_function = ThreeDOrbitFunction()

        t0 = 0.
        t1 = 348.795 * 24 * 3600

        initial_conditions = {
            'r': np.array([ -140699693 , -51614428 , 980 ]) * 1e3 / r_scal,
            'v': np.array([ 9.774596 , -28.07828 , 4.337725e-4 ]) * 1e3 / v_scal,
            'm': 1000.
        }

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

        au = 149597870.7 * 1e3 / r_scal
        plt.plot(prob['state:r'][:, 0] / au, prob['state:r'][:, 1] / au, '-o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
