from ozone.utils.test_utils import OzoneODETestCase
from ozone.tests.ode_function_library.simple_linear_func import SimpleLinearODEFunction


class TestCase(OzoneODETestCase):

    ode_function_class = SimpleLinearODEFunction

    def test(self):
        self.run_error_test()
        self.run_partials_test()

    def test_doc(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from openmdao.api import Problem
        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.simple_linear_func import SimpleLinearODEFunction

        ode_function = SimpleLinearODEFunction()

        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}

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

        plt.plot(prob['times'], prob['state:y'])
        plt.show()
