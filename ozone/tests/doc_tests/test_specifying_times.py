import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test_times(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from openmdao.api import Problem

        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.getting_started_ode_function \
            import GettingStartedODEFunction

        ode_function = GettingStartedODEFunction()
        formulation = 'solver-based'
        method_name = 'RK4'
        initial_conditions = {'y': 1.}

        # Only times is passed in.
        times = np.linspace(0., 3., 101)

        integrator = ODEIntegrator(ode_function, formulation, method_name,
            times=times, initial_conditions=initial_conditions)

        prob = Problem(model=integrator)
        prob.setup(check=False)
        prob.run_model()

        plt.plot(prob['times'], prob['state:y'][:, 0])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.show()

    def test_normalized_dict(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from openmdao.api import Problem

        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.getting_started_ode_function \
            import GettingStartedODEFunction

        ode_function = GettingStartedODEFunction()
        formulation = 'solver-based'
        method_name = 'RK4'
        initial_conditions = {'y': 1.}

        # Here, initial_time, final_time, and normalized_times are passed in.
        initial_time = 0.
        final_time = 3.
        normalized_times = np.linspace(0., 1., 101)

        integrator = ODEIntegrator(ode_function, formulation, method_name,
            initial_time=initial_time, final_time=final_time,
            normalized_times=normalized_times, initial_conditions=initial_conditions)

        prob = Problem(model=integrator)
        prob.setup(check=False)
        prob.run_model()

        plt.plot(prob['times'], prob['state:y'][:, 0])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.show()

    def test_normalized_connected(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from openmdao.api import Problem, IndepVarComp

        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.getting_started_ode_function \
            import GettingStartedODEFunction

        ode_function = GettingStartedODEFunction()
        formulation = 'solver-based'
        method_name = 'RK4'
        initial_conditions={'y': 1.}

        # Only normalized_times is passed in
        normalized_times = np.linspace(0., 1., 101)

        integrator = ODEIntegrator(ode_function, formulation, method_name,
            normalized_times=normalized_times, initial_conditions=initial_conditions)

        # Below, initial_time and final_time are connected from external components.
        prob = Problem()
        prob.model.add_subsystem('initial_time_comp', IndepVarComp('initial_time', 0.))
        prob.model.add_subsystem('final_time_comp', IndepVarComp('final_time', 3.))
        prob.model.add_subsystem('integrator_group', integrator)
        prob.model.connect('initial_time_comp.initial_time', 'integrator_group.initial_time')
        prob.model.connect('final_time_comp.final_time', 'integrator_group.final_time')
        prob.setup(check=False)
        prob.run_model()

        plt.plot(prob['integrator_group.times'], prob['integrator_group.state:y'][:, 0])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.show()
