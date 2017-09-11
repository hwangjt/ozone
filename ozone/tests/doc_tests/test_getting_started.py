import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test_ode(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from openmdao.api import Problem

        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.getting_started_ode_function \
            import GettingStartedODEFunction

        # Instantiate our ODE function; use the solver-based formulation;
        # 4th order Runge--Kutta method; 100 time steps from t=0 to t=3; and y0=1.
        ode_function = GettingStartedODEFunction()
        formulation = 'solver-based'
        method_name = 'RK4'
        times = np.linspace(0., 3, 101)
        initial_conditions={'y': 1.}

        # Pass these arguments to ODEIntegrator to get an OpenMDAO group called integrator.
        integrator = ODEIntegrator(ode_function, formulation, method_name,
            times=times, initial_conditions=initial_conditions)

        # Create an OpenMDAO problem instance where the model is just our integrator,
        # then call setup, which is a mandatory step before running, then run the model.
        prob = Problem(model=integrator)
        prob.setup(check=False)
        prob.run_model()

        plt.plot(prob['times'], prob['state:y'][:, 0])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.show()

    def test_oc(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from openmdao.api import Problem, ScipyOptimizer, IndepVarComp, ExecComp

        from ozone.api import ODEIntegrator
        from ozone.tests.ode_function_library.getting_started_oc_function \
            import GettingStartedOCFunction

        num = 21

        # Instantiate our ODE function; use the solver-based formulation;
        # 6th order Gauss--Legendre method; 20 time steps.
        # We only provide the initial time and a normalized times vector
        # since the final time is variable in this problem.
        ode_function = GettingStartedOCFunction(system_init_kwargs={'g': -9.81})
        formulation = 'solver-based'
        method_name = 'GaussLegendre6'
        initial_time = 0.
        normalized_times = np.linspace(0., 1, num)
        initial_conditions={'x': 0., 'y': 0., 'v': 0.}

        # Pass these arguments to ODEIntegrator to get an OpenMDAO group called integrator.
        integrator = ODEIntegrator(ode_function, formulation, method_name,
            initial_time=initial_time, normalized_times=normalized_times,
            initial_conditions=initial_conditions)

        prob = Problem()

        # Define independent variable components for final time and theta.
        # Final time and theta are, simultaneously, component outputs and model inputs.
        # We add our integrator group and components for our transversality conditions.
        prob.model.add_subsystem('final_time_comp', IndepVarComp('final_time', val=1.0))
        prob.model.add_subsystem('theta_comp', IndepVarComp('theta', shape=(num, 1)))
        prob.model.add_subsystem('integrator_group', integrator)
        prob.model.add_subsystem('x_constraint_comp', ExecComp('x_con = x - 2.'))
        prob.model.add_subsystem('y_constraint_comp', ExecComp('y_con = y + 2.'))

        # We issue connections using 'connect(output_name, input_name)'.
        # src_indices is used when we just want to pull out a subset of entries in a larger array.
        prob.model.connect('final_time_comp.final_time', 'integrator_group.final_time')
        prob.model.connect('theta_comp.theta', 'integrator_group.dynamic_parameter:theta')
        prob.model.connect('integrator_group.state:x', 'x_constraint_comp.x', src_indices=-1)
        prob.model.connect('integrator_group.state:y', 'y_constraint_comp.y', src_indices=-1)

        # We add the final time and theta as design variables, declare final time as the objective
        # and add the transversality constraints.
        prob.model.add_design_var('final_time_comp.final_time', lower=0.5)
        prob.model.add_design_var('theta_comp.theta')
        prob.model.add_objective('final_time_comp.final_time')
        prob.model.add_constraint('x_constraint_comp.x_con', equals=0.)
        prob.model.add_constraint('y_constraint_comp.y_con', equals=0.)

        # We set the SLSQP optimizer as our driver in this problem.
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-6
        prob.driver.options['disp'] = True

        prob.setup(check=False)
        prob.run_driver()

        plt.plot(prob['integrator_group.state:x'][:, 0], prob['integrator_group.state:y'][:, 0])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


if __name__ == '__main__':
    unittest.main()
