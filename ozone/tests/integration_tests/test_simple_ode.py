from __future__ import division
import numpy as np
import unittest
import scipy.integrate
from six import iteritems, itervalues
from itertools import product
from parameterized import parameterized

from openmdao.api import Problem, ScipyOptimizer, IndepVarComp

from ozone.api import ODEIntegrator
from ozone.tests.ode_function_library.simple_linear_ode import SimpleLinearODEFunction
from ozone.tests.ode_function_library.simple_nonlinear_ode import SimpleNonlinearODEFunction
from ozone.utils.suppress_printing import suppress_stdout_stderr


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def run_ode(self, formulation, method_name, ode_function):
        times = np.linspace(0., 1.e-2, 7)
        y0 = -1.

        integrator = ODEIntegrator(ode_function, formulation, method_name,
            times=times, initial_conditions={'y': y0},)

        prob = Problem(integrator)

        if formulation == 'optimizer-based':
            prob.driver = ScipyOptimizer()
            prob.driver.options['optimizer'] = 'SLSQP'
            prob.driver.options['tol'] = 1e-9
            prob.driver.options['disp'] = True

            integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var'))
            integrator.add_objective('dummy_comp.dummy_var')

        with suppress_stdout_stderr():
            prob.setup(check=False)
            prob.run_driver()

        return prob

    def compute_diff(self, formulation, method_name, ode_function, y_ref):
        y = self.run_ode(formulation, method_name, ode_function)['state:y']

        return np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)

    @parameterized.expand(product(
        [
            'ForwardEuler', 'BackwardEuler', 'ExplicitMidpoint', 'ImplicitMidpoint',
            'AB2', 'AB3', 'AB4', 'AB5',
            'AM2', 'AM3', 'AM4', 'AM5',
            'BDF2', 'BDF3', 'BDF4', 'BDF5', 'BDF6',
            'AdamsPEC2', 'AdamsPEC5',
            'AdamsPECE2', 'AdamsPECE5',
        ],  # method
        [SimpleLinearODEFunction(), SimpleNonlinearODEFunction()],  # ODE Function
        ['time-marching', 'solver-based', 'optimizer-based']
    ))
    def test_tm(self, method_name, ode_function, formulation):

        y_ref = self.run_ode('time-marching', method_name, ode_function)['state:y']
        diff = self.compute_diff(formulation, method_name, ode_function, y_ref)
        print('%20s %5s %16.9e' % (method_name, formulation, diff))
        self.assertTrue(diff < 1e-10, 'Error when integrating with %s %s' % (
            formulation, method_name))

    @parameterized.expand(product(
        ['time-marching', 'solver-based', 'optimizer-based']
    ))
    def test_derivs(self, formulation):
        ode_function = SimpleNonlinearODEFunction()
        prob = self.run_ode(formulation, 'ForwardEuler', ode_function)
        with suppress_stdout_stderr():
            jac = prob.check_partials(compact_print=True)
        for comp_name, jac_comp in iteritems(jac):
            for partial_name, jac_partial in iteritems(jac_comp):
                mag_fd = jac_partial['magnitude'].fd
                mag_fwd = jac_partial['magnitude'].forward
                mag_rev = jac_partial['magnitude'].reverse

                abs_fwd = jac_partial['abs error'].forward
                abs_rev = jac_partial['abs error'].reverse

                rel_fwd = jac_partial['rel error'].forward
                rel_rev = jac_partial['rel error'].reverse

                non_zero = np.max([mag_fd, mag_fwd, mag_rev]) > 1e-12
                if non_zero:
                    # print('%16.9e %16.9e %5s %s %s' % (
                    #     jac_partial['rel error'].forward,
                    #     jac_partial['rel error'].reverse,
                    #     formulation, comp_name, partial_name,
                    # ))
                    self.assertTrue(rel_fwd < 1e-3 or abs_fwd < 1e-3)
                    self.assertTrue(rel_rev < 1e-3 or abs_rev < 1e-3)



if __name__ == '__main__':
    unittest.main()
