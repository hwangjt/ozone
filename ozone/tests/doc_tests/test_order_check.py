import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test(self):
        import numpy as np

        from ozone.tests.ode_function_library.simple_ode import NonlinearODEFunction
        from ozone.utils.run_utils import compute_convergence_order


        num_times_vector = np.array([16, 32, 64, 128, 256])
        method_name = 'ImplicitMidpoint'
        formulation = 'solver-based'

        ode_function = NonlinearODEFunction()
        state_name = 'y'
        initial_conditions = {'y': 1.}
        t0 = 0.
        t1 = 1.

        errors_vector, step_sizes_vector, orders_vector, ideal_order = compute_convergence_order(
            num_times_vector, t0, t1, state_name,
            ode_function, formulation, method_name, initial_conditions)

        print('-'*47)
        print('| {:4s} | {:10s} | {:10s} | {:10s} |'.format('Num.', 'h', 'Error', 'Rate'))
        print('-'*47)
        for i in range(len(num_times_vector)):
            print('| {:4d} | {:.4e} | {:.4e} | {:.4e} |'.format(
                num_times_vector[i],
                step_sizes_vector[i],
                errors_vector[i],
                orders_vector[i - 1] if i != 0 else 0.,
            ))
