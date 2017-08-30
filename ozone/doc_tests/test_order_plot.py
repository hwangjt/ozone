import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test(self):
        import numpy as np
        import matplotlib.pylab as plt

        from ozone.tests.ode_functions.simple_ode import LinearODEFunction, SimpleODEFunction, \
            NonlinearODEFunction
        from ozone.utils.compute_order import compute_convergence_order, compute_ideal_error
        from ozone.methods_list import family_names, method_families

        num_times_vector = np.array([10, 15, 20])

        ode_function = SimpleODEFunction()

        initial_conditions = {'y': 1.}
        t0 = 0.
        t1 = 1.

        state_name = 'y'

        formulation = 'solver-based'

        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
        plt.figure(figsize=(25, 25))

        plot_index = 0
        for family_name in family_names:
            method_family = method_families[family_name]

            print(method_family)

            plot_index += 1
            plt.subplot(4, 3, plot_index)

            legend_entries = []
            for j, method_name in enumerate(method_family):

                errors_vector, step_sizes_vector, orders_vector, ideal_order = compute_convergence_order(
                    num_times_vector, t0, t1, state_name,
                    ode_function, formulation, method_name, initial_conditions)

                ideal_step_sizes_vector, ideal_errors_vector = compute_ideal_error(
                    step_sizes_vector, errors_vector, ideal_order)

                plt.loglog(step_sizes_vector, errors_vector, colors[j] + 'o-')
                plt.loglog(ideal_step_sizes_vector, ideal_errors_vector, colors[j] + ':')

                average_order = np.sum(orders_vector) / len(orders_vector)

                print('(Ideal, observed): ', ideal_order, average_order )

                legend_entries.append(method_name)
                legend_entries.append('order %s' % ideal_order)

            plt.legend(legend_entries)

            print()

        plt.show()
