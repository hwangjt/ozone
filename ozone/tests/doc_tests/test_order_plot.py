import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test(self):
        import numpy as np
        import matplotlib.pylab as plt

        from ozone.tests.ode_function_library.simple_ode import \
            LinearODEFunction, SimpleODEFunction, NonlinearODEFunction
        from ozone.utils.run_utils import compute_convergence_order, compute_ideal_error
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

        nrow = 4
        ncol = 3

        for plot_index, family_name in enumerate(family_names):
            method_family = method_families[family_name]

            print(method_family)

            plt.subplot(nrow, ncol, plot_index + 1)

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

            irow = int(np.floor(plot_index / ncol))
            icol = plot_index % ncol

            plt.title(family_name)
            if irow == nrow - 1:
                plt.xlabel('step size')
            if icol == 0:
                plt.ylabel('error')
            plt.legend(legend_entries)

            print()

        plt.show()


if __name__ == '__main__':
    import matplotlib.pylab as plt

    Test().test()
    plt.savefig('order_plot.pdf')
