import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test(self):
        import numpy as np
        import matplotlib.pylab as plt

        from ozone.tests.ode_function_library.simple_linear_func import \
            SimpleLinearODEFunction
        from ozone.utils.run_utils import compute_runtime, compute_ideal_runtimes
        from ozone.methods_list import family_names, method_families

        num_times_vector = np.array([10, 15, 20])
        num_times_vector = np.array([21, 26, 31, 36])

        ode_function = SimpleLinearODEFunction()

        initial_conditions = {'y': 1.}
        t0 = 0.
        t1 = 1.

        state_name = 'y'

        formulations = ['time-marching', 'solver-based', 'optimizer-based']

        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
        plt.figure(figsize=(25, 25))

        nrow = 4
        ncol = 3

        for plot_index, family_name in enumerate(family_names):
            method_family = method_families[family_name]
            method_name = method_family[1]

            print(method_family, method_name)

            plt.subplot(nrow, ncol, plot_index + 1)

            legend_entries = []

            for j, formulation in enumerate(formulations):

                step_sizes_vector, runtimes_vector = compute_runtime(
                    num_times_vector, t0, t1, state_name,
                    ode_function, formulation, method_name, initial_conditions)

                ideal_step_sizes_vector, ideal_runtimes = compute_ideal_runtimes(
                    step_sizes_vector, runtimes_vector)

                plt.loglog(step_sizes_vector, runtimes_vector, colors[j] + 'o-')
                plt.loglog(ideal_step_sizes_vector, ideal_runtimes, colors[j] + ':')

                print('({} time): '.format(formulation), runtimes_vector )

                legend_entries.append(formulation)
                legend_entries.append('linear')

            irow = int(np.floor(plot_index / ncol))
            icol = plot_index % ncol

            plt.title(family_name)
            if irow == nrow - 1:
                plt.xlabel('step size')
            if icol == 0:
                plt.ylabel('computation time (s)')
            plt.legend(legend_entries)

            print()

        plt.show()


if __name__ == '__main__':
    import matplotlib.pylab as plt

    Test().test()
    plt.savefig('timing_plot.pdf')
