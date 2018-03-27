import unittest

import matplotlib
matplotlib.use('Agg')

try:
    import niceplots
except:
    pass


class Test(unittest.TestCase):

    def test(self):
        import numpy as np
        import matplotlib.pylab as plt

        from ozone.tests.ode_function_library.simple_homogeneous_func import \
            SimpleHomogeneousODEFunction
        from ozone.utils.run_utils import compute_convergence_order, compute_ideal_error
        from ozone.methods_list import family_names, method_families

        num_times_vector = np.array([10, 15, 20])

        ode_function = SimpleHomogeneousODEFunction()

        initial_conditions = {'y': 1.}
        t0 = 0.
        t1 = 1.

        state_name = 'y'

        formulation = 'solver-based'

        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
        # plt.figure(figsize=(14, 17))
        plt.figure(figsize=(15, 9))

        nrow = 3
        ncol = 4

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

                plt.loglog(step_sizes_vector * 1e2, errors_vector, colors[j] + 'o-')
                plt.loglog(ideal_step_sizes_vector * 1e2, ideal_errors_vector, colors[j] + ':', label='_nolegend_')

                average_order = np.sum(orders_vector) / len(orders_vector)

                print('(Ideal, observed): ', ideal_order, average_order )

                legend_entries.append(method_name + ' ({})'.format(ideal_order))
                # legend_entries.append('order %s' % ideal_order)

            irow = int(np.floor(plot_index / ncol))
            icol = plot_index % ncol

            plt.title(family_name)
            if irow == nrow - 1:
                plt.xlabel('step size (x 1e-2)')
            if icol == 0:
                plt.ylabel('error')
            plt.legend(legend_entries)
            ax = plt.gca()
            ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            try:
                niceplots.adjust_spines(ax=plt.gca())
            except:
                pass

            print()

        plt.show()


if __name__ == '__main__':
    import matplotlib.pylab as plt

    Test().test()
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig('order_plot.pdf')
