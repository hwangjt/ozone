import unittest

import matplotlib
matplotlib.use('Agg')


class Test(unittest.TestCase):

    def test(self):
        import numpy as np
        import matplotlib.pylab as plt

        from ozone.doc_tests.ode_functions.projectile_dynamics import run_projectile
        from ozone.methods_list import family_names, method_families

        num = 100

        initial_conditions = {'y': 1.}
        t0 = 0.
        t1 = 1.

        state_name = 'y'

        formulation = 'solver-based'

        plt.figure(figsize=(25, 25))
        fig, ax = plt.subplots()

        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'tab:pink']

        family_names = [
            'ExplicitRungeKutta',
            'ImplicitRungeKutta',
            'GaussLegendre',
            'Lobatto',
            'Radau',
            'AB',
            'AM',
            'BDF',
        ]

        legend_entries = []

        plot_index = 0
        for i, family_name in enumerate(family_names):
            method_family = method_families[family_name]

            print(method_family)

            errors = []
            run_times = []
            names = []

            for j, method_name in enumerate(method_family):

                error, run_time = run_projectile(method_name, 'solver-based', num)

                errors.append(error)
                run_times.append(run_time)
                names.append(method_name)

            plt.loglog(errors, run_times, 'o', color=colors[i])
            legend_entries.append(family_name)

            for method_name, error, run_time in zip(names, errors, run_times):
                ax.annotate(method_name, (error, run_time))

        plt.xlabel('error')
        plt.ylabel('run time (s)')
        plt.legend(legend_entries)
        plt.show()


if __name__ == '__main__':
    unittest.main()
