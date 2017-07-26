import numpy as np
import matplotlib.pylab as plt
from six import iteritems

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp

from openode.api import ODEFunction, ODEIntegrator
from openode.tests.ode_functions.simple_ode import LinearODEFunction, SimpleODEFunction, \
    NonlinearODEFunction
from openode.utils.suppress_printing import nostdout
from openode.utils.misc import get_scheme_families


ode_function = NonlinearODEFunction()
# ode_function = LinearODEFunction()
# ode_function = SimpleODEFunction()

initial_conditions = {'y': 1.}
t0 = 0.
t1 = 1.

nums = [21, 26, 31, 36]
# nums = [5]
# nums = [11, 21, 31, 41]

# integrator_name = 'SAND'
integrator_name = 'MDF'
# integrator_name = 'TM'

scheme_family_name = 'basic'

scheme_families = get_scheme_families()

colors = ['b', 'g', 'r', 'c', 'm', 'k']


plt.figure(figsize=(20, 15))

plot_index = 0
for scheme_family_name, scheme_family in iteritems(scheme_families):
    print(scheme_family_name)

    plot_index += 1
    plt.subplot(3, 3, plot_index)

    for j, (scheme_name, order) in enumerate(scheme_family):

        step_sizes = np.zeros(len(nums))
        errs = np.zeros(len(nums))
        for i, num in enumerate(nums):
            times = np.linspace(t0, t1, num)

            integrator = ODEIntegrator(ode_function, integrator_name, scheme_name,
                times=times, initial_conditions=initial_conditions)
            prob = Problem(integrator)

            if integrator_name == 'SAND':
                prob.driver = ScipyOptimizer()
                prob.driver.options['optimizer'] = 'SLSQP'
                prob.driver.options['tol'] = 1e-9
                prob.driver.options['disp'] = True

                integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var', val=1.0))
                integrator.add_objective('dummy_comp.dummy_var')

            with nostdout():
                prob.setup()
                prob.run_driver()

            approx_y = prob['state:y'][-1][0]
            true_y = ode_function.compute_exact_soln(initial_conditions, t0, t1)['y']

            step_sizes[i] = (t1 - t0) / (num - 1)
            errs[i] = np.linalg.norm(approx_y - true_y)

        plt.loglog(step_sizes, errs, colors[j] + 'o-')
        plt.loglog(
            [step_sizes[0], step_sizes[-1]],
            [errs[0], errs[0] * (step_sizes[-1] / step_sizes[0]) ** order],
            colors[j] + ':',
        )

    legend_entries = []
    for scheme_name, order in scheme_family:
        legend_entries.append(scheme_name)
        legend_entries.append('order %s' % order)

    plt.legend(legend_entries)

plt.savefig("order_vs_stepsize_plots.pdf")
