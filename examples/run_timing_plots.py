import numpy as np
import matplotlib.pylab as plt
from six import iteritems
import time

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp

from openode.api import ODEFunction, ODEIntegrator
from openode.tests.ode_functions.simple_ode import LinearODEFunction, SimpleODEFunction, \
    NonlinearODEFunction
from openode.utils.suppress_printing import nostdout
from openode.utils.misc import get_scheme_families


ode_function = NonlinearODEFunction()
ode_function = LinearODEFunction()
# ode_function = SimpleODEFunction()

initial_conditions = {'y': 1.}
t0 = 1.e-7
t1 = 1.e-6

nums = [21, 26, 31, 36]
# nums = [5]
# nums = [11, 21, 31, 41]

integrator_names = ['TM', 'MDF', 'SAND']

scheme_family_name = 'basic'

scheme_families = get_scheme_families()

colors = ['b', 'g', 'r', 'c', 'm', 'k']


plt.figure(figsize=(20, 15))

plot_index = 0
for scheme_family_name, scheme_family in iteritems(scheme_families):
    print(scheme_family_name)

    plot_index += 1
    plt.subplot(3, 3, plot_index)

    scheme_name, order = scheme_family[1]

    for j, integrator_name in enumerate(integrator_names):

        step_sizes = np.zeros(len(nums))
        run_times = np.zeros(len(nums))
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
                time0 = time.time()
                prob.run_driver()
                time1 = time.time()

            step_sizes[i] = (t1 - t0) / (num - 1)
            run_times[i] = time1 - time0

        plt.loglog(step_sizes, run_times, colors[j] + 'o-')
        plt.loglog(
            [step_sizes[0], step_sizes[-1]],
            [run_times[0], run_times[0] * (step_sizes[0] / step_sizes[-1])],
            colors[j] + ':',
        )

    legend_entries = []
    for integrator_name in integrator_names:
        legend_entries.append(integrator_name)
        legend_entries.append('linear')

    plt.title(scheme_name)
    plt.legend(legend_entries)

plt.savefig("time_vs_stepsize_plots.pdf")
