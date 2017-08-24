import numpy as np
import matplotlib.pylab as plt
from six import iteritems

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp

from ozone.api import ODEFunction, ODEIntegrator
from ozone.tests.ode_functions.simple_ode import LinearODEFunction, SimpleODEFunction, \
    NonlinearODEFunction
from ozone.utils.suppress_printing import nostdout
from ozone.utils.compute_order import compute_convergence_order, compute_ideal_error
from ozone.utils.misc import scheme_families


num_time_steps_vector = np.array([10, 15, 20])

ode_function = NonlinearODEFunction()
# ode_function = LinearODEFunction()
ode_function = SimpleODEFunction()

initial_conditions = {'y': 1.}
t0 = 0.
t1 = 1.

state_name = 'y'

integrator_name = 'MDF'

colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
plt.figure(figsize=(30, 15))

plot_index = 0
for scheme_family_name, scheme_family in iteritems(scheme_families):

    print(scheme_family_name)

    plot_index += 1
    plt.subplot(3, 3, plot_index)

    for j, scheme_name in enumerate(scheme_family):

        errors_vector, step_sizes_vector, orders_vector, ideal_order = compute_convergence_order(
            num_time_steps_vector, t0, t1, state_name,
            ode_function, integrator_name, scheme_name, initial_conditions)

        ideal_step_sizes_vector, ideal_errors_vector = compute_ideal_error(
            step_sizes_vector, errors_vector, ideal_order)

        plt.loglog(step_sizes_vector, errors_vector, colors[j] + 'o-')
        plt.loglog(ideal_step_sizes_vector, ideal_errors_vector, colors[j] + ':')

        average_order = np.sum(orders_vector) / len(orders_vector)

        print('(Ideal, observed): ', ideal_order, average_order )

    legend_entries = []
    for scheme_name in scheme_family:
        legend_entries.append(scheme_name)
        legend_entries.append('order %s' % ideal_order)

    plt.legend(legend_entries)

    print()

plt.savefig("order_vs_stepsize_plots.pdf")
