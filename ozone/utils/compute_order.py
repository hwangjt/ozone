import numpy as np

from openmdao.api import Problem

from ozone.api import ODEIntegrator
from ozone.utils.suppress_printing import nostdout
from ozone.methods_list import get_method


def compute_convergence_order(num_times_vector, t0, t1, state_name,
        ode_function, integrator_name, method_name, initial_conditions):

    num = len(num_times_vector)

    errors_vector = np.zeros(num)
    step_sizes_vector = np.zeros(num)
    orders_vector = np.zeros(num)

    for ind, num_times in enumerate(num_times_vector):
        times = np.linspace(t0, t1, num_times)

        integrator = ODEIntegrator(ode_function, integrator_name, method_name,
            times=times, initial_conditions=initial_conditions)
        prob = Problem(integrator)

        with nostdout():
            prob.setup()
            prob.run_driver()

        approx_y = prob['state:%s' % state_name][-1][0]
        true_y = ode_function.compute_exact_soln(initial_conditions, t0, t1)[state_name]

        errors_vector[ind] = np.linalg.norm(approx_y - true_y)
        step_sizes_vector[ind] = (t1 - t0) / (num_times - 1)

    errors0 = errors_vector[:-1]
    errors1 = errors_vector[1:]

    step_sizes0 = step_sizes_vector[:-1]
    step_sizes1 = step_sizes_vector[1:]

    orders_vector = np.log( errors1 / errors0 ) / np.log( step_sizes1 / step_sizes0 )

    ideal_order = get_method(method_name).order

    return errors_vector, step_sizes_vector, orders_vector, ideal_order


def compute_ideal_error(step_sizes_vector, errors_vector, ideal_order):
    ideal_step_sizes_vector = np.array([
        step_sizes_vector[0],
        step_sizes_vector[-1],
    ])

    ideal_errors_vector = np.array([
        errors_vector[0],
        errors_vector[0] * ( step_sizes_vector[-1] / step_sizes_vector[0] ) ** ideal_order,
    ])

    return ideal_step_sizes_vector, ideal_errors_vector
