import numpy as np
import time

from openmdao.api import Problem, ScipyOptimizer, IndepVarComp

from ozone.api import ODEIntegrator
from ozone.utils.suppress_printing import nostdout
from ozone.methods_list import get_method


def compute_runtime(num_times_vector, t0, t1, state_name,
        ode_function, formulation, method_name, initial_conditions):
    num = len(num_times_vector)

    step_sizes_vector = np.zeros(num)
    runtimes_vector = np.zeros(num)

    for ind, num_times in enumerate(num_times_vector):
        times = np.linspace(t0, t1, num_times)

        integrator = ODEIntegrator(ode_function, formulation, method_name,
            times=times, initial_conditions=initial_conditions)
        prob = Problem(integrator)

        if formulation == 'optimizer-based':
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

        step_sizes_vector[ind] = (t1 - t0) / (num - 1)
        runtimes_vector[ind] = time1 - time0

    return step_sizes_vector, runtimes_vector


def compute_ideal_runtimes(step_sizes_vector, runtimes_vector):
    ideal_step_sizes_vector = np.array([
        step_sizes_vector[0],
        step_sizes_vector[-1],
    ])

    ideal_runtimes = np.array([
        runtimes_vector[0],
        runtimes_vector[0] * (step_sizes_vector[0] / step_sizes_vector[-1]),
    ])

    return ideal_step_sizes_vector, ideal_runtimes


def compute_convergence_order(num_times_vector, t0, t1, state_name,
        ode_function, formulation, method_name, initial_conditions):
    num = len(num_times_vector)

    step_sizes_vector = np.zeros(num)
    errors_vector = np.zeros(num)
    orders_vector = np.zeros(num)

    for ind, num_times in enumerate(num_times_vector):
        times = np.linspace(t0, t1, num_times)

        integrator = ODEIntegrator(ode_function, formulation, method_name,
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
