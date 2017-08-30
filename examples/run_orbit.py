import numpy as np

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp

from ozone.api import ODEFunction, ODEIntegrator
from ozone.tests.ode_functions.two_d_orbit import TwoDOrbitFunction


ode_function = TwoDOrbitFunction()
# ode_function = LinearODEFunction()
# ode_function = SimpleODEFunction()

# nums = [11, 16, 21, 26, 31, 36]
# nums = [5]
# nums = [11, 21, 31, 41]
# nums = [51, 101, 151, 201]
# nums = [51, 76, 101, 151]
nums = [101]

# method_name = 'AdamsPECE3'
# method_name = 'BDF4'
# method_name = 'ForwardEuler'
# method_name = 'RK4'
method_name = 'GaussLegendre4'
# method_name = 'Trapezoidal'
# method_name = 'Lobatto4'
# method_name = 'Radau5'

# formulation = 'optimizer-based'
formulation = 'solver-based'
# formulation = 'time-marching'

ecc = 1 / 2

initial_conditions = {'position': np.array([1 - ecc, 0]),
                      'velocity': np.array([0, np.sqrt((1+ecc) / (1 - ecc))])}
t0 = 0.
t1 = 1*np.pi


errs = np.zeros(len(nums))
for i, num in enumerate(nums):
    times = np.linspace(t0, t1, num)

    y_true = np.array([ode_function.compute_exact_soln(initial_conditions, t0, t) for t in times])

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

    prob.setup()
    # prob['coupled_group.vectorized_step_comp.y:y'] = y_true.reshape((num, 1 ,1))
    prob.run_driver()

    approx_y = prob['state:position'][-1]
    true_y = ode_function.compute_exact_soln(initial_conditions, t0, t1)

    errs[i] = np.linalg.norm(approx_y - true_y)


print('-'*40)
print('| {:10s} | {:10s} | {:10s} |'.format('h', 'Error', 'Rate'))
print('-'*40)
for i in range(len(nums)):
    h0 = (t1 - t0) / (nums[i - 1] - 1)
    h1 = (t1 - t0) / (nums[i] - 1)
    err0 = errs[i - 1]
    err1 = errs[i]

    if i == 0:
        rate = 0.
    else:
        rate = np.log(err1 / err0) / np.log(h1 / h0)

    print('| {:.4e} | {:.4e} | {:.4e} |'.format(h1, err1, rate))

# import matplotlib.pyplot as plt
# plt.plot(prob['state:position'][:, 0], prob['state:position'][:, 1])
# plt.axis('equal')
# plt.show()