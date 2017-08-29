import numpy as np

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp

from ozone.api import ODEFunction, ODEIntegrator
from ozone.tests.ode_functions.simple_ode import LinearODEFunction, SimpleODEFunction, \
    NonlinearODEFunction


ode_function = NonlinearODEFunction()
# ode_function = LinearODEFunction()
# ode_function = SimpleODEFunction()

nums = [11, 16, 21, 26, 31, 36]
# nums = [5]
# nums = [11, 21, 31, 41]

# method_name = 'AdamsPECE3'
# method_name = 'ForwardEuler'
# method_name = 'RK4'
# method_name = 'GaussLegendre6'
# method_name = 'Lobatto4'
method_name = 'RadauI5'

# integrator_name = 'SAND'
integrator_name = 'MDF'
# integrator_name = 'TM'

t0 = 0.
t1 = 1.

C1 = 0.6
if C1 > 0:
    assert 2*C1 > t1**2
else:
    assert C1 != 0.

initial_conditions = {'y': 1./C1}
t0 = 0.
t1 = 1.


errs = np.zeros(len(nums))
for i, num in enumerate(nums):
    times = np.linspace(t0, t1, num)

    y_true = np.array([ode_function.compute_exact_soln(initial_conditions, t0, t) for t in times])

    integrator = ODEIntegrator(ode_function, integrator_name, method_name,
        times=times, initial_conditions=initial_conditions)
    prob = Problem(integrator)

    if integrator_name == 'SAND':
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var', val=1.0))
        integrator.add_objective('dummy_comp.dummy_var')

    prob.setup()
    # prob['coupled_group.vectorized_step_comp.y:y'] = y_true.reshape((num, 1 ,1))
    prob.run_driver()

    approx_y = prob['state:y'][-1][0]
    true_y = ode_function.compute_exact_soln(initial_conditions, t0, t1)['y']

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
