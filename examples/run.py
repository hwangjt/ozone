import numpy as np
import time

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp, view_model

from openode.api import ODEFunction, ODEIntegrator
from openode.tests.ode_functions.simple_ode import NonlinearODEFunction, LinearODEFunction, \
    SimpleODEFunction


num = 11

t0 = 0.
t1 = 1.
initial_conditions = {'y': 1.}

times = np.linspace(t0, t1, num)

scheme_name = 'RK4'
# scheme_name = 'AB2'
scheme_name = 'BDF5'

integrator_name = 'SAND'
integrator_name = 'MDF'
integrator_name = 'TM'

ode_function = NonlinearODEFunction()

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

prob.setup()
time0 = time.time()
prob.run_driver()
time1 = time.time()
# prob.check_partials(compact_print=True)
# prob.check_partials(compact_print=False)

np.set_printoptions(precision=10)

exact_soln = ode_function.compute_exact_soln(initial_conditions, t0, t1)
print(prob['state:y'])
print('Error in state value at final time:', np.linalg.norm(prob['state:y'][-1] - exact_soln))
print('Runtime (s):', time1 - time0)
# print(prob['starting:y'])
# view_model(prob)

# print(prob['starting_system.coupled_group.vectorized_stage_comp.Y_out:y'],
#     prob['starting_system.coupled_group.vectorized_stage_comp.Y_out:y'].shape)
# print('----------------')
# print(prob['starting_system.coupled_group.ode_comp.dy_dt'])
# print('----------------')
# print(prob['starting_system.coupled_group.vectorized_step_comp.y:y'])
