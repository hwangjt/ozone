import numpy as np
import time

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp, view_model, ExecComp

from ozone.api import ODEFunction, ODEIntegrator
from ozone.tests.ode_functions.simple_ode import NonlinearODEFunction, LinearODEFunction, \
    SimpleODEFunction
from ozone.tests.ode_functions.cannonball import CannonballODEFunction


num = 5

t0 = 0.
t1 = 1.e-2
# initial_conditions = {'y': 1.}
initial_conditions = {'x': 0., 'y': 0., 'vx': 0.1, 'vy': 0.}

times = np.linspace(t0, t1, num)

scheme_name = 'ForwardEuler'
# scheme_name = 'RK4'
# scheme_name = 'ImplicitMidpoint'
# scheme_name = 'AM4'
# scheme_name = 'BDF2'

integrator_name = 'SAND'
# integrator_name = 'MDF'
# integrator_name = 'TM'

# ode_function = LinearODEFunction()
ode_function = CannonballODEFunction()

integrator = ODEIntegrator(ode_function, integrator_name, scheme_name,
    times=times, initial_conditions=initial_conditions,
    parameters={'g': np.linspace(9.80665, 9.80665, num).reshape((num, 1))})

prob = Problem(integrator)

if integrator_name == 'SAND':
    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var'))
    integrator.add_objective('dummy_comp.dummy_var')

prob.setup()
time0 = time.time()
prob.run_driver()
time1 = time.time()
# prob.check_partials(compact_print=True)
# prob.check_partials(compact_print=False)

np.set_printoptions(precision=10)

exact_soln = ode_function.compute_exact_soln(initial_conditions, t0, t1)

for key in exact_soln:
    print('Error in state %s at final time:' % key,
        np.linalg.norm(prob['state:%s' % key][-1] - exact_soln[key]))
print('Runtime (s):', time1 - time0)
# print(prob['starting:y'])
# view_model(prob)

import matplotlib.pyplot as plt
plt.plot(prob['state:x'], prob['state:y'])
plt.show()

# print(prob['starting_system.coupled_group.vectorized_stage_comp.Y_out:y'],
#     prob['starting_system.coupled_group.vectorized_stage_comp.Y_out:y'].shape)
# print('----------------')
# print(prob['starting_system.coupled_group.ode_comp.dy_dt'])
# print('----------------')
# print(prob['starting_system.coupled_group.vectorized_step_comp.y:y'])
