import numpy as np

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp, view_model

from openode.api import ODEFunction, ODEIntegrator
from openode.tests.ode_functions.simple_ode import LinearODEFunction, SimpleODEFunction


num = 7

scheme_name = 'BDF4'
# scheme_name = 'BackwardEuler'

# integrator_name = 'SAND'
integrator_name = 'MDF'
# integrator_name = 'TM'

ode_function = SimpleODEFunction()

integrator = ODEIntegrator(ode_function, integrator_name, scheme_name,
    times=np.linspace(0., num, num+1), initial_conditions={'y': -1.})

prob = Problem(integrator)

if integrator_name == 'SAND':
    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var', val=1.0))
    integrator.add_objective('dummy_comp.dummy_var')

prob.setup()
prob.run_driver()
# prob.check_partials(compact_print=True)
# prob.check_partials(compact_print=False)

print(prob['state:y'])
# print(prob['starting:y'])
# view_model(prob)
