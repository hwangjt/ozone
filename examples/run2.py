import numpy as np

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp, view_model

from openode.api import ODEFunction, ode_integrator_group
from openode.tests.ode_functions.simple_ode import SimpleODEFunction


num = 5

scheme_name = 'RK4'

# integrator_name = 'SAND'
integrator_name = 'MDF'
# integrator_name = 'TM'

ode_function = SimpleODEFunction()

integrator = ode_integrator_group(ode_function, integrator_name, scheme_name,
    time_spacing=np.arange(num), initial_conditions={'y': 1.}, start_time=0., end_time=1.,)

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

print(prob['output_comp.y'])

# view_model(prob)
