import numpy as np

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp, view_model

from openode.api import ODEFunction, ode_integrator_group


class Comp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=np.eye(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: e^t + sin(2*pi*t)
        two_pi_t = 2*np.pi*inputs['t']
        outputs['dy_dt'] = inputs['y'] + 2*np.pi*np.cos(two_pi_t) - np.sin(two_pi_t)

    def compute_partials(self, inputs, outputs, partials):
        two_pi_t = 2*np.pi*inputs['t']
        partials['dy_dt', 't'] = self.eye \
            * (-(2*np.pi)**2 * np.sin(two_pi_t) - 2*np.pi*np.cos(two_pi_t))


num = 5

scheme_name = 'RK4'
scheme_name = 'backward Euler'

# integrator_name = 'SAND'
integrator_name = 'MDF'
# integrator_name = 'TM'

ode_function = ODEFunction()
ode_function.set_system(Comp)
ode_function.declare_state('y', rate_target='dy_dt', state_targets='y')
ode_function.declare_time('t')

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