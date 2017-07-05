import numpy as np

from openmdao.api import ExplicitComponent, Problem

from openode.api import ODE, ExplicitTimeMarchingIntegrator, ExplicitRelaxedIntegrator


class Comp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))
        self.declare_partials('dy_dt', 'y', val=np.eye(num))

    def compute(self, inputs, outputs):
        outputs['dy_dt'] = inputs['y']


num = 20

ode = ODE(Comp)
ode.declare_state('y', rate_target='dy_dt', state_targets='y')

# intgr = ExplicitRelaxedIntegrator(
intgr = ExplicitTimeMarchingIntegrator(
    ode=ode, time_spacing=np.arange(num),
    scheme='kutta_third_order', initial_conditions={'y': 1.}, start_time=0., end_time=1.)

prob = Problem(intgr)
prob.setup()
prob.run_model()
# prob.check_partials(compact_print=True)

print(prob['output_comp.y'])

from openmdao.api import view_model

# view_model(prob)
