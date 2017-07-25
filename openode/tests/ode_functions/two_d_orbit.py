import numpy as np
from scipy.sparse import block_diag

from openmdao.api import ExplicitComponent

from openode.api import ODEFunction

class TwoDOrbit(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('position', shape=(num, 2))
        self.add_input('velocity', shape=(num, 2))
        self.add_input('t', shape=(num, 1))
        self.add_output('dpos_dt', shape=(num, 2))
        self.add_output('dvel_dt', shape=(num, 2))

        # self.declare_partials('dy_dt', 'y', val=np.eye(num))
        self.declare_partials('*', '*', dependent=False)
        self.declare_partials('dpos_dt', 'velocity', val=block_diag([np.eye(2) for _ in range(num)]))
        self.declare_partials('dvel_dt', 'position', dependent=True)

    def compute(self, inputs, outputs):
        outputs['dpos_dt'] = inputs['velocity']
        attraction = np.linalg.norm(inputs['position']) ** 3
        outputs['dvel_dt'] = -inputs['position'] / attraction

    def compute_partials(self, inputs, outputs, partials):
        x, y = inputs['position'][..., 0], inputs['position'][..., 1]
        scale = (x**2 + y**2) ** (5/2)
        partials['dvel_dt', 'position'] = 1/scale * np.array([[2*x**2 - y**2, 3*x*y],
                                                              [3*x*y, 2*y**2 - x**2]])

class TwoDOrbitFunction(ODEFunction):

    def initialize(self):
        self.set_system(TwoDOrbit)
        self.declare_state('position', rate_target='dpos_dt', state_targets='position', shape=2)
        self.declare_state('velocity', rate_target='dvel_dt', state_targets='velocity', shape=2)
        self.declare_time('t')

    def compute_exact_soln(self, initial_conditions, t0, t):
        # True solution: 2 / (2*C - t^2)
        # outputs['dy_dt'] = inputs['t'] * np.square(inputs['y'])

        ecc = 1 - initial_conditions['position'][0]
        return np.array([-1 - ecc, 0])
