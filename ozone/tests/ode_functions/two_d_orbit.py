import numpy as np
from scipy import sparse, linalg

from openmdao.api import ExplicitComponent

from ozone.api import ODEFunction

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
        self.declare_partials('dpos_dt', 'velocity', val=sparse.block_diag([np.eye(2) for _ in range(num)]))
        self.declare_partials('dvel_dt', 'position', dependent=True)

    def compute(self, inputs, outputs):
        outputs['dpos_dt'] = inputs['velocity']
        attraction = np.linalg.norm(inputs['position'], axis=-1) ** 3
        outputs['dvel_dt'] = -inputs['position'] / attraction[..., None]

    def compute_partials(self, inputs, outputs, partials):
        x, y = inputs['position'][..., 0], inputs['position'][..., 1]
        scale = (x**2 + y**2) ** (5/2)
        jac = 1/scale * np.array([[2*x**2 - y**2, 3*x*y],
                                  [3*x*y, 2*y**2 - x**2]])
        num = self.metadata['num']
        partials['dvel_dt', 'position'] = linalg.block_diag(*(jac[..., i] for i in range(num)))

class TwoDOrbitFunction(ODEFunction):

    def initialize(self):
        self.set_system(TwoDOrbit)
        self.declare_state('position', rate_path='dpos_dt', paths='position', shape=2)
        self.declare_state('velocity', rate_path='dvel_dt', paths='velocity', shape=2)
        self.declare_time('t')

    def compute_exact_soln(self, initial_conditions, t0, t):
        # True solution: 2 / (2*C - t^2)
        # outputs['dy_dt'] = inputs['t'] * np.square(inputs['y'])

        ecc = 1 - initial_conditions['position'][0]
        return np.array([-1 - ecc, 0])
