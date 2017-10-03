from __future__ import division

import numpy as np
from scipy import sparse, linalg

from openmdao.api import ExplicitComponent


class TwoDOrbitSystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', default=1, type_=int)

    def setup(self):
        num = self.metadata['num_nodes']

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

    def compute_partials(self, inputs, partials):
        x, y = inputs['position'][..., 0], inputs['position'][..., 1]
        scale = (x**2 + y**2) ** (5/2)
        jac = 1/scale * np.array([[2*x**2 - y**2, 3*x*y],
                                  [3*x*y, 2*y**2 - x**2]])
        num = self.metadata['num_nodes']
        partials['dvel_dt', 'position'] = linalg.block_diag(*(jac[..., i] for i in range(num)))
