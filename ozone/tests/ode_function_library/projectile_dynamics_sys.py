import numpy as np

from openmdao.api import ExplicitComponent


class ProjectileSystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

        self.g = -9.81

    def setup(self):
        num = self.metadata['num']

        self.add_input('vx', shape=(num, 1))
        self.add_input('vy', shape=(num, 1))

        self.add_output('dx_dt', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))
        self.add_output('dvx_dt', shape=(num, 1))
        self.add_output('dvy_dt', shape=(num, 1))

        self.declare_partials('*', '*', dependent=False)

        self.declare_partials('dx_dt', 'vx', val=1., rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'vy', val=1., rows=np.arange(num), cols=np.arange(num))

    def compute(self, inputs, outputs):
        outputs['dx_dt'] = inputs['vx']
        outputs['dy_dt'] = inputs['vy']
        outputs['dvx_dt'] = 0.
        outputs['dvy_dt'] = self.g