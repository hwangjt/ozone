import numpy as np

from openmdao.api import ExplicitComponent


class GettingStartedOCSystem(ExplicitComponent):

    def initialize(self):
        # We declare a parameter for the class called num,
        # which is necessary to vectorize our ODE function.
        # All states, state rates, and dynamic parameters
        # must be of shape[num,...].
        self.metadata.declare('num_nodes', default=1, type_=int)

        # We make the acceleration due to gravity a parameter for illustration.
        self.metadata.declare('g', default=1., type_=(int, float))

    def setup(self):
        num = self.metadata['num_nodes']
        g = self.metadata['g']

        # Our dynamics depend on theta, x, y, and v.
        # They are all of scalars, so the overall shape is (num, 1).
        self.add_input('theta', shape=(num, 1))
        self.add_input('x', shape=(num, 1))
        self.add_input('y', shape=(num, 1))
        self.add_input('v', shape=(num, 1))

        # Our state variables are x, y, v, so we define rates for each.
        self.add_output('dx_dt', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))
        self.add_output('dv_dt', shape=(num, 1))

        # OpenMDAO assumes all outputs depend on inputs by default, so we first turn them off.
        self.declare_partials('*', '*', dependent=False)

        # dx_dt, dy_dt, and dv_dt are nonlinear in v and theta, so we only define
        # the sparsity structures of the Jacobians and not their non-zero values.
        self.declare_partials('dx_dt', 'v', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'v', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dx_dt', 'theta', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'theta', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dv_dt', 'theta', rows=np.arange(num), cols=np.arange(num))

    def compute(self, inputs, outputs):
        g = self.metadata['g']

        # This component computes dy_dt = -y.
        outputs['dx_dt'] = inputs['v'] * np.sin(inputs['theta'])
        outputs['dy_dt'] = inputs['v'] * np.cos(inputs['theta'])
        outputs['dv_dt'] = g * np.cos(inputs['theta'])

    def compute_partials(self, inputs, partials):
        g = self.metadata['g']

        theta = inputs['theta'][:, 0]
        v = inputs['v'][:, 0]

        # Earlier, we provided the structures of Jacobians; now we specify their values.
        partials['dx_dt', 'v'] = np.sin(theta)
        partials['dy_dt', 'v'] = np.cos(theta)
        partials['dx_dt', 'theta'] =  v * np.cos(theta)
        partials['dy_dt', 'theta'] = -v * np.sin(theta)
        partials['dv_dt', 'theta'] = -g * np.sin(theta)
