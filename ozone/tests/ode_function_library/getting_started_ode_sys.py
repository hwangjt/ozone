import numpy as np

from openmdao.api import ExplicitComponent


class GettingStartedODESystem(ExplicitComponent):

    def initialize(self):
        # We declare a parameter for the class called num,
        # which is necessary to vectorize our ODE function.
        # All states, state rates, and dynamic parameters
        # must be of shape[num,...].
        self.metadata.declare('num_nodes', default=1, type_=int)

    def setup(self):
        num = self.metadata['num_nodes']

        # Our 'f' depends only on y, which is a scalar, so y's shape is (num, 1).
        self.add_input('y', shape=(num, 1))

        # dy_dt is the output of 'f'. dy_dt is also a scalar, so its shape is also (num, 1).
        self.add_output('dy_dt', shape=(num, 1))

        # The derivative of dy_dt with respect to y is constant, so we specify it here.
        # The Jacobian is diagonal, because each entry of dy_dt depends on the
        # corresponding entry of y, with a value of -1.
        self.declare_partials('dy_dt', 'y', val=-1., rows=np.arange(num), cols=np.arange(num))

    def compute(self, inputs, outputs):
        # This component computes dy_dt = -y.
        outputs['dy_dt'] = -inputs['y']
