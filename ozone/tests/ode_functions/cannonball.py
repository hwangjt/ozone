import numpy as np

from openmdao.api import ExplicitComponent

from ozone.api import ODEFunction


class CannonballSystem(ExplicitComponent):
    """
    Equations of motion for the position and velocity of a point mass moving in
    a rectilinear gravity field.
    """
    def initialize(self):
        self.metadata.declare('num', type_=int)

    def setup(self):
        nn = self.metadata['num']
        # Inputs
        self.add_input('vx', shape=(nn, 1), desc='horizontal component of velocity', units='m/s')
        self.add_input('vy', shape=(nn, 1), desc='vertical component of velocity', units='m/s')

        # Params are non-state inputs to the EOM that impact results (dynamic/static controls, etc)
        self.add_input('g', val=9.80665*np.ones((nn, 1)),
                       desc='gravitational acceleration', units='m/s/s')

        # Auxiliary calculations
        self.add_output('gam', val=np.zeros((nn, 1)), desc='Flight path angle', units='rad')
        self.add_output('vel', val=np.zeros((nn, 1)), desc='velocity magnitude', units='m/s')

        self.add_output('xdot', val=np.zeros((nn, 1)), desc='x-velocity magnitude', units='m/s')
        self.add_output('ydot', val=np.zeros((nn, 1)), desc='y-velocity magnitude', units='m/s')
        self.add_output('vxdot', val=np.zeros((nn, 1)), desc='x-acceleration magnitude', units='m/s**2')
        self.add_output('vydot', val=np.zeros((nn, 1)), desc='y-acceleration magnitude', units='m/s**2')

        # Setup partials
        arange = np.arange(self.metadata['num'])
        self.declare_partials(of='*', wrt='*', dependent=False)

        self.declare_partials(of='xdot', wrt='vx', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='ydot', wrt='vy', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='vydot', wrt='g', rows=arange, cols=arange, val=-1.0)

        self.declare_partials(of='gam', wrt='vy', rows=arange, cols=arange)
        self.declare_partials(of='gam', wrt='vx', rows=arange, cols=arange)

        self.declare_partials(of='vel', wrt='vy', rows=arange, cols=arange)
        self.declare_partials(of='vel', wrt='vx', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['xdot'] = inputs['vx']
        outputs['ydot'] = inputs['vy']
        outputs['vxdot'] = 0.0
        outputs['vydot'] = -inputs['g']

        outputs['gam'] = np.arctan2(inputs['vy'], inputs['vx'])
        outputs['vel'] = np.sqrt(inputs['vy']**2 + inputs['vx']**2)

    def compute_partials(self, inputs, jacobian):
        vmag2 = inputs['vx'][:, 0]**2+inputs['vy'][:, 0]**2
        jacobian['gam', 'vx'] = -inputs['vy'][:, 0] / vmag2
        jacobian['gam', 'vy'] = inputs['vx'][:, 0] / vmag2
        jacobian['vel', 'vx'] = inputs['vx'][:, 0] / np.sqrt(vmag2)
        jacobian['vel', 'vy'] = inputs['vy'][:, 0] / np.sqrt(vmag2)


class CannonballODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(CannonballSystem)
        self.declare_state('x', rate_path='xdot', paths=[], units='m')
        self.declare_state('y', rate_path='ydot', paths=[], units='m')
        self.declare_state('vx', rate_path='vxdot', paths=['vx'], units='m/s')
        self.declare_state('vy', rate_path='vydot', paths=['vy'], units='m/s')
        self.declare_time(units='s')

    def compute_exact_soln(self, initial_conditions, t0, t):
        # y' = -gt + a
        # y = -0.5 g t^2 + a t + b

        a = initial_conditions['vy'] + 9.80665 * t0
        b = initial_conditions['y'] + 0.5 * 9.80665 * t0 ** 2 - a * t0

        return {
            'x': initial_conditions['x'] + initial_conditions['vx'] * (t - t0),
            'y': -0.5 * 9.80665 * t ** 2 + a * t + b,
            'vx': initial_conditions['vx'],
            'vy': -9.80665 * t + a,
        }
