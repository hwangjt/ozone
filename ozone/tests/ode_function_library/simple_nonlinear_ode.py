import numpy as np

from openmdao.api import ExplicitComponent

from ozone.api import ODEFunction


class SimpleNonlinearODESystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        # self.declare_partials('dy_dt', 'y', val=np.eye(num))
        self.declare_partials('dy_dt', 't', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'y', rows=np.arange(num), cols=np.arange(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: 2 / (2*C1 - x^2)
        outputs['dy_dt'][:, 0] = inputs['t'] * np.square(inputs['y'][:, 0])

    def compute_partials(self, inputs, partials):
        partials['dy_dt', 'y'] = (2*inputs['t']*inputs['y'][:, 0]).squeeze()
        partials['dy_dt', 't'] = np.square(inputs['y'][:, 0]).squeeze()


class SimpleNonlinearODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(SimpleNonlinearODESystem)
        self.declare_state('y', rate_path='dy_dt', paths='y')
        self.declare_time('t')

    def get_default_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        state_names = ['y']
        return t0, t1, initial_conditions, state_names

    def get_exact_solution(self, initial_conditions, t0, t):
        # True solution: 2 / (2*C - t^2)
        # outputs['dy_dt'] = inputs['t'] * np.square(inputs['y'])

        y0 = initial_conditions['y']
        C = (2. / y0 + t0 ** 2) / 2.
        return {'y': 2. / (2. * C - t ** 2)}
