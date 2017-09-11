import numpy as np
import time

from openmdao.api import Problem, ScipyOptimizer, ExplicitComponent
from ozone.api import ODEIntegrator, ODEFunction


g = -9.81


class ProjectileSystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

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
        outputs['dvy_dt'] = -g


class ProjectileFunction(ODEFunction):

    def initialize(self, system_init_kwargs=None):
        self.set_system(ProjectileSystem, system_init_kwargs)

        self.declare_state('x', shape=1, rate_path='dx_dt')
        self.declare_state('y', shape=1, rate_path='dy_dt')
        self.declare_state('vx', shape=1, rate_path='dvx_dt', paths=['vx'])
        self.declare_state('vy', shape=1, rate_path='dvy_dt', paths=['vy'])

    def get_exact_solution(self, initial_conditions, t0, t):
        x0 = initial_conditions['x']
        y0 = initial_conditions['y']
        vx0 = initial_conditions['vx']
        vy0 = initial_conditions['vy']

        x = x0 + vx0 * (t - t0)
        y = y0 + vy0 * (t - t0) - 0.5 * g * (t - t0) ** 2
        vx = vx0
        vy = vy0 + g * (t - t0)
        return {'x': x, 'y': y, 'vx': vx, 'vy': vy}


def run_projectile(method_name, formulation, num):
    t0 = 0.
    t1 = 1.
    times = np.linspace(t0, t1, num)

    initial_conditions = {
        'x': 0.,
        'y': 0.,
        'vx': 1.,
        'vy': 1.,
    }

    ode_function = ProjectileFunction()

    integrator = ODEIntegrator(ode_function, formulation, method_name,
        times=times, initial_conditions=initial_conditions)

    prob = Problem(model=integrator)

    if formulation == 'optimizer-based':
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var'))
        integrator.add_objective('dummy_comp.dummy_var')

    prob.setup(check=False)
    run_time1 = time.time()
    prob.run_driver()
    run_time2 = time.time()

    final_y = prob['state:y'][-1, 0]
    exact_final_y = ode_function.get_exact_solution(initial_conditions, t0, t1)['y']

    error = abs(exact_final_y - final_y)
    run_time = run_time2 - run_time1

    return error, run_time
