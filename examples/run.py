import numpy as np

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp

from openode.api import ODEFunction, ODEIntegrator

class Comp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=(num, 1))
        self.add_output('dy_dt', shape=(num, 1))

        # self.declare_partials('dy_dt', 'y', val=np.eye(num))
        self.declare_partials('dy_dt', 't', rows=np.arange(num), cols=np.arange(num))
        self.declare_partials('dy_dt', 'y', rows=np.arange(num), cols=np.arange(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        # True solution: 2 / (2*C1 - x^2)
        outputs['dy_dt'] = inputs['t'] * np.square(inputs['y'])

    def compute_partials(self, inputs, outputs, partials):
        partials['dy_dt', 'y'] = (2*inputs['t']*inputs['y']).squeeze()
        partials['dy_dt', 't'] = np.square(inputs['y']).squeeze()

ode_function = ODEFunction()
ode_function.set_system(Comp)
ode_function.declare_state('y', rate_target='dy_dt', state_targets='y')
ode_function.declare_time('t')

nums = [11, 16, 21, 26]
# nums = [5]
# nums = [11, 21, 31, 51]

# scheme_name = 'ForwardEuler'
scheme_name = 'RK4'
# scheme_name = 'ImplicitMidpoint'
# scheme_name = 'GaussLegendre4'

# integrator_name = 'SAND'
# integrator_name = 'MDF'
integrator_name = 'TM'

C1 = -1e-2
t1 = 1

if C1 > 0.:
    assert 2*C1 > t1**2
else:
    assert C1 != 0.

errs = np.zeros(len(nums))
for i, num in enumerate(nums):
    integrator = ODEIntegrator(ode_function, integrator_name, scheme_name,
        times=np.linspace(0., t1, num), initial_conditions={'y': 1./C1},)

    prob = Problem(integrator)

    if integrator_name == 'SAND':
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        integrator.add_subsystem('dummy_comp', IndepVarComp('dummy_var', val=1.0))
        integrator.add_objective('dummy_comp.dummy_var')

        prob.setup()
        prob.run_driver()
    else:
        prob.setup()
        # prob.check_partials(compact_print=True)
        prob.run_model()


    errs[i] = np.abs(prob['state:y'][-1][0] - (2 / (2*C1 - t1**2)))
    # print(prob['state:y'])


print('-'*40)
print('| {:10s} | {:10s} | {:10s} |'.format('h', 'Error', 'Rate'))
print('-'*40)
for i, (n, err) in enumerate(zip(nums, errs)):
    print('| {:.4e} | {:.4e} | {:.4e} |'.format(1./(n-1), err, 0. if i == 0 else
        np.log(errs[i] / errs[i-1]) / np.log((1./(n-1)) / (1./(nums[i-1]-1)))))
