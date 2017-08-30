import numpy as np
import time

from openmdao.api import ExplicitComponent, Problem, ScipyOptimizer, IndepVarComp, view_model, ExecComp, pyOptSparseDriver, DefaultMultiVector

from ozone.api import ODEFunction, ODEIntegrator


r_scal = 1e12
v_scal = 1e3


class MyComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num', default=1, type_=int)

    def setup(self):
        num = self.metadata['num']

        g_m_s2 = 9.80665 # m/s^2
        Isp_s = 2000 # s
        c_m_s = Isp_s * g_m_s2 # m/s

        u_m3_s2 = 132712440018 * 1e9 # m^3/s^2
        Tmax_N = 0.5 # N

        self.c_m_s = c_m_s
        self.u_m3_s2 = u_m3_s2
        self.Tmax_N = Tmax_N

        self.add_input('d', shape=(num, 1))
        self.add_input('a', val=0.5, shape=(num, 1))
        self.add_input('b', val=0.5, shape=(num, 1))

        self.add_input('r', shape=(num, 3))
        self.add_input('v', shape=(num, 3))
        self.add_input('m', shape=(num, 1))

        self.add_output('r_dot', shape=(num, 3))
        self.add_output('v_dot', shape=(num, 3))
        self.add_output('m_dot', shape=(num, 1))

        self.declare_partials('*', '*', dependent=False)

        data = np.ones(3 * num).reshape((num, 3)) * v_scal / r_scal
        arange = np.arange(3 * num).reshape((num, 3))
        self.declare_partials('r_dot', 'v',
            val=data.flatten(), rows=arange.flatten(), cols=arange.flatten())

        arange = np.arange(3 * num).reshape((num, 3))
        rows = np.einsum('ij,k->ijk', arange, np.ones(3, int))
        cols = np.einsum('ik,j->ijk', arange, np.ones(3, int))
        self.declare_partials('v_dot', 'r', rows=rows.flatten(), cols=cols.flatten())

        rows = np.arange(3 * num).reshape((num, 3))
        cols = np.einsum('i,j->ij', np.arange(num), np.ones(3, int))
        self.declare_partials('v_dot', 'd', rows=rows.flatten(), cols=cols.flatten())
        self.declare_partials('v_dot', 'a', rows=rows.flatten(), cols=cols.flatten())
        self.declare_partials('v_dot', 'b', rows=rows.flatten(), cols=cols.flatten())
        self.declare_partials('v_dot', 'm', rows=rows.flatten(), cols=cols.flatten())

        rows = np.arange(num)
        cols = np.arange(num)
        self.declare_partials('m_dot', 'd', val=-Tmax_N / c_m_s, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num = self.metadata['num']

        c_m_s = self.c_m_s
        u_m3_s2 = self.u_m3_s2
        Tmax_N = self.Tmax_N

        r = inputs['r'] * r_scal
        v = inputs['v'] * v_scal
        m = inputs['m'][:, 0]

        d = inputs['d'][:, 0]
        a = inputs['a'][:, 0]
        b = inputs['b'][:, 0]

        r_norm = np.linalg.norm(r, axis=1)
        r_norm = np.sum(r ** 2, axis=1) ** 0.5

        # km / s
        outputs['r_dot'] = v / r_scal

        # km / s^2
        outputs['v_dot'][:, 0] = \
            (-u_m3_s2 / r_norm ** 3 * r[:, 0] + d * Tmax_N / m * np.sin(a) * np.cos(b)) / v_scal
        outputs['v_dot'][:, 1] = \
            (-u_m3_s2 / r_norm ** 3 * r[:, 1] + d * Tmax_N / m * np.sin(a) * np.sin(b)) / v_scal
        outputs['v_dot'][:, 2] = \
            (-u_m3_s2 / r_norm ** 3 * r[:, 2] + d * Tmax_N / m * np.cos(a)) / v_scal

        # kg / s
        outputs['m_dot'][:, 0] = -Tmax_N / c_m_s * d

    def compute_partials(self, inputs, partials):
        num = self.metadata['num']

        u_m3_s2 = self.u_m3_s2
        Tmax_N = self.Tmax_N

        r = inputs['r'] * r_scal
        v = inputs['v'] * v_scal
        m = inputs['m'][:, 0]

        d = inputs['d'][:, 0]
        a = inputs['a'][:, 0]
        b = inputs['b'][:, 0]

        r_norm = np.linalg.norm(r, axis=1)
        r_norm = np.sum(r ** 2, axis=1) ** 0.5

        # outputs['v_dot'][:, 0] = -u / r_norm ** 3 * r[:, 0] + d * Tmax / m * np.sin(a) * np.cos(b)
        # outputs['v_dot'][:, 1] = -u / r_norm ** 3 * r[:, 1] + d * Tmax / m * np.sin(a) * np.sin(b)
        # outputs['v_dot'][:, 2] = -u / r_norm ** 3 * r[:, 2] + d * Tmax / m * np.cos(a)

        # func:  -u * r2 ^ (-3/2) r
        # deriv: 3 u * r2 ^ (-5/2) r x r
        sub_jac = partials['v_dot', 'r'].reshape((num, 3, 3))
        for k in range(3):
            sub_jac[:, k, :] = np.einsum('i,ij->ij', 3 * u_m3_s2 / r_norm ** 5 * r[:, k], r) / v_scal * r_scal
            sub_jac[:, k, k] -= u_m3_s2 / r_norm ** 3 / v_scal * r_scal

        sub_jac = partials['v_dot', 'd'].reshape((num, 3))
        sub_jac[:, 0] = Tmax_N / m * np.sin(a) * np.cos(b) / v_scal
        sub_jac[:, 1] = Tmax_N / m * np.sin(a) * np.sin(b) / v_scal
        sub_jac[:, 2] = Tmax_N / m * np.cos(a) / v_scal

        sub_jac = partials['v_dot', 'a'].reshape((num, 3))
        sub_jac[:, 0] = d * Tmax_N / m * np.cos(a) * np.cos(b) / v_scal
        sub_jac[:, 1] = d * Tmax_N / m * np.cos(a) * np.sin(b) / v_scal
        sub_jac[:, 2] = -d * Tmax_N / m * np.sin(a) / v_scal

        sub_jac = partials['v_dot', 'b'].reshape((num, 3))
        sub_jac[:, 0] = -d * Tmax_N / m * np.sin(a) * np.sin(b) / v_scal
        sub_jac[:, 1] = d * Tmax_N / m * np.sin(a) * np.cos(b) / v_scal
        sub_jac[:, 2] = 0.

        sub_jac = partials['v_dot', 'm'].reshape((num, 3))
        sub_jac[:, 0] = -d * Tmax_N / m ** 2 * np.sin(a) * np.cos(b) / v_scal
        sub_jac[:, 1] = -d * Tmax_N / m ** 2 * np.sin(a) * np.sin(b) / v_scal
        sub_jac[:, 2] = -d * Tmax_N / m ** 2 * np.cos(a) / v_scal
        t2 = time.time()


class MyODEFunction(ODEFunction):

    def initialize(self):
        self.set_system(MyComp)

        self.declare_state('r', 'r_dot', paths='r', shape=3)
        self.declare_state('v', 'v_dot', paths='v', shape=3)
        self.declare_state('m', 'm_dot', paths='m', shape=1)

        self.declare_dynamic_parameter('d', 'd', shape=1)
        self.declare_dynamic_parameter('a', 'a', shape=1)
        self.declare_dynamic_parameter('b', 'b', shape=1)


class ConstraintComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('rf', type_=np.ndarray, required=True)
        self.metadata.declare('vf', type_=np.ndarray, required=True)

    def setup(self):
        self.add_input('r', shape=3)
        self.add_input('v', shape=3)

        self.add_output('con_r', shape=3)
        self.add_output('con_v', shape=3)

        self.declare_partials('con_r', 'r', val=np.eye(3))
        self.declare_partials('con_v', 'v', val=np.eye(3))

    def compute(self, inputs, outputs):
        outputs['con_r'] = inputs['r'] - self.metadata['rf']
        outputs['con_v'] = inputs['v'] - self.metadata['vf']


if __name__ == '__main__':
    num = 100
    t0 = 0.
    t1 = 348.795 * 24 * 3600
    times = np.linspace(t0, t1, num)

    initial_conditions = {
        'r': np.array([ -140699693 , -51614428 , 980 ]) * 1e3 / r_scal,
        'v': np.array([ 9.774596 , -28.07828 , 4.337725e-4 ]) * 1e3 / v_scal,
        'm': 1000.
    }

    final_conditions = {
        'r': np.array([ -172682023 , 176959469 , 7948912 ]) * 1e3 / r_scal,
        'v': np.array([ -16.427384 , -14.860506 , 9.21486e-2 ]) * 1e3 / v_scal,
    }

    ode_function = MyODEFunction()

    # scheme_name = 'ForwardEuler'
    scheme_name = 'BackwardEuler'
    scheme_name = 'RK4'
    scheme_name = 'ImplicitMidpoint'
    # scheme_name = 'ExplicitMidpoint'
    # scheme_name = 'AM4'
    # scheme_name = 'GaussLegendre4'
    # scheme_name = 'BDF2'

    integrator_name = 'SAND'
    integrator_name = 'MDF'
    # integrator_name = 'TM'

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('d', val=0.5, shape=(num, 1))
    comp.add_output('a', shape=(num, 1))
    comp.add_output('b', shape=(num, 1))
    comp.add_design_var('d', lower=0., upper=1.)
    comp.add_design_var('a') #, lower=0., upper=np.pi)
    comp.add_design_var('b') #, lower=0., upper=2 * np.pi)
    prob.model.add_subsystem('inputs', comp, promotes=['*'])

    group = ODEIntegrator(ode_function, integrator_name, scheme_name,
        times=times, initial_conditions=initial_conditions)
    group.add_constraint('state:r', indices=[num-1], )
    prob.model.add_subsystem('integrator', group, promotes=['*'])
    prob.model.connect('d', 'dynamic_parameter:d')
    prob.model.connect('a', 'dynamic_parameter:a')
    prob.model.connect('b', 'dynamic_parameter:b')

    if 1:
        comp = ExecComp('f = -mass')
        comp.add_objective('f')
        prob.model.add_subsystem('objective_comp', comp)
        prob.model.connect('state:m', 'objective_comp.mass', src_indices=[num - 1])

        comp = ConstraintComp(rf=final_conditions['r'], vf=final_conditions['v'])
        comp.add_constraint('con_r', equals=0.)
        comp.add_constraint('con_v', equals=0.)
        prob.model.add_subsystem('constraints_comp', comp)
        prob.model.connect('state:r', 'constraints_comp.r', src_indices=np.arange(3*num - 3, 3*num))
        prob.model.connect('state:v', 'constraints_comp.v', src_indices=np.arange(3*num - 3, 3*num))
    else:
        group.add_subsystem('dummy_comp', IndepVarComp('dummy_var'))
        group.add_objective('dummy_comp.dummy_var')

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major optimality tolerance'] = 2e-6
    prob.driver.opt_settings['Major feasibility tolerance'] = 2e-6
    prob.driver.opt_settings['Verify level'] = -1

    # prob.setup(multi_vector_class=DefaultMultiVector)
    prob.setup()

    print('done setup')
    prob.run_model()

    t1 = time.time()
    prob.run_model()
    t2 = time.time()
    prob.compute_total_derivs(
        of=['objective_comp.f', 'constraints_comp.con_r', 'constraints_comp.con_v',
            # 'integration_group.vectorized_stage_comp.Y_out:r',
            # 'integration_group.vectorized_stage_comp.Y_out:v',
        ],
        wrt=['d', 'a', 'b'],
    )
    t3 = time.time()
    print(t2-t1, t3-t2)

    # prob.check_partials(compact_print=True)
    # view_model(prob)
    # exit()

    prob.run_driver()

    import matplotlib.pyplot as plt

    print()
    print(initial_conditions['r'])
    print(initial_conditions['v'])
    print()
    print(prob['state:r'][0, :])
    print(prob['state:v'][0, :])
    print(prob['state:m'][0, 0])
    print()
    print(final_conditions['r'])
    print(final_conditions['v'])
    print()
    print(prob['state:r'][-1, :])
    print(prob['state:v'][-1, :])
    print(prob['state:m'][-1, 0])

    au = 149597870.7 * 1e3 / r_scal
    r = prob['state:r']
    plt.subplot(2,1,1)
    plt.plot(r[:, 0] / au, r[:, 1] / au, '-o')
    plt.subplot(2,1,2)
    plt.plot(prob['times'], prob['d'], '-o')
    plt.show()
