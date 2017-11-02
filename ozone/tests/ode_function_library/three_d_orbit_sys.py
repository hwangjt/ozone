import numpy as np

from openmdao.api import ExplicitComponent


class ThreeDOrbitSystem(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', default=1, types=int)
        self.metadata.declare('r_scal', default=1e12, types=(int, float))
        self.metadata.declare('v_scal', default=1e3, types=(int, float))

    def setup(self):
        num = self.metadata['num_nodes']
        r_scal = self.metadata['r_scal']
        v_scal = self.metadata['v_scal']

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
        num = self.metadata['num_nodes']
        r_scal = self.metadata['r_scal']
        v_scal = self.metadata['v_scal']

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
        num = self.metadata['num_nodes']
        r_scal = self.metadata['r_scal']
        v_scal = self.metadata['v_scal']

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
