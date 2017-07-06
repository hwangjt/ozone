import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ImplicitComponent

from openode.utils.var_names import get_F_name, get_y_old_name, get_y_new_name
from openode.utils.units import get_rate_units


class VectorizedStepComp(ImplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('time_units', values=(None,), type_=str, required=True)
        self.metadata.declare('num_time_steps', type_=int, required=True)
        self.metadata.declare('num_stages', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('glm_B', type_=np.ndarray, required=True)
        self.metadata.declare('glm_V', type_=np.ndarray, required=True)

    def setup(self):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        self.add_input('h_vec', shape=num_time_steps, units=time_units)

        self.dy_dy = dy_dy = {}
        self.dy_dy_inv = dy_dy_inv = {}
        self.dy_dhF = dy_dhF = {}
        self.dy_dy0 = dy_dy0 = {}
        self.h_mtx = h_mtx = {}

        h_arange = np.arange(num_time_steps - 1)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            F_name = 'F:%s' % state_name
            y0_name = 'y0:%s' % state_name
            y_name = 'y:%s' % state_name

            y0_arange = np.arange(size * num_step_vars).reshape((size, num_step_vars))

            y_arange = np.arange(size * num_time_steps * num_step_vars).reshape(
                (size, num_time_steps, num_step_vars))

            F_arange = np.arange(size * (num_time_steps - 1) * num_stages).reshape(
                (size, num_time_steps - 1, num_stages))

            self.add_input(F_name,
                shape=(size, num_time_steps - 1, num_stages),
                units=get_rate_units(state['units'], time_units))

            self.add_input(y0_name,
                shape=(size, num_step_vars),
                units=state['units'])

            self.add_output(y_name,
                shape=(size, num_time_steps, num_step_vars),
                units=state['units'])

            # -----------------

            # AIJ(size, num_time_steps, num_step_vars)
            data1 = np.ones(size * num_time_steps * num_step_vars)
            rows1 = np.arange(size * num_time_steps * num_step_vars)
            cols1 = np.arange(size * num_time_steps * num_step_vars)

            # AIJ(size, num_time_steps - 1, num_step_vars, num_step_vars)
            data2 = np.einsum('ij,kl->ijkl', np.ones((size, num_time_steps - 1)), -glm_V).flatten()
            rows2 = np.einsum('ijk,l->ijkl', y_arange[:, 1:, :], np.ones(num_step_vars)).flatten()
            cols2 = np.einsum('ijl,k->ijkl', y_arange[:, :-1, :], np.ones(num_step_vars)).flatten()

            data = np.concatenate([data1, data2])
            rows = np.concatenate([rows1, rows2])
            cols = np.concatenate([cols1, cols2])

            dy_dy[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(
                    size * num_time_steps * num_step_vars,
                    size * num_time_steps * num_step_vars))

            dy_dy_inv[state_name] = scipy.sparse.linalg.splu(dy_dy[state_name])

            self.declare_partials(y_name, y_name, val=data, rows=rows, cols=cols)

            # -----------------

            # (size, num_step_vars)
            data = -np.ones((size, num_step_vars)).flatten()
            rows = y_arange[:, 0, :].flatten()
            cols = y0_arange.flatten()

            self.declare_partials(y_name, y0_name, val=data, rows=rows, cols=cols)

            # -----------------

            # AIJ(size, num_time_steps - 1, num_step_vars, num_stages)
            # h(num_time_steps - 1)
            # y(size, num_time_steps, num_step_vars)
            # F(size, num_time_steps - 1, num_stages)
            rows = np.einsum('ijk,l->ijkl', y_arange[:, 1:, :], np.ones(num_stages)).flatten()

            cols = np.einsum('ikl,j->ijkl',
                np.ones((size, num_step_vars, num_stages)), h_arange).flatten()
            self.declare_partials(y_name, 'h_vec', rows=rows, cols=cols)

            cols = np.einsum('ijl,k->ijkl', F_arange, np.ones(num_step_vars)).flatten()
            self.declare_partials(y_name, F_name, rows=rows, cols=cols)

    def apply_nonlinear(self, inputs, outputs, residuals):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        dy_dy = self.dy_dy
        dy_dhF = self.dy_dhF
        dy_dy0 = self.dy_dy0
        h_mtx = self.h_mtx

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            F_name = 'F:%s' % state_name
            y0_name = 'y0:%s' % state_name
            y_name = 'y:%s' % state_name

            # dy_dy term
            in_vec = outputs[y_name].reshape((size * num_time_steps * num_step_vars))
            out_vec = dy_dy[state_name].dot(in_vec).reshape((size, num_time_steps, num_step_vars))

            residuals[y_name] = out_vec # y term
            residuals[y_name][:, 0, :] -= inputs[y0_name] # y0 term
            residuals[y_name][:, 1:, :] -= np.einsum('kl,j,ijl->ijk',
                glm_B, inputs['h_vec'], inputs[F_name]) # hF term

    def solve_nonlinear(self, inputs, outputs):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        dy_dy = self.dy_dy
        dy_dy_inv = self.dy_dy_inv
        dy_dhF = self.dy_dhF
        dy_dy0 = self.dy_dy0
        h_mtx = self.h_mtx

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            F_name = 'F:%s' % state_name
            y0_name = 'y0:%s' % state_name
            y_name = 'y:%s' % state_name

            vec = np.zeros((size, num_time_steps, num_step_vars))
            vec[:, 0, :] += inputs[y0_name] # y0 term
            vec[:, 1:, :] += np.einsum('kl,j,ijl->ijk',
                glm_B, inputs['h_vec'], inputs[F_name]) # hF term

            outputs[y_name] = dy_dy_inv[state_name].solve(vec.flatten(), 'N').reshape(
                (size, num_time_steps, num_step_vars))

    def linearize(self, inputs, outputs, partials):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        dy_dy = self.dy_dy
        dy_dhF = self.dy_dhF
        dy_dy0 = self.dy_dy0
        h_mtx = self.h_mtx

        h_arange = np.arange(num_time_steps - 1)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            F_name = 'F:%s' % state_name
            y0_name = 'y0:%s' % state_name
            y_name = 'y:%s' % state_name

            # (size, num_time_steps - 1, num_step_vars, num_stages)
            partials[y_name, F_name] = np.einsum(
                'i,kl,j->ijkl', np.ones(size), glm_B, inputs['h_vec']).flatten()

            # (size, num_time_steps - 1, num_step_vars, num_stages)
            partials[y_name, 'h_vec'] = np.einsum(
                'kl,ijl->ijkl', glm_B, inputs[F_name]).flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        dy_dy_inv = self.dy_dy_inv
        num_time_steps = self.metadata['num_time_steps']
        num_step_vars = self.metadata['num_step_vars']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_name = 'y:%s' % state_name

            if mode == 'fwd':
                rhs_vec = d_residuals[y_name].flatten()
                solve_mode = 'N'
            elif mode == 'rev':
                rhs_vec = d_outputs[y_name].flatten()
                solve_mode = 'T'

            sol_vec = dy_dy_inv[state_name].solve(rhs_vec, solve_mode)

            if mode == 'fwd':
                d_outputs[y_name] = sol_vec.reshape((size, num_time_steps, num_step_vars))
            elif mode == 'rev':
                d_residuals[y_name] = sol_vec.reshape((size, num_time_steps, num_step_vars))
