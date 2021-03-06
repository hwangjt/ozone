import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ExplicitComponent

from ozone.utils.var_names import get_name
from ozone.utils.units import get_rate_units


class VectorizedStep2Comp(ExplicitComponent):

    def initialize(self):
        self.options.declare('states', types=dict)
        self.options.declare('time_units', types=str, allow_none=True)
        self.options.declare('num_times', types=int)
        self.options.declare('num_stages', types=int)
        self.options.declare('num_step_vars', types=int)
        self.options.declare('glm_B', types=np.ndarray)
        self.options.declare('glm_V', types=np.ndarray)

    def setup(self):
        time_units = self.options['time_units']
        num_times = self.options['num_times']
        num_stages = self.options['num_stages']
        num_step_vars = self.options['num_step_vars']
        glm_B = self.options['glm_B']
        glm_V = self.options['glm_V']

        self.mtx_lu_dict = {}
        self.mtx_y0_dict = {}
        self.mtx_h_dict = {}
        self.mtx_hf_dict = {}

        self.num_y0_dict = {}
        self.num_F_dict = {}
        self.num_y_dict = {}

        h_arange = np.arange(num_times - 1)
        num_h = num_times - 1

        self.add_input('h_vec', shape=(num_times - 1), units=time_units)

        for state_name, state in iteritems(self.options['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            y_name = get_name('y', state_name)

            # --------------------------------------------------------------------------------

            y0_arange = np.arange(num_step_vars * size).reshape((num_step_vars,) + shape)

            F_arange = np.arange((num_times - 1) * num_stages * size).reshape(
                (num_times - 1, num_stages,) + shape)

            y_arange = np.arange(num_times * num_step_vars * size).reshape(
                (num_times, num_step_vars,) + shape)

            num_y0 = np.prod(y0_arange.shape)
            num_F = np.prod(F_arange.shape)
            num_y = np.prod(y_arange.shape)

            # --------------------------------------------------------------------------------

            self.add_input(y0_name,
                shape=(num_step_vars,) + shape,
                units=state['units'])

            self.add_input(F_name,
                shape=(num_times - 1, num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_output(y_name,
                shape=(num_times, num_step_vars,) + shape,
                units=state['units'])

            # --------------------------------------------------------------------------------

            data_list = []
            rows_list = []
            cols_list = []

            # y identity
            data = np.ones(num_y)
            rows = np.arange(num_y)
            cols = np.arange(num_y)
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # V blocks: (num_times - 1) x num_step_var x num_step_var x ...
            data = np.einsum('jk,i...->ijk...',
                -glm_V, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...',
                y_arange[1:, :, :], np.ones(num_step_vars, int)).flatten()
            cols = np.einsum('ik...,j->ijk...',
                y_arange[:-1, :, :], np.ones(num_step_vars, int)).flatten()
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # concatenate
            data = np.concatenate(data_list)
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)

            mtx = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_y, num_y))

            self.mtx_lu_dict[state_name] = scipy.sparse.linalg.splu(mtx)

            # --------------------------------------------------------------------------------

            data = np.ones(num_y0)
            rows = y_arange[0, :, :].flatten()
            cols = np.arange(num_y0)
            self.mtx_y0_dict[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_y, num_y0))

            # --------------------------------------------------------------------------------

            data = np.ones(num_F)
            rows = np.arange(num_F)
            cols = np.einsum('i,j...->ij...',
                h_arange, np.ones((num_stages,) + shape, int)).flatten()
            self.mtx_h_dict[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_F, num_h))

            # --------------------------------------------------------------------------------

            # B blocks: (num_times - 1) x num_step_vars x num_stage x ...
            data = np.einsum('jk,i...->ijk...',
                glm_B, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...',
                y_arange[1:, :, :], np.ones(num_stages, int)).flatten()
            cols = np.einsum('ik...,j->ijk...',
                F_arange, np.ones(num_step_vars, int)).flatten()

            self.mtx_hf_dict[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_y, num_F))

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        num_stages = self.options['num_stages']
        num_step_vars = self.options['num_step_vars']

        for state_name, state in iteritems(self.options['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            y_name = get_name('y', state_name)

            mtx_lu = self.mtx_lu_dict[state_name]
            mtx_y0 = self.mtx_y0_dict[state_name]
            mtx_h = self.mtx_h_dict[state_name]
            mtx_hf = self.mtx_hf_dict[state_name]

            # ------------------------------------------------------------------------------

            vec = mtx_h.dot(inputs['h_vec']) * inputs[F_name].flatten()
            vec = mtx_hf.dot(vec)
            vec = mtx_lu.solve(vec)

            outputs[y_name] = vec.reshape((num_times, num_step_vars,) + shape)

            outputs[y_name][0, :, :] -= inputs[y0_name]

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        num_times = self.options['num_times']
        num_stages = self.options['num_stages']
        num_step_vars = self.options['num_step_vars']

        for state_name, state in iteritems(self.options['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            y_name = get_name('y', state_name)

            mtx_lu = self.mtx_lu_dict[state_name]
            mtx_y0 = self.mtx_y0_dict[state_name]
            mtx_h = self.mtx_h_dict[state_name]
            mtx_hf = self.mtx_hf_dict[state_name]

            # ------------------------------------------------------------------------------

            if mode == 'fwd':
                if y_name in d_outputs:
                    if y0_name in d_inputs:
                        d_outputs[y_name][0, :, :] -= d_inputs[y0_name]

                    if F_name in d_inputs:
                        vec = mtx_h.dot(inputs['h_vec']) * d_inputs[F_name].flatten()
                        vec = mtx_hf.dot(vec)
                        vec = mtx_lu.solve(vec)

                        d_outputs[y_name] += vec.reshape(
                            (num_times, num_step_vars,) + shape)

                    if 'h_vec' in d_inputs:
                        vec = mtx_h.dot(d_inputs['h_vec']) * inputs[F_name].flatten()
                        vec = mtx_hf.dot(vec)
                        vec = mtx_lu.solve(vec)

                        d_outputs[y_name] += vec.reshape(
                            (num_times, num_step_vars,) + shape)

            # ------------------------------------------------------------------------------

            elif mode == 'rev':
                if y_name in d_outputs:
                    if y0_name in d_inputs:
                        d_inputs[y0_name] -= d_outputs[y_name][0, :, :]

                    vec = d_outputs[y_name].flatten()
                    vec = mtx_lu.solve(vec, 'T')
                    vec = mtx_hf.T.dot(vec)

                    if F_name in d_inputs:
                        d_inputs[F_name] += (mtx_h.dot(inputs['h_vec']) * vec).reshape(
                            (num_times - 1, num_stages,) + shape)

                    if 'h_vec' in d_inputs:
                        d_inputs['h_vec'] += mtx_h.T.dot(vec * inputs[F_name].flatten())
