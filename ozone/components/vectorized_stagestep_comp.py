import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ExplicitComponent

from ozone.utils.var_names import get_name
from ozone.utils.units import get_rate_units


class VectorizedStageStepComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('time_units', type_=(str, type(None)), required=True)
        self.metadata.declare('num_time_steps', type_=int, required=True)
        self.metadata.declare('num_stages', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('glm_A', type_=np.ndarray, required=True)
        self.metadata.declare('glm_U', type_=np.ndarray, required=True)
        self.metadata.declare('glm_B', type_=np.ndarray, required=True)
        self.metadata.declare('glm_V', type_=np.ndarray, required=True)

    def setup(self):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        self.mtx_lu_dict = {}
        self.mtx_y0_dict_dict = {}
        self.mtx_h_dict = {}
        self.mtx_hf_dict = {}

        self.num_y0_dict = {}
        self.num_F_dict = {}
        self.num_Y_dict = {}
        self.num_y_dict = {}

        h_arange = np.arange(num_time_steps - 1)
        num_h = num_time_steps - 1

        self.add_input('h_vec', shape=(num_time_steps - 1), units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            Y_name = get_name('Y', state_name)
            y_name = get_name('y', state_name)

            # --------------------------------------------------------------------------------

            y0_arange = np.arange(num_step_vars * size).reshape((num_step_vars,) + shape)

            F_arange = np.arange((num_time_steps - 1) * num_stages * size).reshape(
                (num_time_steps - 1, num_stages,) + shape)

            Y_arange = np.arange((num_time_steps - 1) * num_stages * size).reshape(
                (num_time_steps - 1, num_stages,) + shape)

            y_arange = np.arange(num_time_steps * num_step_vars * size).reshape(
                (num_time_steps, num_step_vars,) + shape)

            self.num_y0_dict[state_name] = num_y0 = np.prod(y0_arange.shape)
            self.num_F_dict[state_name] = num_F = np.prod(F_arange.shape)
            self.num_Y_dict[state_name] = num_Y = np.prod(Y_arange.shape)
            self.num_y_dict[state_name] = num_y = np.prod(y_arange.shape)

            # --------------------------------------------------------------------------------

            self.add_input(y0_name,
                shape=(num_step_vars,) + shape,
                units=state['units'])

            self.add_input(F_name,
                shape=(num_time_steps - 1, num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_output(y_name,
                shape=(num_time_steps, num_step_vars,) + shape,
                units=state['units'])

            self.add_output(Y_name,
                shape=(num_time_steps - 1, num_stages,) + shape,
                units=state['units'])

            # --------------------------------------------------------------------------------

            data_list = []
            rows_list = []
            cols_list = []

            # Y identity
            data = np.ones(num_Y)
            rows = np.arange(num_Y)
            cols = np.arange(num_Y)
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # y identity
            data = np.ones(num_y)
            rows = np.arange(num_y) + num_Y
            cols = np.arange(num_y) + num_Y
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # U blocks: (num_time_steps - 1) x num_stage x num_step_var x ...
            data = np.einsum('jk,i...->ijk...',
                -glm_U, np.ones((num_time_steps - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...',
                Y_arange, np.ones(num_step_vars, int)).flatten()
            cols = np.einsum('ik...,j->ijk...',
                y_arange[:-1, :, :], np.ones(num_stages, int)).flatten()
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # V blocks: (num_time_steps - 1) x num_step_var x num_step_var x ...
            data = np.einsum('jk,i...->ijk...',
                -glm_V, np.ones((num_time_steps - 1,) + shape)).flatten()
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
                shape=(num_Y + num_y, num_Y + num_y))

            print(mtx.todense())
            print(num_time_steps, num_step_vars, num_stages, size)

            self.mtx_lu_dict[state_name] = scipy.sparse.linalg.splu(mtx)

            # --------------------------------------------------------------------------------

            data = np.ones(num_y0)
            rows = num_Y + y_arange[0, :, :].flatten()
            cols = np.arange(num_y0)
            self.mtx_y0_dict[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_Y + num_y, num_y0))

            # --------------------------------------------------------------------------------

            data = np.ones(num_F)
            rows = np.arange(num_F)
            cols = np.einsum('i,j...->ij...',
                h_arange, np.ones((num_stages,) + shape, int)).flatten()
            self.mtx_h_dict[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_F, num_h))

            # --------------------------------------------------------------------------------

            data_list = []
            rows_list = []
            cols_list = []

            # A blocks: (num_time_steps - 1) x num_stage x num_stage x ...
            data = np.einsum('jk,i...->ijk...',
                glm_A, np.ones((num_time_steps - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...',
                Y_arange, np.ones(num_stage, int)).flatten()
            cols = np.einsum('ik...,j->ijk...',
                F_arange, np.ones(num_stage, int)).flatten()
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # B blocks: (num_time_steps - 1) x num_step_vars x num_stage x ...
            data = np.einsum('jk,i...->ijk...',
                glm_B, np.ones((num_time_steps - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...',
                Y_arange, np.ones(num_stage, int)).flatten()
            cols = np.einsum('ik...,j->ijk...',
                F_arange, np.ones(num_step_vars, int)).flatten()
            data_list.append(data); rows_list.append(rows); cols_list.append(cols)

            # concatenate
            data = np.concatenate(data_list)
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)

            self.mtx_hf_dict[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(num_Y + num_y, num_F))

    def compute(self, inputs, outputs):
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            Y_name = get_name('Y', state_name)
            y_name = get_name('y', state_name)

            mtx_lu = self.mtx_lu_dict[state_name]
            mtx_y0 = self.mtx_y0_dict[state_name]
            mtx_h = self.mtx_h_dict[state_name]
            mtx_hf = self.mtx_hf_dict[state_name]

            num_Y = self.num_Y_dict[state_name]

            # ------------------------------------------------------------------------------

            vec = mtx_h.dot(inputs['h_vec']) * inputs[F_name].flatten()
            vec = mtx_hf.dot(vec)
            vec = mtx_lu.solve(vec)

            outputs[Y_name] = vec[:num_Y].reshape((num_time_steps - 1, num_stages,) + shape)
            outputs[y_name] = vec[num_Y:].reshape((num_time_steps, num_step_vars,) + shape)

            outputs[y_name][0, :, :] -= inputs[y0_name]

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        for state_name, state in iteritems(self.metadata['states']):
            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            Y_name = get_name('Y', state_name)
            y_name = get_name('y', state_name)

            mtx_lu = self.mtx_lu_dict[state_name]
            mtx_y0 = self.mtx_y0_dict[state_name]
            mtx_h = self.mtx_h_dict[state_name]
            mtx_hf = self.mtx_hf_dict[state_name]

            num_Y = self.num_Y_dict[state_name]

            # ------------------------------------------------------------------------------

            if mode == 'fwd':
                if y_name in d_outputs:
                    if y0_name in d_inputs:
                        d_outputs[y_name][0, :, :] -= d_inputs[y0_name]

                if F_name in d_inputs:
                    vec = mtx_h.dot(inputs['h_vec']) * d_inputs[F_name].flatten()
                    vec = mtx_hf.dot(vec)
                    vec = mtx_lu.solve(vec)

                    if Y_name in d_outputs:
                        d_outputs[Y_name] += vec[:num_Y].reshape(
                            (num_time_steps - 1, num_stages,) + shape)

                    if y_name in d_outputs:
                        d_outputs[y_name] += vec[num_Y:].reshape(
                            (num_time_steps, num_step_vars,) + shape)

                if 'h_vec' in d_inputs:
                    vec = mtx_h.dot(d_inputs['h_vec']) * inputs[F_name].flatten()
                    vec = mtx_hf.dot(vec)
                    vec = mtx_lu.solve(vec)

                    if Y_name in d_outputs:
                        d_outputs[Y_name] += vec[:num_Y].reshape(
                            (num_time_steps - 1, num_stages,) + shape)

                    if y_name in d_outputs:
                        d_outputs[y_name] += vec[num_Y:].reshape(
                            (num_time_steps, num_step_vars,) + shape)

            # ------------------------------------------------------------------------------

            elif mode == 'rev':
                if y_name in d_outputs:
                    if y0_name in d_inputs:
                        d_inputs[y0_name] -= d_outputs[y_name][0, :, :]

                if Y_name in d_outputs:
                    vec = np.concatenate([d_outputs[Y_name].flatten(), np.zeros(num_y)])
                    vec = mtx_lu.solve(vec, 'T')
                    vec = mtx_hf.T.dot(vec)

                    if F_name in d_inputs:
                        d_inputs[F_name] += (mtx_h.dot(inputs['h_vec']) * vec).reshape(
                            (num_time_steps - 1, num_stages,) + shape)

                    if 'h_vec' in d_inputs:
                        d_inputs['h_vec'] += (mtx_h.T.dot(vec))

                if y_name in d_outputs:
                    vec = np.concatenate([np.zeros(num_Y), d_outputs[y_name].flatten()])
                    vec = mtx_lu.solve(vec, 'T')
                    vec = mtx_hf.T.dot(vec)

                    if F_name in d_inputs:
                        d_inputs[F_name] += (mtx_h.dot(inputs['h_vec']) * vec).reshape(
                            (num_time_steps - 1, num_stages,) + shape)

                    if 'h_vec' in d_inputs:
                        d_inputs['h_vec'] += (mtx_h.T.dot(vec))
