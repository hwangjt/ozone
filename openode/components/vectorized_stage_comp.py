import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_F_name, get_y_old_name, get_y_new_name
from openode.utils.units import get_rate_units


class VectorizedStageComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('time_units', values=(None,), type_=str, required=True)
        self.metadata.declare('num_time_steps', type_=int, required=True)
        self.metadata.declare('num_stages', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('glm_A', type_=np.ndarray, required=True)
        self.metadata.declare('glm_U', type_=np.ndarray, required=True)

    def setup(self):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        h_arange = np.arange(num_time_steps - 1)

        self.add_input('h_vec', shape=(num_time_steps - 1), units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = 'F:%s' % state_name
            y_name = 'y:%s' % state_name
            Y_out_name = 'Y_out:%s' % state_name
            Y_in_name = 'Y_in:%s' % state_name

            Y_arange = np.arange((num_time_steps - 1) * num_stages * size).reshape(
                (num_time_steps - 1, num_stages,) + shape)

            F_arange = np.arange((num_time_steps - 1) * num_stages * size).reshape(
                (num_time_steps - 1, num_stages,) + shape)

            y_arange = np.arange(num_time_steps * num_step_vars * size).reshape(
                (num_time_steps, num_step_vars,) + shape)

            self.add_input(F_name,
                shape=(num_time_steps - 1, num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_input(y_name,
                shape=(num_time_steps, num_step_vars,) + shape,
                units=state['units'])

            self.add_input(Y_in_name, val=0.,
                shape=(num_time_steps - 1, num_stages,) + shape,
                units=state['units'])

            self.add_output(Y_out_name,
                shape=(num_time_steps - 1, num_stages,) + shape,
                units=state['units'])

            # -----------------

            ones = -np.ones((num_time_steps - 1) * num_stages * size)
            arange = np.arange((num_time_steps - 1) * num_stages * size)
            self.declare_partials(Y_out_name, Y_in_name, val=ones, rows=arange, cols=arange)

            # -----------------

            # (num_time_steps - 1, num_stages, num_stages,) + shape
            rows = np.einsum('ij...,k->ijk...', Y_arange, np.ones(num_stages)).flatten()

            cols = np.einsum('jk...,i->ijk...',
                np.ones((num_stages, num_stages,) + shape), h_arange).flatten()
            self.declare_partials(Y_out_name, 'h_vec', rows=rows, cols=cols)

            cols = np.einsum('ik...,j->ijk...', F_arange, np.ones(num_stages)).flatten()
            self.declare_partials(Y_out_name, F_name, rows=rows, cols=cols)

            # -----------------

            # (num_time_steps - 1, num_stages, num_step_vars,) + shape
            data = np.einsum('jk,i...->ijk...',
                glm_U, np.ones((num_time_steps - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...', Y_arange, np.ones(num_step_vars)).flatten()
            cols = np.einsum('ikl,j->ijk...', y_arange[:-1, :, :], np.ones(num_stages)).flatten()

            self.declare_partials(Y_out_name, y_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = 'F:%s' % state_name
            y_name = 'y:%s' % state_name
            Y_out_name = 'Y_out:%s' % state_name
            Y_in_name = 'Y_in:%s' % state_name

            outputs[Y_out_name] = -inputs[Y_in_name] \
                + np.einsum('jk,i,ik...->ij...', glm_A, inputs['h_vec'], inputs[F_name]) \
                + np.einsum('jk,ik...->ij...', glm_U, inputs[y_name][:-1, :, :])

    def compute_partials(self, inputs, outputs, partials):
        time_units = self.metadata['time_units']
        num_time_steps = self.metadata['num_time_steps']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = 'F:%s' % state_name
            y_name = 'y:%s' % state_name
            Y_out_name = 'Y_out:%s' % state_name
            Y_in_name = 'Y_in:%s' % state_name

            # (num_time_steps - 1, num_stages, num_stages,) + shape

            partials[Y_out_name, F_name] = np.einsum(
                '...,jk,i->ijk...', np.ones(shape), glm_A, inputs['h_vec']).flatten()

            partials[Y_out_name, 'h_vec'] = np.einsum(
                'jk,ik...->ijk...', glm_A, inputs[F_name]).flatten()
