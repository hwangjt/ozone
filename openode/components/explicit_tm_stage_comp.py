import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_F_name, get_y_old_name, get_Y_name
from openode.utils.units import get_rate_units


class ExplicitTMStageComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('time_units', values=(None,), type_=str, required=True)
        self.metadata.declare('num_stages', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('glm_A', type_=np.ndarray, required=True)
        self.metadata.declare('glm_U', type_=np.ndarray, required=True)
        self.metadata.declare('i_stage', type_=int, required=True)

    def setup(self):
        time_units = self.metadata['time_units']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_stage = self.metadata['i_stage']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        self.add_input('h', units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            for j_stage in range(i_stage):
                F_name = get_F_name(j_stage, state_name)

                self.add_input(F_name, shape=(1,) + state['shape'],
                    units=get_rate_units(state['units'], time_units))

            y_old_name = get_y_old_name(state_name)
            Y_name = get_Y_name(i_stage, state_name)

            self.add_input(y_old_name, shape=(num_step_vars,) + state['shape'],
                units=state['units'])

            self.add_output(Y_name, shape=(1,) + state['shape'],
                units=state['units'])

            vals = np.zeros((num_step_vars, size))
            rows = np.zeros((num_step_vars, size), int)
            cols = np.arange(num_step_vars * size)
            for i_step in range(num_step_vars):
                vals[i_step, :] = glm_U[i_stage, i_step]
                rows[i_step, :] = np.arange(size)
            vals = vals.flatten()
            rows = rows.flatten()

            self.declare_partials(Y_name, y_old_name, val=vals, rows=rows, cols=cols)

            for j_stage in range(i_stage):
                F_name = get_F_name(j_stage, state_name)

                arange = np.arange(size)
                self.declare_partials(Y_name, F_name, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_stage = self.metadata['i_stage']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_old_name = get_y_old_name(state_name)
            Y_name = get_Y_name(i_stage, state_name)

            outputs[Y_name] = np.einsum('i,ij...->j...', glm_U[i_stage, :], inputs[y_old_name])

            for j_stage in range(i_stage):
                F_name = get_F_name(j_stage, state_name)

                outputs[Y_name] += inputs['h'] * glm_A[i_stage, j_stage] * inputs[F_name]

    def compute_partials(self, inputs, outputs, partials):
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_stage = self.metadata['i_stage']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            Y_name = get_Y_name(i_stage, state_name)

            partials[Y_name, 'h'][:, 0] = 0.

            for j_stage in range(i_stage):
                F_name = get_F_name(j_stage, state_name)

                partials[Y_name, F_name] = inputs['h'] * glm_A[i_stage, j_stage]

                partials[Y_name, 'h'][:, 0] += glm_A[i_stage, j_stage] * inputs[F_name].flatten()
