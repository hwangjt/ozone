import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_name
from openode.utils.units import get_rate_units


class ExplicitTMStepComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('time_units', values=(None,), type_=str, required=True)
        self.metadata.declare('num_stages', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('glm_B', type_=np.ndarray, required=True)
        self.metadata.declare('glm_V', type_=np.ndarray, required=True)
        self.metadata.declare('i_step', type_=int, required=True)

    def setup(self):
        time_units = self.metadata['time_units']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_step = self.metadata['i_step']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        self.dy_dF = dy_dF = {}

        self.declare_partials('*', '*', dependent=False)

        self.add_input('h', units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                self.add_input(F_name, shape=(1,) + state['shape'],
                    units=get_rate_units(state['units'], time_units))

            self.add_input(y_old_name, shape=(num_step_vars,) + state['shape'],
                units=state['units'])

            self.add_output(y_new_name, shape=(num_step_vars,) + state['shape'],
                units=state['units'])

            self.declare_partials(y_new_name, 'h', dependent=True)

            vals = np.zeros((num_step_vars, num_step_vars, size))
            rows = np.zeros((num_step_vars, num_step_vars * size), int)
            cols = np.zeros((num_step_vars, num_step_vars * size), int)
            for ii_step in range(num_step_vars):
                for jj_step in range(num_step_vars):
                    vals[ii_step, jj_step, :] = glm_V[ii_step, jj_step]
                    rows[ii_step, :] = np.arange(num_step_vars * size)
                    cols[jj_step, :] = np.arange(num_step_vars * size)
            vals = vals.flatten()
            rows = rows.flatten()
            cols = cols.flatten()

            self.declare_partials(y_new_name, y_old_name, val=vals, rows=rows, cols=cols)

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                vals = np.zeros((num_step_vars, 1, size))
                rows = np.arange(num_step_vars * 1 * size)
                cols = np.zeros((num_step_vars, 1 * size), int)
                for ii_step in range(num_step_vars):
                    vals[ii_step, 0, :] = glm_B[ii_step, j_stage]
                    cols[ii_step, :] = np.arange(1 * size)
                vals = vals.flatten()
                cols = cols.flatten()

                self.declare_partials(y_new_name, F_name, rows=rows, cols=cols)
                dy_dF[state_name, j_stage] = vals

    def compute(self, inputs, outputs):
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']
        i_step = self.metadata['i_step']

        ii_step = 0

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            outputs[y_new_name] = np.einsum('ij,jk...->ik...', glm_V, inputs[y_old_name])

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                outputs[y_new_name] += inputs['h'] * glm_B[ii_step, j_stage] * inputs[F_name]

    def compute_partials(self, inputs, outputs, partials):
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']
        i_step = self.metadata['i_step']

        ii_step = 0

        dy_dF = self.dy_dF

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_new_name = get_name('y_new', state_name, i_step=i_step)

            partials[y_new_name, 'h'][:, 0] = 0.

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                partials[y_new_name, F_name] = inputs['h'] * dy_dF[state_name, j_stage]

                partials[y_new_name, 'h'][:, 0] += glm_B[ii_step, j_stage] \
                    * inputs[F_name].flatten()
