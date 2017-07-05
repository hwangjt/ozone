import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from openode.utils.units import get_rate_units


class VectorizedStepComp(ExplicitComponent):

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

        self.mtxs = {}

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            self.add_input('F:%s' % state_name,
                shape=(size, num_time_steps - 1, num_stages),
                units=state['units'])

            self.add_output(state_name,
                shape=(size, num_time_steps - 1, num_stages + num_step_vars),
                units=state['units'])

            num = size * (num_time_steps - 1)
            data = np.zeros((num, num_stages + num_step_vars, num_stages))
            rows = np.zeros((num, num_stages + num_step_vars, num_stages), int)
            cols = np.zeros((num, num_stages + num_step_vars, num_stages), int)

            for ind_j in range(num_stages):
                for ind_i in range(num_stages):
                    data[:, ind_i, ind_j] = glm_A[ind_i, ind_j]
                for ind_i in range(num_stages, num_stages + num_step_vars):
                    data[:, ind_i, ind_j] = glm_B[ind_i, ind_j]

            for ind_j in range(num_stages):
                rows[:, :, ind_j] = np.arange(num * (num_stages + num_step_vars)).reshape(
                    (num, num_stages + num_step_vars))

            for ind_i in range(num_stages + num_step_vars)
                cols[:, ind_i, :] = np.arange(num * (num_stages)).reshape(
                    (num, num_stages))

            data = data.flatten()
            rows = rows.flatten()
            cols = cols.flatten()
            mtx = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(
                    num * (num_stages + num_step_vars),
                    np.arange(num * (num_stages)))
