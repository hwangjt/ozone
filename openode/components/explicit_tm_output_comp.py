import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_y_new_name, get_step_name


class ExplicitTMOutputComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('time_spacing', type_=np.ndarray, required=True)
        self.metadata.declare('num_stages', type_=int, required=True)

    def setup(self):
        time_spacing = self.metadata['time_spacing']

        num_time_steps = len(time_spacing)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            for i_step in range(num_time_steps):
                name = get_step_name(i_step, state_name)

                self.add_input(name, src_indices=np.arange(size).reshape(state['shape']),
                    units=state['units'])

            self.add_output(state_name, shape=(num_time_steps,) + state['shape'],
                units=state['units'])

            vals = np.ones(size)
            full_rows = np.arange(num_time_steps * size).reshape((num_time_steps, size))
            cols = np.arange(size)

            for i_step in range(num_time_steps):
                name = get_step_name(i_step, state_name)

                rows = full_rows[i_step, :]

                mtx = scipy.sparse.csc_matrix(
                    (vals, (rows, cols)),
                    shape=(num_time_steps * size, size))
                mtx = mtx.todense()

                self.declare_partials(state_name, name, val=mtx)

    def compute(self, inputs, outputs):
        time_spacing = self.metadata['time_spacing']

        num_time_steps = len(time_spacing)

        for state_name, state in iteritems(self.metadata['states']):

            for i_step in range(num_time_steps):
                name = get_step_name(i_step, state_name)

                outputs[state_name][i_step, :] = inputs[name]
