import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_y_new_name, get_step_name, get_name


class VectorizedOutputComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('num_time_steps', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)

    def setup(self):
        num_time_steps = self.metadata['num_time_steps']
        num_step_vars = self.metadata['num_step_vars']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y_name = get_name('y', state_name)

            self.add_input(y_name,
                shape=(num_time_steps, num_step_vars,) + shape,
                units=state['units'])

            self.add_output(state_name,
                shape=(num_time_steps,) + shape,
                units=state['units'])

            y_arange = np.arange(num_time_steps * num_step_vars * size).reshape(
                (num_time_steps, num_step_vars,) + shape)

            state_arange = np.arange(num_time_steps * size).reshape(
                (num_time_steps,) + shape)

            data = np.ones(num_time_steps * size, int)
            rows = y_arange[:, 0, :].flatten()
            cols = state_arange.flatten()

            self.declare_partials(state_name, y_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        for state_name, state in iteritems(self.metadata['states']):
            y_name = get_name('y', state_name)

            outputs[state_name] = inputs[y_name][:, 0, :]
