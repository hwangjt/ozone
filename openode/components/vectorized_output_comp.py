import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_name


class VectorizedOutputComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('num_time_steps', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('starting_coeffs', values=(None,), type_=np.ndarray)

    def setup(self):
        num_time_steps = self.metadata['num_time_steps']
        num_step_vars = self.metadata['num_step_vars']
        starting_coeffs = self.metadata['starting_coeffs']

        if starting_coeffs is not None:
            num_starting = starting_coeffs.shape[0]

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y_name = get_name('y', state_name)
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)

            self.add_input(y_name,
                shape=(num_time_steps, num_step_vars,) + shape,
                units=state['units'])

            self.add_output(out_state_name,
                shape=(num_time_steps,) + shape,
                units=state['units'])

            if starting_coeffs is not None:
                self.add_output(starting_name,
                    shape=(num_starting,) + shape,
                    units=state['units'])

            y_arange = np.arange(num_time_steps * num_step_vars * size).reshape(
                (num_time_steps, num_step_vars,) + shape)

            state_arange = np.arange(num_time_steps * size).reshape(
                (num_time_steps,) + shape)

            data = np.ones(num_time_steps * size, int)
            rows = state_arange.flatten()
            cols = y_arange[:, 0, :].flatten()

            self.declare_partials(out_state_name, y_name, val=data, rows=rows, cols=cols)

            if starting_coeffs is not None:
                starting_arange = np.arange(num_starting * size).reshape(
                    (num_starting,) + shape)

                # (num_starting, num_time_steps, num_step_vars,) + shape
                data = np.einsum('ijk,...->ijk...', starting_coeffs, np.ones(shape)).flatten()
                rows = np.einsum('jk,i...->ijk...',
                    np.ones((num_time_steps, num_step_vars), int), starting_arange).flatten()
                cols = np.einsum('i,jk...->ijk...',
                    np.ones(num_starting, int), y_arange).flatten()

                self.declare_partials(starting_name, y_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        starting_coeffs = self.metadata['starting_coeffs']

        for state_name, state in iteritems(self.metadata['states']):
            y_name = get_name('y', state_name)
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)

            outputs[out_state_name] = inputs[y_name][:, 0, :]

            if starting_coeffs is not None:
                outputs[starting_name] = np.einsum('ijk,jk...->i...',
                    starting_coeffs, inputs[y_name])
