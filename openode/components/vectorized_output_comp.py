import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_name


class VectorizedOutputComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)
        self.metadata.declare('num_starting_time_steps', type_=int, required=True)
        self.metadata.declare('num_my_time_steps', type_=int, required=True)
        self.metadata.declare('num_step_vars', type_=int, required=True)
        self.metadata.declare('starting_coeffs', type_=(np.ndarray, type(None)))

    def setup(self):
        num_starting_time_steps = self.metadata['num_starting_time_steps']
        num_my_time_steps = self.metadata['num_my_time_steps']
        num_step_vars = self.metadata['num_step_vars']
        starting_coeffs = self.metadata['starting_coeffs']

        num_time_steps = num_starting_time_steps + num_my_time_steps - 1

        has_starting_method = num_starting_time_steps > 1
        is_starting_method = starting_coeffs is not None

        if is_starting_method:
            num_starting = starting_coeffs.shape[0]

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y_name = get_name('y', state_name)
            starting_state_name = get_name('starting_state', state_name)
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)

            self.add_input(y_name,
                shape=(num_my_time_steps, num_step_vars,) + shape,
                units=state['units'])

            if has_starting_method:
                self.add_input(starting_state_name,
                    shape=(num_starting_time_steps,) + shape,
                    units=state['units'])

            self.add_output(out_state_name,
                shape=(num_time_steps,) + shape,
                units=state['units'])

            if is_starting_method:
                self.add_output(starting_name,
                    shape=(num_starting,) + shape,
                    units=state['units'])

            y_arange = np.arange(num_my_time_steps * num_step_vars * size).reshape(
                (num_my_time_steps, num_step_vars,) + shape)

            out_state_arange = np.arange(num_time_steps * size).reshape(
                (num_time_steps,) + shape)

            data = np.ones(num_my_time_steps * size, int)
            rows = out_state_arange[num_starting_time_steps - 1:, :].flatten()
            cols = y_arange[:, 0, :].flatten()

            self.declare_partials(out_state_name, y_name, val=data, rows=rows, cols=cols)

            if has_starting_method:

                starting_state_arange = np.arange(num_starting_time_steps * size).reshape(
                    (num_starting_time_steps,) + shape)

                data = np.ones((num_starting_time_steps - 1) * size, int)
                rows = out_state_arange[:num_starting_time_steps - 1, :].flatten()
                cols = starting_state_arange[:-1, :].flatten()

                self.declare_partials(out_state_name, starting_state_name,
                    val=data, rows=rows, cols=cols)

            if is_starting_method:

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
        num_starting_time_steps = self.metadata['num_starting_time_steps']
        num_my_time_steps = self.metadata['num_my_time_steps']
        starting_coeffs = self.metadata['starting_coeffs']

        has_starting_method = num_starting_time_steps > 1
        is_starting_method = starting_coeffs is not None

        for state_name, state in iteritems(self.metadata['states']):
            y_name = get_name('y', state_name)
            starting_state_name = get_name('starting_state', state_name)
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)

            outputs[out_state_name][num_starting_time_steps - 1:] = inputs[y_name][:, 0, :]

            if has_starting_method:

                outputs[out_state_name][:num_starting_time_steps - 1] = \
                    inputs[starting_state_name][:-1, :]

            if is_starting_method:

                outputs[starting_name] = np.einsum('ijk,jk...->i...',
                    starting_coeffs, inputs[y_name])
