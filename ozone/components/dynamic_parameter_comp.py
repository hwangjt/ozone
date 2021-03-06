import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from ozone.utils.var_names import get_name
from ozone.utils.units import get_rate_units
from ozone.utils.sparse_linear_spline import get_sparse_linear_spline


class DynamicParameterComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('dynamic_parameters', types=dict)
        self.options.declare('normalized_times', types=np.ndarray)
        self.options.declare('stage_norm_times', types=np.ndarray)

    def setup(self):
        normalized_times = self.options['normalized_times']
        stage_norm_times = self.options['stage_norm_times']

        num_times = len(normalized_times)
        num_stage_times = len(stage_norm_times)

        data0, rows0, cols0 = get_sparse_linear_spline(normalized_times, stage_norm_times)
        nnz = len(data0)

        self.mtx = scipy.sparse.csc_matrix((data0, (rows0, cols0)),
            shape=(num_stage_times, num_times))

        for parameter_name, parameter in iteritems(self.options['dynamic_parameters']):
            size = np.prod(parameter['shape'])
            shape = parameter['shape']

            in_name = get_name('in', parameter_name)
            out_name = get_name('out', parameter_name)

            self.add_input(in_name,
                shape=(num_times,) + shape,
                units=parameter['units'])

            self.add_output(out_name,
                shape=(num_stage_times,) + shape,
                units=parameter['units'])

            # (num_stage_times, num_out,) + shape
            data = np.einsum('i,...->i...', data0, np.ones(shape)).flatten()
            rows = (
                np.einsum('i,...->i...', rows0, size * np.ones(shape, int))
                + np.einsum('i,...->i...', np.ones(nnz, int), np.arange(size).reshape(shape))
            ).flatten()
            cols = (
                np.einsum('i,...->i...', cols0, size * np.ones(shape, int))
                + np.einsum('i,...->i...', np.ones(nnz, int), np.arange(size).reshape(shape))
            ).flatten()

            self.declare_partials(out_name, in_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        normalized_times = self.options['normalized_times']
        stage_norm_times = self.options['stage_norm_times']

        num_times = len(normalized_times)
        num_stage_times = len(stage_norm_times)

        for parameter_name, parameter in iteritems(self.options['dynamic_parameters']):
            size = np.prod(parameter['shape'])
            shape = parameter['shape']
            in_name = get_name('in', parameter_name)
            out_name = get_name('out', parameter_name)

            outputs[out_name] = self.mtx.dot(inputs[in_name].reshape((num_times, size))).reshape((num_stage_times,) + shape)