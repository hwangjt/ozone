import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from ozone.utils.var_names import get_name
from ozone.utils.units import get_rate_units
from ozone.utils.sparse_linear_spline import get_sparse_linear_spline


class StaticParameterComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('static_parameters', types=dict)

    def setup(self):
        for parameter_name, parameter in iteritems(self.options['static_parameters']):
            size = np.prod(parameter['shape'])
            shape = parameter['shape']

            in_name = get_name('in', parameter_name)
            out_name = get_name('out', parameter_name)

            self.add_input(in_name, shape=shape, units=parameter['units'])
            self.add_output(out_name, shape=shape, units=parameter['units'])

            ones = np.ones(size)
            arange = np.arange(size)
            self.declare_partials(out_name, in_name, val=ones, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for parameter_name, parameter in iteritems(self.options['static_parameters']):
            in_name = get_name('in', parameter_name)
            out_name = get_name('out', parameter_name)

            outputs[out_name] = inputs[in_name]
