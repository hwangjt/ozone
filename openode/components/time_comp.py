import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent


class TimeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('time_units', type_=(str, type(None)), required=True)
        self.metadata.declare('glm_abscissa', type_=np.ndarray, required=True)
        self.metadata.declare('num_time_steps', type_=int, required=True)

    def setup(self):
        time_units = self.metadata['time_units']
        abscissa = self.metadata['glm_abscissa']
        num_time_steps = self.metadata['num_time_steps']
        num_abscissa = len(abscissa)
        num_h_vec = num_time_steps - 1

        self.add_input('times', shape=num_time_steps, units=time_units)
        self.add_output('h_vec', shape=num_h_vec, units=time_units)
        self.add_output('abscissa_times', shape=num_h_vec * num_abscissa)

        arange_h = np.arange(num_h_vec)

        data1 = np.ones(num_h_vec)
        rows1 = arange_h
        cols1 = arange_h + 1

        data2 = -np.ones(num_h_vec)
        rows2 = arange_h
        cols2 = arange_h

        data = np.concatenate([data1, data2])
        rows = np.concatenate([rows1, rows2])
        cols = np.concatenate([cols1, cols2])

        self.declare_partials('h_vec', 'times', val=data, rows=rows, cols=cols)

        arange_a = np.arange(num_h_vec * num_abscissa)

        data1 = np.tile(abscissa, num_h_vec)
        rows1 = arange_a
        cols1 = np.repeat(arange_h, num_abscissa) + 1

        data2 = np.tile(1 - abscissa, num_h_vec)
        rows2 = arange_a
        cols2 = np.repeat(arange_h, num_abscissa)

        data = np.concatenate([data1, data2])
        rows = np.concatenate([rows1, rows2])
        cols = np.concatenate([cols1, cols2])

        self.declare_partials('abscissa_times', 'times', val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        abscissa = self.metadata['glm_abscissa']

        outputs['h_vec'] = h_vec = inputs['times'][1:] - inputs['times'][:-1]
        outputs['abscissa_times'] = (inputs['times'][:-1, np.newaxis]
            + np.outer(h_vec, abscissa)).ravel()
