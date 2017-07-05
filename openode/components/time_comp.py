import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent


class TimeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('time_spacing', type_=np.ndarray, required=True)
        self.metadata.declare('time_units', values=(None,), type_=str, required=True)

    def setup(self):
        time_spacing = self.metadata['time_spacing']
        time_units = self.metadata['time_units']

        self.add_input('start_time', val=-1., units=time_units)
        self.add_input('end_time', val=1., units=time_units)

        self.add_output('times', shape=len(time_spacing), units=time_units)
        self.add_output('h_vec', shape=len(time_spacing) - 1, units=time_units)

        self.declare_partials('times', 'start_time', val=1 - time_spacing)
        self.declare_partials('times', 'end_time', val=time_spacing)

        self.declare_partials('h_vec', 'start_time', val=1 - time_spacing[1:] + time_spacing[:-1])
        self.declare_partials('h_vec', 'end_time', val=time_spacing[1:] - time_spacing[:-1])

    def compute(self, inputs, outputs):
        time_spacing = self.metadata['time_spacing']

        outputs['times'] = inputs['start_time'] \
            + (inputs['end_time'] - inputs['start_time']) * time_spacing

        outputs['h_vec'] = inputs['start_time'] \
            + (inputs['end_time'] - inputs['start_time']) * (time_spacing[1:] - time_spacing[:-1])