import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent


class TimeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('time_spacing', type_=np.ndarray, required=True)
        self.metadata.declare('time_units', values=(None,), type_=str, required=True)
        self.metadata.declare('glm_abscissa', type_=np.ndarray, required=True)

    def setup(self):
        time_spacing = self.metadata['time_spacing']
        time_units = self.metadata['time_units']
        abscissa = self.metadata['glm_abscissa']
        num_abscissa = len(abscissa)
        num_times = len(time_spacing)

        self.add_input('start_time', val=-1., units=time_units)
        self.add_input('end_time', val=1., units=time_units)

        self.add_output('times', shape=num_times, units=time_units)
        self.add_output('h_vec', shape=num_times - 1, units=time_units)
        self.add_output('abscissa_times',
                        shape=(num_times - 1) * num_abscissa
                        )

        self.declare_partials('times', 'start_time', val=1 - time_spacing)
        self.declare_partials('times', 'end_time', val=time_spacing)

        h = np.diff(time_spacing)

        self.declare_partials('h_vec', 'start_time', val=-h)
        self.declare_partials('h_vec', 'end_time', val=h)

        # abscissa times are equivalent to
        # np.repeat(times[:-1], len(absiccas))
        #   +  np.repeat(h, len(absiccas))*np.tile(absiccas, len(times)-1)
        self.declare_partials('abscissa_times', 'start_time',
                              val=np.repeat(1-time_spacing[:-1], num_abscissa)
                                  + np.repeat(-h, num_abscissa) * np.tile(abscissa, num_times - 1))

        self.declare_partials('abscissa_times', 'end_time',
                              val=np.repeat(time_spacing[:-1], num_abscissa)
                                  + np.repeat(h, num_abscissa) * np.tile(abscissa, num_times - 1))

    def compute(self, inputs, outputs):
        time_spacing = self.metadata['time_spacing']

        outputs['times'] = times = inputs['start_time'] \
            + (inputs['end_time'] - inputs['start_time']) * time_spacing

        h = np.diff(time_spacing)

        outputs['h_vec'] = h_vec = (inputs['end_time'] - inputs['start_time']) * h

        abscissa = self.metadata['glm_abscissa']

        outputs['abscissa_times'] = (times[:-1, np.newaxis] + np.outer(h_vec, abscissa)).ravel()
