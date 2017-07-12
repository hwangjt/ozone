import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from openode.utils.var_names import get_y_new_name


class StartingComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)

    def setup(self):
        self.declare_partials('*', '*', dependent=False)
        
        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_new_name = get_y_new_name(state_name)

            self.add_input(state_name, shape=state['shape'], units=state['units'])
            self.add_output(y_new_name, shape=(1,) + state['shape'], units=state['units'])

            ones = np.ones(size)
            arange = np.arange(size)
            self.declare_partials(y_new_name, state_name, val=ones, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for state_name, state in iteritems(self.metadata['states']):
            y_new_name = get_y_new_name(state_name)

            outputs[y_new_name] = inputs[state_name].reshape((1,) + state['shape'])
