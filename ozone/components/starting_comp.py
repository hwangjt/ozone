import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from ozone.utils.var_names import get_name


class StartingComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', type_=dict, required=True)

    def setup(self):
        self.declare_partials('*', '*', dependent=False)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            IC_name = get_name('IC', state_name)
            starting_name = get_name('starting', state_name)

            self.add_input(IC_name, shape=state['shape'], units=state['units'])
            self.add_output(starting_name, shape=(1,) + state['shape'], units=state['units'])

            ones = np.ones(size)
            arange = np.arange(size)
            self.declare_partials(starting_name, IC_name, val=ones, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for state_name, state in iteritems(self.metadata['states']):
            IC_name = get_name('IC', state_name)
            starting_name = get_name('starting', state_name)

            outputs[starting_name] = inputs[IC_name].reshape((1,) + state['shape'])
