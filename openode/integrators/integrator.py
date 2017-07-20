from __future__ import division

import numpy as np
from openmdao.api import Group, IndepVarComp
from six import iteritems

import openode.schemes.scheme as schemes
from openode.components.time_comp import TimeComp
from openode.components.starting_comp import StartingComp
from openode.schemes.scheme import GLMScheme
from openode.schemes.runge_kutta import RK4
from openode.ode_function import ODEFunction
from openode.utils.var_names import get_name


class Integrator(Group):
    """
    The base class for all integration schemes.
    """

    def initialize(self):
        self.metadata.declare('ode_function', type_=ODEFunction, required=True)
        self.metadata.declare('times', type_=np.ndarray, required=True)
        self.metadata.declare('initial_conditions', type_=dict)
        self.metadata.declare('scheme', default=RK4(), type_=GLMScheme)

    def setup(self):
        ode_function = self.metadata['ode_function']
        states = ode_function._states

        # Ensure that all initial_conditions are valid
        for state_name, value in iteritems(self.metadata['initial_conditions']):
            assert state_name in ode_function._states, \
                'State name %s is not valid in the initial conditions' % state_name

            if np.isscalar(value):
                assert ode_function._states[state_name]['shape'] == (1,), \
                    'The initial condition for state %s has the wrong shape' % state_name
            else:
                assert ode_function._states[state_name]['shape'] == value.shape, \
                    'The initial condition for state %s has the wrong shape' % state_name

        initial_conditions = self.metadata['initial_conditions']
        times = self.metadata['times']
        time_units = ode_function._time_options['units']

        promotes = []
        # Initial conditions
        if len(initial_conditions) > 0:
            ivcomp = IndepVarComp()

            for state_name, value in iteritems(initial_conditions):
                state = ode_function._states[state_name]
                ivcomp.add_output(state_name, val=value, units=state['units'])
                promotes.append((state_name, 'IC:{}'.format(state_name)))
            self.add_subsystem('initial_conditions', ivcomp, promotes_outputs=promotes)

        # Start and end times
        if times is not None:
            comp = IndepVarComp()
            comp.add_output('times', val=times, units=time_units)
            self.add_subsystem('inputs_t', comp)

        # Time comp
        abscissa = self.metadata['scheme'].abscissa
        self.add_subsystem('time_comp',
            TimeComp(time_units=time_units, glm_abscissa=abscissa, num_time_steps=len(times)))
        self.connect('inputs_t.times', 'time_comp.times')

        # Starting method
        self.add_subsystem('starting_comp', StartingComp(states=states), promotes_inputs=promotes)
        # self._connect_states(
        #     'initial_conditions', 'state_name',
        #     'starting_comp', 'state_name')

    def _get_names(self, comp, type_, i_step=None, i_stage=None, j_stage=None):
        names_list = []
        for state_name, state in iteritems(self.metadata['ode_function']._states):
            if type_ == 'rate_target':
                names = '{}.{}'.format(comp, state['rate_target'])
            elif type_ == 'state_targets':
                names = ['{}.{}'.format(comp, tgt) for tgt in state['state_targets']]
            else:
                names = '{}.{}'.format(comp, get_name(
                    type_, state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage))

            names_list.append(names)

        return names_list

    def _connect_states(self, srcs_list, tgts_list):
        for srcs, tgts in zip(srcs_list, tgts_list):
            self.connect(srcs, tgts)

    def _connect_states00(self, src_comp, src_type, tgt_comp, tgt_type,
            src_stage=None, tgt_stage=None, src_step=None, tgt_step=None):

        for state_name, state in iteritems(self.metadata['ode_function']._states):

            if src_type == 'state_name':
                src_name = state_name
            elif src_type == 'F_name':
                src_name = get_F_name(src_stage, state_name)
            elif src_type == 'Y_name':
                src_name = get_Y_name(src_stage, state_name)
            elif src_type == 'y_old_name':
                src_name = get_y_old_name(state_name)
            elif src_type == 'y_new_name':
                src_name = get_y_new_name(state_name)
            elif src_type == 'rate_target':
                src_name = state['rate_target']
            else:
                src_name = src_type + ':' + state_name

            if tgt_type == 'state_name':
                tgt_names = [state_name]
            elif tgt_type == 'F_name':
                tgt_names = [get_F_name(tgt_stage, state_name)]
            elif tgt_type == 'Y_name':
                tgt_names = [get_Y_name(tgt_stage, state_name)]
            elif tgt_type == 'y_name':
                tgt_names = ['y:%s' % state_name]
            elif tgt_type == 'y_old_name':
                tgt_names = [get_y_old_name(state_name)]
            elif tgt_type == 'y_new_name':
                tgt_names = [get_y_new_name(state_name)]
            elif tgt_type == 'state_targets':
                tgt_names = state['state_targets']
            elif tgt_type == 'step_name':
                tgt_names = [get_step_name(tgt_step, state_name)]
            else:
                tgt_names = [tgt_type + ':' + state_name]

            for tgt_name in tgt_names:
                self.connect(
                    '%s.%s' % (src_comp, src_name),
                    '%s.%s' % (tgt_comp, tgt_name))

    def _create_ode(self, num):
        ode_function = self.metadata['ode_function']
        return ode_function._system_class(num=num, **ode_function._system_init_kwargs)

    def _get_meta(self):
        ode_function = self.metadata['ode_function']

        states = ode_function._states
        time_units = ode_function._time_options['units']
        times = self.metadata['times']

        return states, time_units, times

    def _get_scheme(self):
        scheme = self.metadata['scheme']

        return scheme.A, scheme.B, scheme.U, scheme.V, scheme.num_stages, scheme.num_values
