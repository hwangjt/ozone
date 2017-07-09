import numpy as np
from openmdao.api import Group, IndepVarComp
from six import iteritems

import openode.schemes.scheme as schemes
from openode.components.time_comp import TimeComp
from openode.components.starting_comp import StartingComp
from openode.schemes.scheme import GLMScheme
from openode.schemes.runge_kutta import RK4
from openode.ode_function import ODEFunction
from openode.utils.var_names import get_Y_name, get_F_name, get_y_old_name, get_y_new_name, \
    get_step_name


class Integrator(Group):
    """
    The base class for all integration schemes.
    """

    def initialize(self):
        self.metadata.declare('ode_function', type_=ODEFunction, required=True)
        self.metadata.declare('time_spacing', type_=np.ndarray, required=True)
        self.metadata.declare('initial_conditions', type_=dict)
        self.metadata.declare('start_time', values=(None,), type_=(int, float))
        self.metadata.declare('end_time', values=(None,), type_=(int, float))
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

        # Normalize time_spacing
        spacing = self.metadata['time_spacing']
        self.metadata['time_spacing'] = (spacing - spacing[0]) / (spacing[-1] - spacing[0])

        initial_conditions = self.metadata['initial_conditions']
        start_time = self.metadata['start_time']
        end_time = self.metadata['end_time']

        time_units = ode_function._time_options['units']
        time_spacing = self.metadata['time_spacing']

        # Initial conditions
        if len(initial_conditions) > 0:
            comp = IndepVarComp()
            for state_name, value in iteritems(initial_conditions):
                state = ode_function._states[state_name]
                comp.add_output(state_name, val=value, units=state['units'])
            self.add_subsystem('initial_conditions', comp)

        # Start and end times
        if start_time is not None or end_time is not None:
            comp = IndepVarComp()
            if start_time is not None:
                comp.add_output('start_time', val=start_time, units=time_units)
            if end_time is not None:
                comp.add_output('end_time', val=end_time, units=time_units)
            self.add_subsystem('time_interval', comp)

        # Time comp
        abscissa = self.metadata['scheme'].abscissa
        self.add_subsystem('time_comp',
            TimeComp(time_spacing=time_spacing, time_units=time_units,
                     glm_abscissa=abscissa))
        self.connect('time_interval.start_time', 'time_comp.start_time')
        self.connect('time_interval.end_time', 'time_comp.end_time')

        # Starting method
        self.add_subsystem('starting_comp', StartingComp(states=states))
        self._connect_states(
            'initial_conditions', 'state_name',
            'starting_comp', 'state_name')

    def _connect_states(self, src_comp, src_type, tgt_comp, tgt_type,
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
        time_spacing = self.metadata['time_spacing']

        return states, time_units, time_spacing

    def _get_scheme(self):
        scheme = self.metadata['scheme']

        return scheme.A, scheme.B, scheme.U, scheme.V, scheme.num_stages, scheme.num_values
