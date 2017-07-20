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
from openode.utils.misc import get_scheme


class Integrator(Group):
    """
    The base class for all integration schemes.
    """

    def initialize(self):
        self.metadata.declare('ode_function', type_=ODEFunction, required=True)
        self.metadata.declare('times', type_=np.ndarray, required=True)
        self.metadata.declare('initial_conditions', type_=dict)
        self.metadata.declare('scheme', default=RK4(), type_=GLMScheme)
        self.metadata.declare('starting_coeffs', values=(None,), type_=np.ndarray)

    def setup(self):
        ode_function = self.metadata['ode_function']
        initial_conditions = self.metadata['initial_conditions']
        starting_coeffs = self.metadata['starting_coeffs']
        scheme = self.metadata['scheme']

        states, time_units, starting_times, my_times = self._get_meta()

        # Ensure that all initial_conditions are valid
        if initial_conditions is not None:
            for state_name, value in iteritems(initial_conditions):
                assert state_name in ode_function._states, \
                    'State name %s is not valid in the initial conditions' % state_name

                if np.isscalar(value):
                    assert ode_function._states[state_name]['shape'] == (1,), \
                        'The initial condition for state %s has the wrong shape' % state_name
                else:
                    assert ode_function._states[state_name]['shape'] == value.shape, \
                        'The initial condition for state %s has the wrong shape' % state_name

        # (num_starting, num_time_steps, num_step_vars,)
        if starting_coeffs is not None:
            assert len(starting_coeffs.shape) == 3, \
                'starting_coeffs must be a rank-3 array, but its rank is %i' \
                % len(starting_coeffs.shape)
            assert starting_coeffs.shape[1:] == (len(my_times), scheme.num_values), \
                'starting_coeffs must have shape (num_starting, num_time_steps, num_step_vars,).' \
                + 'It has shape %i x %i x %i, but it should have shape (? x %i x %i)' % (
                    starting_coeffs.shape[0], starting_coeffs.shape[1], starting_coeffs.shape[2],
                    len(my_times), scheme.num_values
                )

        # Initial conditions
        if initial_conditions is not None:
            ivcomp = IndepVarComp()

            promotes = []
            for state_name in states:
                promotes.append((state_name, 'IC:{}'.format(state_name)))

            for state_name, value in iteritems(initial_conditions):
                state = ode_function._states[state_name]
                ivcomp.add_output(state_name, val=value, units=state['units'])
            self.add_subsystem('initial_conditions', ivcomp, promotes_outputs=promotes)

        # Time values just at the time steps
        comp = IndepVarComp()
        comp.add_output('times', val=my_times, units=time_units)
        self.add_subsystem('inputs_t', comp)

        # Time comp
        abscissa = self.metadata['scheme'].abscissa
        self.add_subsystem('time_comp',
            TimeComp(time_units=time_units, glm_abscissa=abscissa,
                num_time_steps=len(my_times)))
        self.connect('inputs_t.times', 'time_comp.times')

        # Starting method
        promotes_inputs = []
        for state_name in states:
            promotes_inputs.append('IC:{}'.format(state_name))
        if scheme.starting_method is None:
            self.add_subsystem('starting_system', StartingComp(states=states),
                promotes_inputs=promotes_inputs)
        else:
            starting_scheme_name, starting_coeffs, starting_time_steps = scheme.starting_method
            scheme_class = get_scheme(starting_scheme_name)
            num_starting = starting_coeffs.shape[0]

            starting_integrator = self.__class__(
                ode_function=ode_function, times=starting_times, scheme=scheme_class(),
                starting_coeffs=starting_coeffs,
            )
            self.add_subsystem('starting_system', starting_integrator,
                promotes_inputs=promotes_inputs)

    def _get_names(self, comp, type_, i_step=None, i_stage=None, j_stage=None):
        names_list = []
        for state_name, state in iteritems(self.metadata['ode_function']._states):
            if comp is not None:
                prefix = comp + '.'
            else:
                prefix = ''
            if type_ == 'rate_target':
                names = '{}{}'.format(prefix, state['rate_target'])
            elif type_ == 'state_targets':
                names = ['{}{}'.format(prefix, tgt) for tgt in state['state_targets']]
            else:
                names = '{}{}'.format(prefix, get_name(
                    type_, state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage))

            names_list.append(names)

        return names_list

    def _connect_states(self, srcs_list, tgts_list):
        for srcs, tgts in zip(srcs_list, tgts_list):
            self.connect(srcs, tgts)

    def _create_ode(self, num):
        ode_function = self.metadata['ode_function']
        return ode_function._system_class(num=num, **ode_function._system_init_kwargs)

    def _get_meta(self):
        ode_function = self.metadata['ode_function']
        scheme = self.metadata['scheme']

        times = self.metadata['times']

        if scheme.starting_method is not None:
            start_time_index = scheme.starting_method[2]
        else:
            start_time_index = 0

        states = ode_function._states
        time_units = ode_function._time_options['units']

        return states, time_units, times[:start_time_index+1], times[start_time_index:]

    def _get_scheme(self):
        scheme = self.metadata['scheme']

        return scheme.A, scheme.B, scheme.U, scheme.V, scheme.num_stages, scheme.num_values
