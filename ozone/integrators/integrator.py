from __future__ import division

import numpy as np
from openmdao.api import Group, IndepVarComp
from six import iteritems

import ozone.schemes.scheme as schemes
from ozone.components.time_comp import TimeComp
from ozone.components.starting_comp import StartingComp
from ozone.components.parameter_comp import ParameterComp
from ozone.schemes.scheme import GLMScheme
from ozone.schemes.runge_kutta import RK4
from ozone.ode_function import ODEFunction
from ozone.utils.var_names import get_name
from ozone.utils.misc import get_scheme


class Integrator(Group):
    """
    The base class for all integration schemes.
    """

    def initialize(self):
        self.metadata.declare('ode_function', type_=ODEFunction, required=True)
        self.metadata.declare('times', type_=np.ndarray, required=True)
        self.metadata.declare('initial_conditions', type_=dict)
        self.metadata.declare('parameters', type_=dict)
        self.metadata.declare('scheme', default=RK4(), type_=GLMScheme)
        self.metadata.declare('starting_coeffs', type_=(np.ndarray, type(None)))

    def setup(self):
        ode_function = self.metadata['ode_function']
        initial_conditions = self.metadata['initial_conditions']
        given_parameters = self.metadata['parameters']
        starting_coeffs = self.metadata['starting_coeffs']
        scheme = self.metadata['scheme']

        num_step_vars = scheme.num_values

        has_starting_method = scheme.starting_method is not None
        is_starting_method = starting_coeffs is not None

        states, time_units, starting_times, my_times = self._get_meta()
        parameters = ode_function._parameters

        # (num_starting, num_time_steps, num_step_vars,)
        if is_starting_method:
            assert len(starting_coeffs.shape) == 3, \
                'starting_coeffs must be a rank-3 array, but its rank is %i' \
                % len(starting_coeffs.shape)
            assert starting_coeffs.shape[1:] == (len(my_times), scheme.num_values), \
                'starting_coeffs must have shape (num_starting, num_time_steps, num_step_vars,).' \
                + 'It has shape %i x %i x %i, but it should have shape (? x %i x %i)' % (
                    starting_coeffs.shape[0], starting_coeffs.shape[1], starting_coeffs.shape[2],
                    len(my_times), scheme.num_values
                )

        promotes_ICs = []
        for state_name in states:
            IC_name = get_name('IC', state_name)
            promotes_ICs.append(IC_name)

        # Initial conditions
        if initial_conditions is not None:
            ivcomp = IndepVarComp()

            for state_name, value in iteritems(initial_conditions):
                IC_name = get_name('IC', state_name)
                state = ode_function._states[state_name]
                ivcomp.add_output(IC_name, val=value, units=state['units'])

            self.add_subsystem('initial_conditions', ivcomp, promotes_outputs=promotes_ICs)

        # Given parameters
        if given_parameters is not None:
            comp = IndepVarComp()

            promotes = []
            for parameter_name, value in iteritems(given_parameters):
                parameter = ode_function._parameters[parameter_name]
                name = get_name('parameter', parameter_name)
                comp.add_output(name, val=value, units=parameter['units'])
                promotes.append(name)

            self.add_subsystem('inputs_p', comp, promotes_outputs=promotes)

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

        # Parameter comp
        if len(parameters) > 1:
            abscissa_times = self._get_abscissa_times()
            promotes_parameters = []
            for parameter_name, value in iteritems(parameters):
                old_name = get_name('in', parameter_name)
                new_name = get_name('parameter', parameter_name)
                promotes_parameters.append((old_name, new_name))
            self.add_subsystem('parameter_comp',
                ParameterComp(parameters=parameters, times=my_times, parameter_times=abscissa_times),
                promotes_inputs=promotes_parameters)

        # Starting method
        if not has_starting_method:
            starting_system = StartingComp(states=states, num_step_vars=num_step_vars)
        else:
            starting_scheme_name, starting_coeffs, starting_time_steps = scheme.starting_method
            scheme = get_scheme(starting_scheme_name)

            starting_system = self.__class__(
                ode_function=ode_function, times=starting_times, scheme=scheme,
                starting_coeffs=starting_coeffs,
            )

        self.add_subsystem('starting_system', starting_system,
            promotes_inputs=promotes_ICs)

    def _get_names(self, comp, type_, i_step=None, i_stage=None, j_stage=None):
        names_list = []
        for state_name, state in iteritems(self.metadata['ode_function']._states):
            if type_ == 'rate_path':
                names = '{}.{}'.format(comp, state['rate_path'])
            elif type_ == 'paths':
                names = ['{}.{}'.format(comp, tgt) for tgt in state['paths']]
            else:
                names = '{}.{}'.format(comp, get_name(
                    type_, state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage))

            names_list.append(names)

        return names_list

    def _get_parameter_names(self, comp, type_, i_step=None, i_stage=None, j_stage=None):
        names_list = []
        for parameter_name, parameter in iteritems(self.metadata['ode_function']._parameters):
            if type_ == 'paths':
                names = ['{}.{}'.format(comp, tgt) for tgt in parameter['paths']]
            else:
                names = '{}.{}'.format(comp, get_name(
                    type_, parameter_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage))

            names_list.append(names)

        return names_list

    def _connect_multiple(self, srcs_list, tgts_list, src_indices_list=None):
        if src_indices_list is None:
            for srcs, tgts in zip(srcs_list, tgts_list):
                self.connect(srcs, tgts)
        else:
            for srcs, tgts, src_indices in zip(srcs_list, tgts_list, src_indices_list):
                self.connect(srcs, tgts, src_indices=src_indices)

    def _create_ode(self, num):
        ode_function = self.metadata['ode_function']
        return ode_function._system_class(num=num, **ode_function._system_init_kwargs)

    def _get_meta(self):
        ode_function = self.metadata['ode_function']
        scheme = self.metadata['scheme']
        times = self.metadata['times']

        has_starting_method = scheme.starting_method is not None

        if has_starting_method:
            start_time_index = scheme.starting_method[2]
        else:
            start_time_index = 0

        states = ode_function._states
        time_units = ode_function._time_options['units']

        return states, time_units, times[:start_time_index+1], times[start_time_index:]

    def _get_scheme(self):
        scheme = self.metadata['scheme']

        return scheme.A, scheme.B, scheme.U, scheme.V, scheme.num_stages, scheme.num_values

    def _get_abscissa_times(self):
        states, time_units, starting_times, my_times = self._get_meta()
        abscissa = self.metadata['scheme'].abscissa

        repeated_times1 = np.repeat(my_times[:-1], len(abscissa))
        repeated_times2 = np.repeat(my_times[1:], len(abscissa))
        tiled_abscissa = np.tile(abscissa, len(my_times) - 1)

        abscissa_times = repeated_times1 + (repeated_times2 - repeated_times1) * tiled_abscissa

        return abscissa_times
