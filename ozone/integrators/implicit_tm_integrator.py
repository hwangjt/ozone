import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, NewtonSolver, DirectSolver, DenseJacobian

from ozone.integrators.integrator import Integrator
from ozone.components.starting_comp import StartingComp
from ozone.components.implicit_tm_stage_comp import ImplicitTMStageComp
from ozone.components.implicit_tm_step_comp import ImplicitTMStepComp
from ozone.components.tm_output_comp import TMOutputComp
from ozone.utils.var_names import get_name


class ImplicitTMIntegrator(Integrator):
    """
    Integrate an implicit scheme with a time-marching approach.
    """

    def setup(self):
        super(ImplicitTMIntegrator, self).setup()

        states, time_units, starting_times, my_times = self._get_meta()
        scheme = self.metadata['scheme']
        glm_A = scheme.A
        glm_B = scheme.B
        glm_U = scheme.U
        glm_V = scheme.V
        num_stages = scheme.num_stages
        num_step_vars = scheme.num_values
        starting_coeffs = self.metadata['starting_coeffs']
        ode_function =  self.metadata['ode_function']
        parameters = ode_function._parameters

        has_starting_method = scheme.starting_method is not None
        is_starting_method = starting_coeffs is not None

        for i_step in range(len(my_times) - 1):
            group = Group()
            group_old_name = 'step_%i' % (i_step - 1)
            group_new_name = 'step_%i' % i_step
            self.add_subsystem(group_new_name, group)

            comp = self._create_ode(num_stages)
            group.add_subsystem('ode_comp', comp)
            self.connect('time_comp.abscissa_times',
                ['.'.join((group_new_name + '.ode_comp', t)) for t in
                ode_function._time_options['paths']],
                src_indices=i_step * (num_stages) + np.arange(num_stages))

            if len(parameters) > 1:
                src_indices_list = []
                for parameter_name, value in iteritems(parameters):
                    size = np.prod(value['shape'])
                    shape = value['shape']

                    arange = np.arange(((len(my_times) - 1) * num_stages * size)).reshape(
                        ((len(my_times) - 1, num_stages,) + shape))
                    src_indices = arange[i_step, :, :]
                    src_indices_list.append(src_indices)
                self._connect_multiple(
                    self._get_parameter_names('parameter_comp', 'out'),
                    self._get_parameter_names(group_new_name + '.ode_comp', 'paths'),
                    src_indices_list,
                )

            comp = ImplicitTMStageComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_A=glm_A, glm_U=glm_U, i_step=i_step,
            )
            group.add_subsystem('stage_comp', comp)
            self.connect('time_comp.h_vec', group_new_name + '.stage_comp.h', src_indices=i_step)

            comp = ImplicitTMStepComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_B=glm_B, glm_V=glm_V, i_step=i_step,
            )
            group.add_subsystem('step_comp', comp)
            self.connect('time_comp.h_vec', group_new_name + '.step_comp.h', src_indices=i_step)

            self._connect_multiple(
                self._get_names(group_new_name + '.ode_comp', 'rate_path'),
                self._get_names(group_new_name + '.step_comp', 'F', i_step=i_step),
            )

            self._connect_multiple(
                self._get_names(group_new_name + '.ode_comp', 'rate_path'),
                self._get_names(group_new_name + '.stage_comp', 'F', i_step=i_step),
            )

            self._connect_multiple(
                self._get_names(group_new_name + '.stage_comp', 'Y', i_step=i_step),
                self._get_names(group_new_name + '.ode_comp', 'paths'),
            )

            if i_step == 0:
                self._connect_multiple(
                    self._get_names('starting_system', 'starting'),
                    self._get_names(group_new_name + '.step_comp', 'y_old', i_step=i_step),
                )
                self._connect_multiple(
                    self._get_names('starting_system', 'starting'),
                    self._get_names(group_new_name + '.stage_comp', 'y_old', i_step=i_step),
                )
            else:
                self._connect_multiple(
                    self._get_names(group_old_name + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_names(group_new_name + '.step_comp', 'y_old', i_step=i_step),
                )
                self._connect_multiple(
                    self._get_names(group_old_name + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_names(group_new_name + '.stage_comp', 'y_old', i_step=i_step),
                )

            group.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
            group.linear_solver = DirectSolver()
            group.jacobian = DenseJacobian()

        promotes_outputs = []
        for state_name in states:
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)
            promotes_outputs.append(out_state_name)
            if is_starting_method:
                promotes_outputs.append(starting_name)

        comp = TMOutputComp(
            states=states, num_starting_time_steps=len(starting_times),
            num_my_time_steps=len(my_times), num_step_vars=num_step_vars,
            starting_coeffs=starting_coeffs)
        self.add_subsystem('output_comp', comp, promotes_outputs=promotes_outputs)
        if has_starting_method:
            self._connect_multiple(
                self._get_names('starting_system', 'state'),
                self._get_names('output_comp', 'starting_state'),
            )

        for i_step in range(len(my_times)):
            if i_step == 0:
                self._connect_multiple(
                    self._get_names('starting_system', 'starting'),
                    self._get_names('output_comp', 'y', i_step=i_step),
                )
            else:
                self._connect_multiple(
                    self._get_names('step_%i' % (i_step - 1) + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_names('output_comp', 'y', i_step=i_step),
                )
