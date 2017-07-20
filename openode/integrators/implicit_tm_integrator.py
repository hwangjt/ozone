import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, NewtonSolver, DirectSolver, DenseJacobian

from openode.integrators.integrator import Integrator
from openode.components.starting_comp import StartingComp
from openode.components.implicit_tm_stage_comp import ImplicitTMStageComp
from openode.components.implicit_tm_step_comp import ImplicitTMStepComp
from openode.components.tm_output_comp import TMOutputComp


class ImplicitTMIntegrator(Integrator):
    """
    Integrate an implicit scheme with a time-marching approach.
    """

    def setup(self):
        super(ImplicitTMIntegrator, self).setup()

        states, time_units, times = self._get_meta()
        scheme = self.metadata['scheme']
        glm_A = scheme.A
        glm_B = scheme.B
        glm_U = scheme.U
        glm_V = scheme.V
        num_stages = scheme.num_stages
        num_step_vars = scheme.num_values
        ode_function =  self.metadata['ode_function']

        for i_step in range(len(times) - 1):
            group = Group()
            group_old_name = 'step_%i' % (i_step - 1)
            group_new_name = 'step_%i' % i_step
            self.add_subsystem(group_new_name, group)

            comp = self._create_ode(num_stages)
            group.add_subsystem('ode_comp', comp)
            self.connect('time_comp.abscissa_times',
                ['.'.join((group_new_name + '.ode_comp', t)) for t in
                ode_function._time_options['targets']],
                src_indices=i_step * (num_stages) + np.arange(num_stages))

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

            self._connect_states(
                self._get_names(group_new_name + '.ode_comp', 'rate_target'),
                self._get_names(group_new_name + '.step_comp', 'F', i_step=i_step),
            )

            self._connect_states(
                self._get_names(group_new_name + '.ode_comp', 'rate_target'),
                self._get_names(group_new_name + '.stage_comp', 'F', i_step=i_step),
            )

            self._connect_states(
                self._get_names(group_new_name + '.stage_comp', 'Y', i_step=i_step),
                self._get_names(group_new_name + '.ode_comp', 'state_targets'),
            )

            if i_step == 0:
                self._connect_states(
                    self._get_names('starting_comp', 'y_new'),
                    self._get_names(group_new_name + '.step_comp', 'y_old', i_step=i_step),
                )
                self._connect_states(
                    self._get_names('starting_comp', 'y_new'),
                    self._get_names(group_new_name + '.stage_comp', 'y_old', i_step=i_step),
                )
            else:
                self._connect_states(
                    self._get_names(group_old_name + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_names(group_new_name + '.step_comp', 'y_old', i_step=i_step),
                )
                self._connect_states(
                    self._get_names(group_old_name + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_names(group_new_name + '.stage_comp', 'y_old', i_step=i_step),
                )

            group.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
            group.linear_solver = DirectSolver()
            group.jacobian = DenseJacobian()

        comp = TMOutputComp(states=states, times=times)
        self.add_subsystem('output_comp', comp)

        for i_step in range(len(times)):
            if i_step == 0:
                self._connect_states(
                    self._get_names('starting_comp', 'y_new'),
                    self._get_names('output_comp', 'y', i_step=i_step),
                )
            else:
                self._connect_states(
                    self._get_names('step_%i' % (i_step - 1) + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_names('output_comp', 'y', i_step=i_step),
                )
