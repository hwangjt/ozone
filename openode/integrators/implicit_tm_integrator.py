import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, NewtonSolver, DirectSolver, DenseJacobian

from openode.integrators.integrator import Integrator
from openode.components.starting_comp import StartingComp
from openode.components.implicit_tm_stage_comp import ImplicitTMStageComp
from openode.components.implicit_tm_step_comp import ImplicitTMStepComp
from openode.components.tm_output_comp import TMOutputComp
from openode.utils.var_names import get_y_new_name


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
                glm_A=glm_A, glm_U=glm_U,
            )
            group.add_subsystem('stage_comp', comp)
            self.connect('time_comp.h_vec', group_new_name + '.stage_comp.h', src_indices=i_step)

            comp = ImplicitTMStepComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_B=glm_B, glm_V=glm_V,
            )
            group.add_subsystem('step_comp', comp)
            self.connect('time_comp.h_vec', group_new_name + '.step_comp.h', src_indices=i_step)

            for state_name, state in iteritems(self.metadata['ode_function']._states):
                tgt = state['rate_target']
                self.connect(
                    group_new_name + '.ode_comp.%s' % (tgt),
                    group_new_name + '.step_comp.%s' % ('F:%s' % state_name),
                )
                self.connect(
                    group_new_name + '.ode_comp.%s' % (tgt),
                    group_new_name + '.stage_comp.%s' % ('F:%s' % state_name),
                )

                for tgt in state['state_targets']:
                    self.connect(
                        group_new_name + '.%s.%s' % ('stage_comp', 'Y:%s' % state_name),
                        group_new_name + '.%s.%s' % ('ode_comp', tgt),
                    )

            if i_step == 0:
                self._connect_states(
                    'starting_comp', 'y_new_name',
                    group_new_name + '.step_comp', 'y_old_name')
                self._connect_states(
                    'starting_comp', 'y_new_name',
                    group_new_name + '.stage_comp', 'y_name')
            else:
                self._connect_states(
                    group_old_name + '.step_comp', 'y_new_name',
                    group_new_name + '.step_comp', 'y_old_name')
                self._connect_states(
                    group_old_name + '.step_comp', 'y_new_name',
                    group_new_name + '.stage_comp', 'y_name')

            group.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
            group.linear_solver = DirectSolver()
            group.jacobian = DenseJacobian()

        comp = TMOutputComp(states=states, times=times)
        self.add_subsystem('output_comp', comp)

        for i_step in range(len(times)):
            if i_step == 0:
                step_comp_name = 'starting_comp'
            else:
                step_comp_name = 'step_%i' % (i_step - 1) + '.step_comp'

            self._connect_states(
                step_comp_name, 'y_new_name',
                'output_comp', 'step_name',
                tgt_step=i_step)
