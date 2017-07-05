import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp

from openode.integrators.integrator import Integrator
from openode.components.starting_comp import StartingComp
from openode.components.stage_comp import StageComp
from openode.components.step_comp import StepComp
from openode.components.output_comp import OutputComp


class ExplicitRelaxedIntegrator(Integrator):
    """
    Integrate an explicit scheme with a relaxed time-marching approach.
    """

    def setup(self):
        super(ExplicitRelaxedIntegrator, self).setup()

        states, time_units, time_spacing = self._get_meta()
        glm_A, glm_B, glm_U, glm_V, num_stages, num_step_vars = self._get_scheme()

        # Starting method
        self.add_subsystem('starting_comp', StartingComp(states=states))
        self._connect_states(
            'initial_conditions', 'state_name',
            'starting_comp', 'state_name')

        comp = self._create_ode(num_stages * (len(time_spacing) - 1))
        self.add_subsystem('ode_comp', comp)

        comp = IntegrationComp()
        self.add_subsystem('int_comp', comp)

        for state_name, state in iteritems(states):
            size = np.prod(state['shape'])

            src_indices = np.zeros(size * (len(time_spacing) - 1) * num_stages, int)
            arange = np.arange(size * (len(time_spacing) - 1) * (num_stages + num_step_vars))
            for ind in range(num_stages):
                src_indices[ind::num_stages] = arange[ind::num_stages + num_step_vars]

            self.connect(
                'int_comp.%s' % state_name,
                'ode_comp.%s' % state_name,
                src_indices=src_indices,
            )





        return

        for i_step in range(len(time_spacing) - 1):

            step_comp_old_name = 'step_comp_%i' % (i_step - 1)
            step_comp_new_name = 'step_comp_%i' % (i_step)

            for i_stage in range(num_stages):
                stage_comp_name = 'stage_comp_%i_%i' % (i_step, i_stage)
                ode_comp_name = 'ode_comp_%i_%i' % (i_step, i_stage)

                comp = StageComp(
                    states=states, time_units=time_units,
                    num_stages=num_stages, num_step_vars=num_step_vars,
                    glm_A=glm_A, glm_U=glm_U, i_stage=i_stage,
                )
                self.add_subsystem(stage_comp_name, comp)
                self.connect('time_comp.h_vec', '%s.h' % stage_comp_name, src_indices=i_step)

                for j_stage in range(i_stage):
                    ode_comp_tmp_name = 'ode_comp_%i_%i' % (i_step, j_stage)
                    self._connect_states(
                        ode_comp_tmp_name, 'rate_target',
                        stage_comp_name, 'F_name',
                        tgt_stage=j_stage)

                if i_step == 0:
                    self._connect_states(
                        'starting_comp', 'y_new_name',
                        stage_comp_name, 'y_old_name')
                else:
                    self._connect_states(
                        step_comp_old_name, 'y_new_name',
                        stage_comp_name, 'y_old_name')

                comp = self._create_ode(1)
                self.add_subsystem(ode_comp_name, comp)
                self._connect_states(
                    stage_comp_name, 'Y_name',
                    ode_comp_name, 'state_targets',
                    src_stage=i_stage)

            comp = StepComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_B=glm_B, glm_V=glm_V,
            )
            self.add_subsystem(step_comp_new_name, comp)
            self.connect('time_comp.h_vec', '%s.h' % step_comp_new_name, src_indices=i_step)
            for j_stage in range(num_stages):
                ode_comp_tmp_name = 'ode_comp_%i_%i' % (i_step, j_stage)
                self._connect_states(
                    ode_comp_tmp_name, 'rate_target',
                    step_comp_new_name, 'F_name',
                    tgt_stage=j_stage)

            if i_step == 0:
                self._connect_states(
                    'starting_comp', 'y_new_name',
                    step_comp_new_name, 'y_old_name')
            else:
                self._connect_states(
                    step_comp_old_name, 'y_new_name',
                    step_comp_new_name, 'y_old_name')

        comp = OutputComp(states=states, time_spacing=time_spacing)
        self.add_subsystem('output_comp', comp)

        for i_step in range(len(time_spacing)):
            if i_step == 0:
                step_comp_name = 'starting_comp'
            else:
                step_comp_name = 'step_comp_%i' % (i_step - 1)

            self._connect_states(
                step_comp_name, 'y_new_name',
                'output_comp', 'step_name',
                tgt_step=i_step)
