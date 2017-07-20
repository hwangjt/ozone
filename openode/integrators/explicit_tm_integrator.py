import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp

from openode.integrators.integrator import Integrator
from openode.components.starting_comp import StartingComp
from openode.components.explicit_tm_stage_comp import ExplicitTMStageComp
from openode.components.explicit_tm_step_comp import ExplicitTMStepComp
from openode.components.tm_output_comp import TMOutputComp


class ExplicitTMIntegrator(Integrator):
    """
    Integrate an explicit scheme with a time-marching approach.
    """

    def setup(self):
        super(ExplicitTMIntegrator, self).setup()

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

            step_comp_old_name = 'step_comp_%i' % (i_step - 1)
            step_comp_new_name = 'step_comp_%i' % (i_step)

            for i_stage in range(num_stages):
                stage_comp_name = 'stage_comp_%i_%i' % (i_step, i_stage)
                ode_comp_name = 'ode_comp_%i_%i' % (i_step, i_stage)

                self.connect('time_comp.abscissa_times',
                             ['.'.join((ode_comp_name, t)) for t in
                              ode_function._time_options['targets']],
                             src_indices=i_step * (num_stages) + i_stage
                             )

                comp = ExplicitTMStageComp(
                    states=states, time_units=time_units,
                    num_stages=num_stages, num_step_vars=num_step_vars,
                    glm_A=glm_A, glm_U=glm_U, i_stage=i_stage, i_step=i_step,
                )
                self.add_subsystem(stage_comp_name, comp)
                self.connect('time_comp.h_vec', '%s.h' % stage_comp_name, src_indices=i_step)

                for j_stage in range(i_stage):
                    ode_comp_tmp_name = 'ode_comp_%i_%i' % (i_step, j_stage)
                    self._connect_states(
                        self._get_names(ode_comp_tmp_name, 'rate_target'),
                        self._get_names(stage_comp_name, 'F', i_step=i_step, i_stage=i_stage, j_stage=j_stage),
                    )

                comp = self._create_ode(1)
                self.add_subsystem(ode_comp_name, comp)
                self._connect_states(
                    self._get_names(stage_comp_name, 'Y', i_step=i_step, i_stage=i_stage),
                    self._get_names(ode_comp_name, 'state_targets'),
                )

            comp = ExplicitTMStepComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_B=glm_B, glm_V=glm_V, i_step=i_step,
            )
            self.add_subsystem(step_comp_new_name, comp)
            self.connect('time_comp.h_vec', '%s.h' % step_comp_new_name, src_indices=i_step)
            for j_stage in range(num_stages):
                ode_comp_tmp_name = 'ode_comp_%i_%i' % (i_step, j_stage)
                self._connect_states(
                    self._get_names(ode_comp_tmp_name, 'rate_target'),
                    self._get_names(step_comp_new_name, 'F', i_step=i_step, j_stage=j_stage),
                )

            if i_step == 0:
                self._connect_states(
                    self._get_names('starting_comp', 'y_new'),
                    self._get_names(step_comp_new_name, 'y_old', i_step=i_step),
                )
                for i_stage in range(num_stages):
                    stage_comp_name = 'stage_comp_%i_%i' % (i_step, i_stage)
                    self._connect_states(
                        self._get_names('starting_comp', 'y_new'),
                        self._get_names(stage_comp_name, 'y_old', i_step=i_step, i_stage=i_stage),
                    )
            else:
                self._connect_states(
                    self._get_names(step_comp_old_name, 'y_new', i_step=i_step - 1),
                    self._get_names(step_comp_new_name, 'y_old', i_step=i_step),
                )
                for i_stage in range(num_stages):
                    stage_comp_name = 'stage_comp_%i_%i' % (i_step, i_stage)
                    self._connect_states(
                        self._get_names(step_comp_old_name, 'y_new', i_step=i_step - 1),
                        self._get_names(stage_comp_name, 'y_old', i_step=i_step, i_stage=i_stage),
                    )

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
                    self._get_names('step_comp_%i' % (i_step - 1), 'y_new', i_step=i_step - 1),
                    self._get_names('output_comp', 'y', i_step=i_step),
                )
