import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, NewtonSolver, ScipyIterativeSolver, DirectSolver

from openode.integrators.integrator import Integrator
from openode.components.vectorized_step_comp import VectorizedStepComp
from openode.components.vectorized_stage_comp import VectorizedStageComp


class ExplicitRelaxedIntegrator(Integrator):
    """
    Integrate an explicit scheme with a relaxed time-marching approach.
    """

    def setup(self):
        super(ExplicitRelaxedIntegrator, self).setup()

        states, time_units, time_spacing = self._get_meta()
        glm_A, glm_B, glm_U, glm_V, num_stages, num_step_vars = self._get_scheme()

        num_time_steps = len(time_spacing)

        comp = self._create_ode((num_time_steps - 1) * num_stages)
        self.add_subsystem('ode_comp', comp)

        comp = VectorizedStepComp(states=states, time_units=time_units,
            num_time_steps=num_time_steps, num_stages=num_stages, num_step_vars=num_step_vars,
            glm_B=glm_B, glm_V=glm_V,
        )
        self.add_subsystem('vectorized_step_comp', comp)
        self.connect('time_comp.h_vec', 'vectorized_step_comp.h_vec')
        self._connect_states(
            'starting_comp', 'y_new_name',
            'vectorized_step_comp', 'y0')

        comp = VectorizedStageComp(states=states, time_units=time_units,
            num_time_steps=num_time_steps, num_stages=num_stages, num_step_vars=num_step_vars,
            glm_A=glm_A, glm_U=glm_U,
        )
        self.add_subsystem('vectorized_stage_comp', comp)
        self.connect('time_comp.h_vec', 'vectorized_stage_comp.h_vec')

        for state_name, state in iteritems(states):
            size = np.prod(state['shape'])
            shape = state['shape']

            src_indices = np.arange((num_time_steps - 1) * num_stages * size)
            self.connect(
                'ode_comp.%s' % state['rate_target'],
                'vectorized_step_comp.F:%s' % state_name,
                src_indices=src_indices,
            )
            self.connect(
                'ode_comp.%s' % state['rate_target'],
                'vectorized_stage_comp.F:%s' % state_name,
                src_indices=src_indices,
            )

            self.connect(
                'vectorized_step_comp.y:%s' % state_name,
                'vectorized_stage_comp.y:%s' % state_name,
            )

            src_indices = np.arange((num_time_steps - 1) * num_stages * size).reshape(
                ((num_time_steps - 1) * num_stages,) + shape)
            self.connect(
                'vectorized_stage_comp.Y:%s' % state_name,
                ['ode_comp.%s' % tgt for tgt in state['state_targets']],
                # src_indices=src_indices,
            )

        self.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
        self.linear_solver = ScipyIterativeSolver(iprint=2)
        self.linear_solver = DirectSolver()
