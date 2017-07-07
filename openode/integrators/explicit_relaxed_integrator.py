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

    def initialize(self):
        super(ExplicitRelaxedIntegrator, self).initialize()

        self.metadata.declare('formulation', default='MDF', values=['MDF', 'SAND'])

    def setup(self):
        super(ExplicitRelaxedIntegrator, self).setup()

        formulation = self.metadata['formulation']
        ode = self.metadata['ode']

        states, time_units, time_spacing = self._get_meta()
        glm_A, glm_B, glm_U, glm_V, num_stages, num_step_vars = self._get_scheme()

        num_time_steps = len(time_spacing)

        if formulation == 'SAND':
            comp = IndepVarComp()
            for state_name, state in iteritems(states):
                comp.add_output('Y:%s' % state_name,
                    shape=(num_time_steps - 1, num_stages,) + state['shape'],
                    units=state['units'])
                comp.add_design_var('Y:%s' % state_name)
            self.add_subsystem('desvars_comp', comp)

        comp = self._create_ode((num_time_steps - 1) * num_stages)
        self.add_subsystem('ode_comp', comp)
        self.connect(
            'time_comp.abscissa_times',
            ['.'.join(('ode_comp', t)) for t in ode._time_options['targets']],
        )

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
            if formulation == 'MDF':
                self.connect(
                    'vectorized_stage_comp.Y_out:%s' % state_name,
                    ['ode_comp.%s' % tgt for tgt in state['state_targets']],
                )
            elif formulation == 'SAND':
                self.connect(
                    'desvars_comp.Y:%s' % state_name,
                    ['ode_comp.%s' % tgt for tgt in state['state_targets']],
                )
                self.connect(
                    'desvars_comp.Y:%s' % state_name,
                    'vectorized_stage_comp.Y_in:%s' % state_name,
                )
                self.add_constraint('vectorized_stage_comp.Y_out:%s' % state_name, equals=0.)

        if formulation == 'MDF':
            self.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
            self.linear_solver = DirectSolver()
