import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, NewtonSolver, DirectSolver, DenseJacobian

from ozone.integrators.integrator import Integrator
from ozone.components.vectorized_step_comp import VectorizedStepComp
from ozone.components.vectorized_stage_comp import VectorizedStageComp
from ozone.components.vectorized_output_comp import VectorizedOutputComp
from ozone.utils.var_names import get_name


class VectorizedIntegrator(Integrator):
    """
    Integrate an explicit scheme with a relaxed time-marching approach.
    """

    def initialize(self):
        super(VectorizedIntegrator, self).initialize()

        self.metadata.declare('formulation', default='MDF', values=['MDF', 'SAND'])

    def setup(self):
        super(VectorizedIntegrator, self).setup()

        coupled_group = Group()
        self.add_subsystem('coupled_group', coupled_group)

        formulation = self.metadata['formulation']
        ode_function = self.metadata['ode_function']
        starting_coeffs = self.metadata['starting_coeffs']
        scheme = self.metadata['scheme']

        has_starting_method = scheme.starting_method is not None
        is_starting_method = starting_coeffs is not None

        states, time_units, starting_times, my_times = self._get_meta()
        glm_A, glm_B, glm_U, glm_V, num_stages, num_step_vars = self._get_scheme()

        parameters = ode_function._parameters

        num_time_steps = len(my_times)

        if formulation == 'SAND':
            comp = IndepVarComp()
            for state_name, state in iteritems(states):
                comp.add_output('Y:%s' % state_name,
                    shape=(num_time_steps - 1, num_stages,) + state['shape'],
                    units=state['units'])
                comp.add_design_var('Y:%s' % state_name)
            coupled_group.add_subsystem('desvars_comp', comp)

        comp = self._create_ode((num_time_steps - 1) * num_stages)
        coupled_group.add_subsystem('ode_comp', comp)
        self.connect(
            'time_comp.abscissa_times',
            ['.'.join(('coupled_group.ode_comp', t)) for t in ode_function._time_options['paths']],
        )
        if len(parameters) > 1:
            self._connect_states(
                self._get_parameter_names('parameter_comp', 'out'),
                self._get_parameter_names('coupled_group.ode_comp', 'paths'),
            )

        comp = VectorizedStepComp(states=states, time_units=time_units,
            num_time_steps=num_time_steps, num_stages=num_stages, num_step_vars=num_step_vars,
            glm_B=glm_B, glm_V=glm_V,
        )
        coupled_group.add_subsystem('vectorized_step_comp', comp)
        self.connect('time_comp.h_vec', 'coupled_group.vectorized_step_comp.h_vec')
        self._connect_states(
            self._get_names('starting_system', 'starting'),
            self._get_names('coupled_group.vectorized_step_comp', 'y0'),
        )

        comp = VectorizedStageComp(states=states, time_units=time_units,
            num_time_steps=num_time_steps, num_stages=num_stages, num_step_vars=num_step_vars,
            glm_A=glm_A, glm_U=glm_U,
        )
        coupled_group.add_subsystem('vectorized_stage_comp', comp)
        self.connect('time_comp.h_vec', 'coupled_group.vectorized_stage_comp.h_vec')

        promotes_outputs = []
        for state_name in states:
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)
            promotes_outputs.append(out_state_name)
            if is_starting_method:
                promotes_outputs.append(starting_name)

        comp = VectorizedOutputComp(states=states,
            num_starting_time_steps=len(starting_times), num_my_time_steps=len(my_times),
            num_step_vars=num_step_vars, starting_coeffs=starting_coeffs,
        )
        self.add_subsystem('output_comp', comp, promotes_outputs=promotes_outputs)
        self._connect_states(
            self._get_names('coupled_group.vectorized_step_comp', 'y'),
            self._get_names('output_comp', 'y'),
        )
        if has_starting_method:
            self._connect_states(
                self._get_names('starting_system', 'state'),
                self._get_names('output_comp', 'starting_state'),
            )

        src_indices_to_ode = []
        src_indices_from_ode = []
        for state_name, state in iteritems(states):
            size = np.prod(state['shape'])
            shape = state['shape']

            src_indices_to_ode.append(
                np.arange((num_time_steps - 1) * num_stages * size).reshape(
                    ((num_time_steps - 1) * num_stages,) + shape ))

            src_indices_from_ode.append(
                np.arange((num_time_steps - 1) * num_stages * size).reshape(
                    (num_time_steps - 1, num_stages,) + shape ))

        self._connect_states(
            self._get_names('coupled_group.ode_comp', 'rate_path'),
            self._get_names('coupled_group.vectorized_step_comp', 'F'),
            src_indices_from_ode,
        )
        self._connect_states(
            self._get_names('coupled_group.ode_comp', 'rate_path'),
            self._get_names('coupled_group.vectorized_stage_comp', 'F'),
            src_indices_from_ode,
        )

        self._connect_states(
            self._get_names('coupled_group.vectorized_step_comp', 'y'),
            self._get_names('coupled_group.vectorized_stage_comp', 'y'),
        )

        if formulation == 'MDF':
            self._connect_states(
                self._get_names('coupled_group.vectorized_stage_comp', 'Y_out'),
                self._get_names('coupled_group.ode_comp', 'paths'),
                src_indices_to_ode,
            )
        elif formulation == 'SAND':
            self._connect_states(
                self._get_names('coupled_group.desvars_comp', 'Y'),
                self._get_names('coupled_group.ode_comp', 'paths'),
                src_indices_to_ode,
            )
            self._connect_states(
                self._get_names('coupled_group.desvars_comp', 'Y'),
                self._get_names('coupled_group.vectorized_stage_comp', 'Y_in'),
            )
            for state_name, state in iteritems(states):
                coupled_group.add_constraint('vectorized_stage_comp.Y_out:%s' % state_name, equals=0.)

        if formulation == 'MDF':
            coupled_group.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
            coupled_group.linear_solver = DirectSolver()
            coupled_group.jacobian = DenseJacobian()
