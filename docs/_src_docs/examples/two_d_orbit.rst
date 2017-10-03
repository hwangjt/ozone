2-D orbit ODE
=============

1. ODE system
-------------

.. code-block:: python

  import numpy as np
  from scipy import sparse, linalg

  from openmdao.api import ExplicitComponent


  class TwoDOrbitSystem(ExplicitComponent):

      def initialize(self):
          self.metadata.declare('num_nodes', default=1, type_=int)

      def setup(self):
          num = self.metadata['num_nodes']

          self.add_input('position', shape=(num, 2))
          self.add_input('velocity', shape=(num, 2))
          self.add_input('t', shape=(num, 1))
          self.add_output('dpos_dt', shape=(num, 2))
          self.add_output('dvel_dt', shape=(num, 2))

          # self.declare_partials('dy_dt', 'y', val=np.eye(num))
          self.declare_partials('*', '*', dependent=False)
          self.declare_partials('dpos_dt', 'velocity', val=sparse.block_diag([np.eye(2) for _ in range(num)]))
          self.declare_partials('dvel_dt', 'position', dependent=True)

      def compute(self, inputs, outputs):
          outputs['dpos_dt'] = inputs['velocity']
          attraction = np.linalg.norm(inputs['position'], axis=-1) ** 3
          outputs['dvel_dt'] = -inputs['position'] / attraction[..., None]

      def compute_partials(self, inputs, partials):
          x, y = inputs['position'][..., 0], inputs['position'][..., 1]
          scale = (x**2 + y**2) ** (5./2.)
          jac = 1/scale * np.array([[2*x**2 - y**2, 3*x*y],
                                    [3*x*y, 2*y**2 - x**2]])
          num = self.metadata['num_nodes']
          partials['dvel_dt', 'position'] = linalg.block_diag(*(jac[..., i] for i in range(num)))


2. ODEFunction
--------------

.. code-block:: python

  import numpy as np

  from ozone.api import ODEFunction
  from ozone.tests.ode_function_library.two_d_orbit_sys import TwoDOrbitSystem


  class TwoDOrbitFunction(ODEFunction):

      def initialize(self):
          self.set_system(TwoDOrbitSystem)
          self.declare_state('position', 'dpos_dt', targets='position', shape=2)
          self.declare_state('velocity', 'dvel_dt', targets='velocity', shape=2)
          self.declare_time(targets='t')

      def get_test_parameters(self):
          ecc = 1. / 2.
          initial_conditions = {
              'position': np.array([1 - ecc, 0]),
              'velocity': np.array([0, np.sqrt((1+ecc) / (1 - ecc))])
          }
          t0 = 0. * np.pi
          t1 = 1. * np.pi
          return initial_conditions, t0, t1

      def get_exact_solution(self, initial_conditions, t0, t):
          ecc = 1 - initial_conditions['position'][0]
          return {'position': np.array([-1 - ecc, 0])}


3. Run script and output
------------------------

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from openmdao.api import Problem
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.two_d_orbit_func import TwoDOrbitFunction

  ode_function = TwoDOrbitFunction()

  ecc = 1. / 2.
  initial_conditions = {
      'position': np.array([1 - ecc, 0]),
      'velocity': np.array([0, np.sqrt((1+ecc) / (1 - ecc))])
  }
  t0 = 0. * np.pi
  t1 = 1. * np.pi

  num = 100

  times = np.linspace(t0, t1, num)

  method_name = 'RK4'
  formulation = 'solver-based'

  integrator = ODEIntegrator(ode_function, formulation, method_name,
      times=times, initial_conditions=initial_conditions,
  )

  prob = Problem(integrator)
  prob.setup()
  prob.run_model()

  plt.plot(prob['state:position'][:, 0], prob['state:position'][:, 1])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()

::


  =================
  integration_group
  =================
  NL: NLBGS 0 ; 74.6056844 1
  NL: NLBGS 1 ; 35.7616574 0.479342261
  NL: NLBGS 2 ; 15.3558738 0.205827128
  NL: NLBGS 3 ; 13.2164384 0.177150556
  NL: NLBGS 4 ; 6.04246296 0.0809919915
  NL: NLBGS 5 ; 6.49100594 0.0870041739
  NL: NLBGS 6 ; 4.12712318 0.0553191518
  NL: NLBGS 7 ; 3.32775778 0.0446046143
  NL: NLBGS 8 ; 2.15686415 0.0289101851
  NL: NLBGS 9 ; 1.66275182 0.0222872001
  NL: NLBGS 10 ; 0.993202633 0.0133126938
  NL: NLBGS 11 ; 0.607553079 0.00814352263
  NL: NLBGS 12 ; 0.302714386 0.00405752442
  NL: NLBGS 13 ; 0.145366647 0.0019484661
  NL: NLBGS 14 ; 0.0577220054 0.00077369447
  NL: NLBGS 15 ; 0.0220749799 0.000295888713
  NL: NLBGS 16 ; 0.00724063659 9.70520766e-05
  NL: NLBGS 17 ; 0.00231961036 3.10916035e-05
  NL: NLBGS 18 ; 0.000651829785 8.73699893e-06
  NL: NLBGS 19 ; 0.000180911335 2.42490015e-06
  NL: NLBGS 20 ; 4.47924274e-05 6.00388935e-07
  NL: NLBGS 21 ; 1.10239435e-05 1.47762782e-07
  NL: NLBGS 22 ; 2.44686866e-06 3.27973489e-08
  NL: NLBGS 23 ; 5.42146712e-07 7.26682848e-09
  NL: NLBGS 24 ; 1.09257963e-07 1.46447236e-09
  NL: NLBGS 25 ; 2.20438932e-08 2.95472032e-10
  NL: NLBGS 26 ; 4.0715827e-09 5.45746981e-11
  NL: NLBGS 27 ; 7.54603598e-10 1.0114559e-11
  NL: NLBGS 28 ; 1.28706974e-10 1.7251631e-12
  NL: NLBGS 29 ; 2.20646826e-11 2.9575069e-13
  NL: NLBGS Converged

.. figure:: two_d_orbit_TestCase_test_doc.png
  :scale: 80 %
  :align: center
