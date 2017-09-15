Projectile dynamics ODE
=======================

1. ODE system
-------------

.. code-block:: python

  import numpy as np
  
  from openmdao.api import ExplicitComponent
  
  
  class ProjectileSystem(ExplicitComponent):
  
      def initialize(self):
          self.metadata.declare('num_nodes', default=1, type_=int)
  
          self.g = -9.81
  
      def setup(self):
          num = self.metadata['num_nodes']
  
          self.add_input('vx', shape=(num, 1))
          self.add_input('vy', shape=(num, 1))
  
          self.add_output('dx_dt', shape=(num, 1))
          self.add_output('dy_dt', shape=(num, 1))
          self.add_output('dvx_dt', shape=(num, 1))
          self.add_output('dvy_dt', shape=(num, 1))
  
          self.declare_partials('*', '*', dependent=False)
  
          self.declare_partials('dx_dt', 'vx', val=1., rows=np.arange(num), cols=np.arange(num))
          self.declare_partials('dy_dt', 'vy', val=1., rows=np.arange(num), cols=np.arange(num))
  
      def compute(self, inputs, outputs):
          outputs['dx_dt'] = inputs['vx']
          outputs['dy_dt'] = inputs['vy']
          outputs['dvx_dt'] = 0.
          outputs['dvy_dt'] = self.g
  

2. ODEFunction
--------------

.. code-block:: python

  import numpy as np
  import time
  
  from ozone.api import ODEFunction
  from ozone.tests.ode_function_library.projectile_dynamics_sys import ProjectileSystem
  
  
  class ProjectileFunction(ODEFunction):
  
      def initialize(self, system_init_kwargs=None):
          self.set_system(ProjectileSystem, system_init_kwargs)
  
          self.declare_state('x', 'dx_dt', shape=1)
          self.declare_state('y', 'dy_dt', shape=1)
          self.declare_state('vx', 'dvx_dt', shape=1, targets=['vx'])
          self.declare_state('vy', 'dvy_dt', shape=1, targets=['vy'])
  
      def get_test_parameters(self):
          t0 = 0.
          t1 = 1.
          initial_conditions = {
              'x': 0.,
              'y': 0.,
              'vx': 1.,
              'vy': 1.,
          }
          return initial_conditions, t0, t1
  
      def get_exact_solution(self, initial_conditions, t0, t):
          g = -9.81
  
          x0 = initial_conditions['x']
          y0 = initial_conditions['y']
          vx0 = initial_conditions['vx']
          vy0 = initial_conditions['vy']
  
          x = x0 + vx0 * (t - t0)
          y = y0 + vy0 * (t - t0) + 0.5 * g * (t - t0) ** 2
          vx = vx0
          vy = vy0 + g * (t - t0)
          return {'x': x, 'y': y, 'vx': vx, 'vy': vy}
  

3. Run script and output
------------------------

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from openmdao.api import Problem
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.projectile_dynamics_func import ProjectileFunction
  
  ode_function = ProjectileFunction()
  
  t0 = 0.
  t1 = 1.
  initial_conditions = {
      'x': 0.,
      'y': 0.,
      'vx': 1.,
      'vy': 1.,
  }
  
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
  
  plt.plot(prob['state:x'], prob['state:y'])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
  
::

  WARNING: Group 'integration_group' has the following cycles: [['ode_comp', 'vectorized_stagestep_comp']]
  WARNING: System 'integration_group.ode_comp' executes out-of-order with respect to its source systems ['integration_group.vectorized_stagestep_comp']
  
  =================
  integration_group
  =================
  NL: NLBGS 0 ; 217.25336 1
  NL: NLBGS 1 ; 112.709038 0.51879077
  NL: NLBGS 2 ; 0 0
  NL: NLBGS Converged
  
.. figure:: projectile_TestCase_test_doc.png
  :scale: 80 %
  :align: center
