Simple homogeneous ODE
======================

1. ODE system
-------------

.. code-block:: python

  import numpy as np
  
  from openmdao.api import ExplicitComponent
  
  
  class SimpleHomogeneousODESystem(ExplicitComponent):
  
      def initialize(self):
          self.metadata.declare('num_nodes', default=1, type_=int)
          self.metadata.declare('a', default=1., type_=(int, float))
  
      def setup(self):
          num = self.metadata['num_nodes']
  
          self.add_input('y', shape=(num, 1))
          self.add_input('t', shape=num)
          self.add_output('dy_dt', shape=(num, 1))
  
          self.declare_partials('dy_dt', 'y', val=self.metadata['a'] * np.eye(num))
  
          self.eye = np.eye(num)
  
      def compute(self, inputs, outputs):
          outputs['dy_dt'] = self.metadata['a'] * inputs['y']
  

2. ODEFunction
--------------

.. code-block:: python

  import numpy as np
  
  from ozone.api import ODEFunction
  from ozone.tests.ode_function_library.simple_homogeneous_sys import SimpleHomogeneousODESystem
  
  
  class SimpleHomogeneousODEFunction(ODEFunction):
  
      def initialize(self, system_init_kwargs=None):
          self.set_system(SimpleHomogeneousODESystem, system_init_kwargs=system_init_kwargs)
          self.declare_state('y', 'dy_dt', targets='y')
          self.declare_time(targets='t')
  
      def get_test_parameters(self):
          t0 = 0.
          t1 = 1.
          initial_conditions = {'y': 1.}
          return initial_conditions, t0, t1
  
      def get_exact_solution(self, initial_conditions, t0, t):
          a = 1.0 if 'a' not in self._system_init_kwargs else self._system_init_kwargs['a']
          y0 = initial_conditions['y']
          C = y0 / np.exp(a * t0)
          return {'y': C * np.exp(a * t)}
  

3. Run script and output
------------------------

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from openmdao.api import Problem
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.simple_homogeneous_func import \
      SimpleHomogeneousODEFunction
  
  ode_function = SimpleHomogeneousODEFunction()
  
  t0 = 0.
  t1 = 1.
  initial_conditions = {'y': 1.}
  
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
  
  plt.plot(prob['times'], prob['state:y'])
  plt.xlabel('time (s)')
  plt.ylabel('y')
  plt.show()
  
::

  
  =================
  integration_group
  =================
  NL: NLBGS 0 ; 11.4891986 1
  NL: NLBGS 1 ; 11.4891986 1
  NL: NLBGS 2 ; 4.44981368 0.387304098
  NL: NLBGS 3 ; 1.25362245 0.109113133
  NL: NLBGS 4 ; 0.27640666 0.0240579584
  NL: NLBGS 5 ; 0.0500058905 0.00435242635
  NL: NLBGS 6 ; 0.00766683522 0.000667308097
  NL: NLBGS 7 ; 0.00101969308 8.87523244e-05
  NL: NLBGS 8 ; 0.00011973738 1.04217348e-05
  NL: NLBGS 9 ; 1.258542e-05 1.09541322e-06
  NL: NLBGS 10 ; 1.19721272e-06 1.04203327e-07
  NL: NLBGS 11 ; 1.04007429e-07 9.052627e-09
  NL: NLBGS 12 ; 8.31420102e-09 7.23653698e-10
  NL: NLBGS 13 ; 6.15473577e-10 5.35697572e-11
  NL: NLBGS 14 ; 4.24249932e-11 3.69259814e-12
  NL: NLBGS 15 ; 2.73466584e-12 2.38020592e-13
  NL: NLBGS Converged
  
.. figure:: simple_homogeneous_TestCase_test_doc.png
  :scale: 80 %
  :align: center
