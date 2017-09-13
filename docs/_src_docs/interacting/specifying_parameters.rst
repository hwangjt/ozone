Specifying parameters
=====================

When calling :code:`ODEIntegrator`, there are two ways of
specifying the static or dynamic parameters.

1. Fixed parameters
-------------------

If the static or dynamic parameters are fixed, the simplest way to specify them is to pass them in
when calling :code:`ODEIntegrator`:

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from openmdao.api import Problem
  
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.getting_started_oc_func \
      import GettingStartedOCFunction
  
  num = 101
  
  ode_function = GettingStartedOCFunction()
  formulation = 'solver-based'
  method_name = 'RK4'
  times = np.linspace(0., 3., num)
  initial_conditions = {'x': 0., 'y': 0., 'v': 0.}
  
  # Here, the dynamic parameter array is passed in
  dynamic_parameters = {'theta': np.zeros((num, 1))}
  
  integrator = ODEIntegrator(ode_function, formulation, method_name,
      times=times, initial_conditions=initial_conditions,
      dynamic_parameters=dynamic_parameters)
  
  prob = Problem(model=integrator)
  prob.setup(check=False)
  prob.run_model()
  
  plt.plot(prob['times'], prob['state:y'][:, 0])
  plt.xlabel('t')
  plt.ylabel('y')
  plt.show()
  
.. figure:: specifying_parameters_Test_test_fixed.png
  :scale: 80 %
  :align: center

2. Variable parameters
----------------------

If the static or dynamic parameters vary during optimization,
they must be connected from an external component:

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from openmdao.api import Problem, IndepVarComp
  
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.getting_started_oc_func \
      import GettingStartedOCFunction
  
  num = 101
  
  ode_function = GettingStartedOCFunction()
  formulation = 'solver-based'
  method_name = 'RK4'
  times = np.linspace(0., 3., num)
  initial_conditions = {'x': 0., 'y': 0., 'v': 0.}
  
  integrator = ODEIntegrator(ode_function, formulation, method_name,
      times=times, initial_conditions=initial_conditions)
  
  # Below, the parameter is connected from an external component.
  prob = Problem()
  prob.model.add_subsystem('parameter_comp',
      IndepVarComp('theta', val=0., shape=(num, 1)))
  prob.model.add_subsystem('integrator_group', integrator)
  prob.model.connect('parameter_comp.theta', 'integrator_group.dynamic_parameter:theta')
  prob.setup(check=False)
  prob.run_model()
  
  plt.plot(prob['integrator_group.times'], prob['integrator_group.state:y'][:, 0])
  plt.xlabel('t')
  plt.ylabel('y')
  plt.show()
  
.. figure:: specifying_parameters_Test_test_variable.png
  :scale: 80 %
  :align: center
