Specifying times
================

When calling :code:`ODEIntegrator`, there are three general ways of specifying the integration times.

1. Absolute times
-----------------

The first way is to specify the actual times values instead of
specifying the initial time, final time, and time spacing separately.
This first way is the simplest approach and is sufficient if the initial and final times
do not change during optimization.
As shown below, we provide the :code:`times` argument when calling :code:`ODEIntegrator`.

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from openmdao.api import Problem
  
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.getting_started_ode_func \
      import GettingStartedODEFunction
  
  ode_function = GettingStartedODEFunction()
  formulation = 'solver-based'
  method_name = 'RK4'
  initial_conditions = {'y': 1.}
  
  # Only times is passed in.
  times = np.linspace(0., 3., 101)
  
  integrator = ODEIntegrator(ode_function, formulation, method_name,
      times=times, initial_conditions=initial_conditions)
  
  prob = Problem(model=integrator)
  prob.setup(check=False)
  prob.run_model()
  
  plt.plot(prob['times'], prob['state:y'][:, 0])
  plt.xlabel('t')
  plt.ylabel('y')
  plt.show()
  
.. figure:: specifying_times_Test_test_times.png
  :scale: 80 %
  :align: center

2. Normalized times with fixed initial and final times
------------------------------------------------------

Alternatively, we can specify the initial time, final time, and relative positions of the times as a vector.
In this case, we provide the :code:`initial_time`, :code:`final_time`, and :code:`normalized_times`
arguments when calling :code:`ODEIntegrator`:

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from openmdao.api import Problem
  
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.getting_started_ode_func \
      import GettingStartedODEFunction
  
  ode_function = GettingStartedODEFunction()
  formulation = 'solver-based'
  method_name = 'RK4'
  initial_conditions = {'y': 1.}
  
  # Here, initial_time, final_time, and normalized_times are passed in.
  initial_time = 0.
  final_time = 3.
  normalized_times = np.linspace(0., 1., 101)
  
  integrator = ODEIntegrator(ode_function, formulation, method_name,
      initial_time=initial_time, final_time=final_time,
      normalized_times=normalized_times, initial_conditions=initial_conditions)
  
  prob = Problem(model=integrator)
  prob.setup(check=False)
  prob.run_model()
  
  plt.plot(prob['times'], prob['state:y'][:, 0])
  plt.xlabel('t')
  plt.ylabel('y')
  plt.show()
  
.. figure:: specifying_times_Test_test_normalized_dict.png
  :scale: 80 %
  :align: center

3. Normalized times with variable initial and final times
---------------------------------------------------------

If the initial and/or final time changes during optimization, we can connect one or both from
an external component that computes the initial and/or final time as an output.
Here is an example where both are connected externally.

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from openmdao.api import Problem, IndepVarComp
  
  from ozone.api import ODEIntegrator
  from ozone.tests.ode_function_library.getting_started_ode_func \
      import GettingStartedODEFunction
  
  ode_function = GettingStartedODEFunction()
  formulation = 'solver-based'
  method_name = 'RK4'
  initial_conditions={'y': 1.}
  
  # Only normalized_times is passed in
  normalized_times = np.linspace(0., 1., 101)
  
  integrator = ODEIntegrator(ode_function, formulation, method_name,
      normalized_times=normalized_times, initial_conditions=initial_conditions)
  
  # Below, initial_time and final_time are connected from external components.
  prob = Problem()
  prob.model.add_subsystem('initial_time_comp', IndepVarComp('initial_time', 0.))
  prob.model.add_subsystem('final_time_comp', IndepVarComp('final_time', 3.))
  prob.model.add_subsystem('integrator_group', integrator)
  prob.model.connect('initial_time_comp.initial_time', 'integrator_group.initial_time')
  prob.model.connect('final_time_comp.final_time', 'integrator_group.final_time')
  prob.setup(check=False)
  prob.run_model()
  
  plt.plot(prob['integrator_group.times'], prob['integrator_group.state:y'][:, 0])
  plt.xlabel('t')
  plt.ylabel('y')
  plt.show()
  
.. figure:: specifying_times_Test_test_normalized_connected.png
  :scale: 80 %
  :align: center
