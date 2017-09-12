Manually checking convergence order
===================================

Here is a script for manually checking convergence order for a given method and a given problem.
The output table is shown below.

.. code-block:: python

  import numpy as np
  
  from ozone.tests.ode_function_library.simple_nonlinear_func import SimpleNonlinearODEFunction
  from ozone.utils.run_utils import compute_convergence_order
  
  
  num_times_vector = np.array([16, 32, 64, 128, 256])
  method_name = 'ImplicitMidpoint'
  formulation = 'solver-based'
  
  ode_function = SimpleNonlinearODEFunction()
  state_name = 'y'
  initial_conditions = {'y': 1.}
  t0 = 0.
  t1 = 1.
  
  errors_vector, step_sizes_vector, orders_vector, ideal_order = compute_convergence_order(
      num_times_vector, t0, t1, state_name,
      ode_function, formulation, method_name, initial_conditions)
  
  print('-'*47)
  print('| {:4s} | {:10s} | {:10s} | {:10s} |'.format('Num.', 'h', 'Error', 'Rate'))
  print('-'*47)
  for i in range(len(num_times_vector)):
      print('| {:4d} | {:.4e} | {:.4e} | {:.4e} |'.format(
          num_times_vector[i],
          step_sizes_vector[i],
          errors_vector[i],
          orders_vector[i - 1] if i != 0 else 0.,
      ))
  
::

  -----------------------------------------------
  | Num. | h          | Error      | Rate       |
  -----------------------------------------------
  |   16 | 6.6667e-02 | 2.7362e-03 | 0.0000e+00 |
  |   32 | 3.2258e-02 | 6.3909e-04 | 2.0033e+00 |
  |   64 | 1.5873e-02 | 1.5465e-04 | 2.0008e+00 |
  |  128 | 7.8740e-03 | 3.8052e-05 | 2.0002e+00 |
  |  256 | 3.9216e-03 | 9.4381e-06 | 2.0000e+00 |
  
