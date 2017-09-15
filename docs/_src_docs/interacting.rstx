Interacting with the integrator group
=====================================

The following docs explain how to specify times, initial conditions, and dynamic and static parameters.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  interacting/specifying_times
  interacting/specifying_ics
  interacting/specifying_parameters

The rest of the OpenMDAO model can interact with an integrator group
(assumed to be named :code:`integrator_group`) in the following ways:

Inputs:

- :code:`integrator_group.initial_time`
- :code:`integrator_group.final_time`
- :code:`integrator_group.initial_condition:*state_name*`
- :code:`integrator_group.static_parameter:*static_parameter_name*`
- :code:`integrator_group.dynamic_parameter:*dynamic_parameter_name*`

Outputs:

- :code:`integrator_group.times`
- :code:`integrator_group.state:*state_name*`

Inputs are variables that external components can optionally connect to.
Outputs are variables that external components can optionally connect from.
