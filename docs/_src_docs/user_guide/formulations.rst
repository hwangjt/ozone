Formulations
============

In ozone, there are 3 formulations for solving the ODE.
To illustrate, let us consider applying two time steps of the backward Euler method:

.. math ::
  y_1 &= y_0 + h f(t, y_1) \\
  y_2 &= y_1 + h f(t, y_2) \\

1. Time-marching: integrating the ODE one time step at a time,
evaluating the appropriate equations (explicit methods)
or solving the system of equations (implicit methods) at each time step.
In the example, time marching would involve solving the two nonlinear equations
(for :math:`y_1` and :math:`y_2`, respectively), one after the other.

2. System-based: integrating the ODE by combining the equations from all time steps into a single nonlinear system of equations and solving with a nonlinear solver.
In the example, the system-based formulation would formulate and solve the 2-by-2 nonlinear system,

  .. math::
    y_1 - y_0 - h f(t, y_1) = 0 \\
    y_2 - y_1 - h f(t, y_2) = 0 \\

3. Optimizer-based: integrating the ODE by treating the ODE state variables and their implicit equations
as design variables and constraints, respectively, in the optimization problem:

  .. math::
    \text{min} & \quad \text{objective} \\
    \text{with respect to} & \quad y_1, y_2 \\
    \text{subject to} & \quad y_1 - y_0 - h f(t, y_1) = 0 \\
    & \quad y_2 - y_1 - h f(t, y_2) = 0 \\

  .. toctree::
    :maxdepth: 1
    :titlesonly:

    formulations/timing_plot
