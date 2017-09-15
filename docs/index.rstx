.. ozone documentation master file, created by
   sphinx-quickstart on Tue Aug 29 15:18:57 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ozone (O3): (O)pen-source (O)DE and (O)ptimal control solver
============================================================

Ozone is an open-source tool for solving ordinary differential equations (ODEs) and optimal control problems.
It is designed to enable the integration of ODEs in gradient-based multidisciplinary design optimization (MDO) problems,
where the ODE is a single component in the larger model and derivatives of the ODE integration process are required.
Ozone can also be used for solving optimal control problems with direct transcription or indirect approaches.
It is built on top of the `OpenMDAO framework <https://github.com/openmdao/blue>`_, which is described and documented `here <https://blue.readthedocs.io>`_.

An ODE is of the form

.. math ::
  \frac{\partial \textbf y}{\partial t} = \textbf f(t, \textbf y, \textbf x) , \qquad  \textbf y(t_0) = \textbf y_0

where
:math:`\textbf y` is the vector of *state variables* (the variable being integrated),
:math:`t` is *time* (the independent variable),
:math:`\textbf x` is the vector of *parameters* (an input to the ODE),
:math:`\textbf f` is the *ODE function*,
and
:math:`\textbf y_0` is the vector of initial conditions.

Ozone provides a large library of Runge--Kutta and linear multistep methods thanks to its use of a unified formulation called general linear methods.
The list of methods can be found below.
There are also 3 formulations for solving the ODE:
time-marching,
system-based (formulate the ODE equations as a nonlinear system),
optimizer-based (formulate the ODE state variables and equations as design variables and constraints in an optimization problem).

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   _src_docs/getting_started
   _src_docs/api
   _src_docs/interacting
   _src_docs/methods
   _src_docs/formulations
   _src_docs/examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
