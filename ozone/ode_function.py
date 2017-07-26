from __future__ import print_function, division, absolute_import

from collections import Iterable
from six import iteritems, string_types
import numpy as np

from openmdao.utils.options_dictionary import OptionsDictionary


class ODEFunction(object):
    """
    Define an ODE of the form y' = f(t, x, y).

    Attributes
    ----------
    _system_class : System
        OpenMDAO Group or Component class defining our ODE.
    _system_init_kwargs : dict
        Dictionary of kwargs that should be passed in when instantiating system_class.
    _time_options : OptionsDictionary
        Options for the time or time-like variable.
    _states : dict of OptionsDictionary
        Dictionary of options dictionaries for each state.
    _parameters : dict of OptionsDictionary
        Dictionary of options dictionaries for each parameter.
    """

    def __init__(self, **kwargs):
        """
        Initialize class attributes.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments that will be passed to the initialize method.
        """
        self._system_class = None
        self._system_init_kwargs = {}

        time_options = OptionsDictionary()
        time_options.declare('targets', default=[], type_=Iterable)
        time_options.declare('units', default=None, type_=(string_types, type(None)))

        self._time_options = time_options
        self._states = {}
        self._parameters = {}

        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        """
        Optional method that calls declare_time, declare_state, and/or declare_parameter.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed in during instantiation.
        """
        pass

    def set_system(self, system_class, system_init_kwargs=None):
        """
        Set the OpenMDAO System that computes the ODE function.

        Parameters
        ----------
        system_class : System
            OpenMDAO Group or Component class defining our ODE.
        system_init_kwargs : dict or None
            Dictionary of kwargs that should be passed in when instantiating system_class.
        """
        self._system_class = system_class
        if system_init_kwargs is not None:
            self._system_init_kwargs = system_init_kwargs

    def declare_time(self, targets=None, units=None):
        """
        Specify the targets and units of time or the time-like variable.

        Parameters
        ----------
        targets : string_types or Iterable or None
            Targets for the time or time-like variable within the ODE, or None if no models
            are explicitly time-dependent. Default is None.
        units : str or None
            Units for the integration variable within the ODE. Default is None.
        """
        if isinstance(targets, string_types):
            self._time_options['targets'] = [targets]
        elif isinstance(targets, Iterable):
            self._time_options['targets'] = targets
        elif targets is not None:
            raise ValueError('targets must be of type string_types or Iterable or None')
        if units is not None:
            self._time_options['units'] = units

    def declare_state(self, name, rate_target, state_targets=None, shape=None, units=None):
        """
        Add an ODE state variable.

        Parameters
        ----------
        name : str
            The name of the state variable as seen by the driver. This variable will
            exist as an interface to the ODE.
        rate_target : str
            The path to the variable within the ODE which represents the derivative of
            the state variable w.r.t. the variable of integration.
        state_targets : string_types or Iterable or None
            Paths to inputs in the ODE to which the incoming value of the state variable
            needs to be connected.
        shape : int or tuple or None
            The shape of the variable to potentially be provided as a control.
        units : str or None
            Units of the variable.
        """
        if name in self._states:
            raise ValueError('State {0} has already been declared.'.format(name))

        state_options = OptionsDictionary()
        state_options.declare('name', type_=string_types)
        state_options.declare('rate_target', type_=string_types)
        state_options.declare('state_targets', default=[], type_=Iterable)
        state_options.declare('shape', default=(1,), type_=tuple)
        state_options.declare('units', default=None, type_=string_types)

        state_options['name'] = name
        state_options['rate_target'] = rate_target
        if isinstance(state_targets, string_types):
            state_options['state_targets'] = [state_targets]
        elif isinstance(state_targets, Iterable):
            state_options['state_targets'] = state_targets
        elif state_targets is not None:
            raise ValueError('state_targets must be of type string_types or Iterable or None')
        if np.isscalar(shape):
            state_options['shape'] = (shape,)
        elif isinstance(shape, Iterable):
            state_options['shape'] = tuple(shape)
        elif shape is not None:
            raise ValueError('shape must be of type int or Iterable or None')
        if units is not None:
            state_options['units'] = units

        self._states[name] = state_options

    def declare_parameter(self, name, targets):
        """
        Declare an input to the ODE.

        Parameters
        ----------
        name : str
            The name of the state variable as seen by the driver. This variable will
            exist as an interface to the ODE.
        targets : string_types or Iterable or None
            Paths to inputs in the ODE to which the incoming value of the state variable
            needs to be connected.
        """
        if name in self._parameters:
            raise ValueError('Parameter {0} has already been declared.'.format(name))

        parameter_options = OptionsDictionary()
        parameter_options.declare('name', type_=string_types)
        parameter_options.declare('targets', default=[], type_=Iterable)

        parameter_options['name'] = name
        parameter_options['targets'] = targets

        self._parameters[name] = parameter_options
