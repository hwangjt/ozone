from __future__ import division

import numpy as np

from ozone.methods.linear_multistep.linear_multistep import LinearMultistep
from ozone.methods.linear_multistep.adams_coeffs import ab_coeffs, am_coeffs


class AdamsAlt(LinearMultistep):

    def __init__(self, order, num_steps, coeffs):
        assert isinstance(num_steps, int) and 2 <= num_steps <= 5, \
            'For Adams methods (alternate), num_steps must be between 2 and 5, inclusive'

        self.order = order

        A = np.zeros((num_steps + 1, num_steps + 1))
        U = np.zeros((num_steps + 1, num_steps))
        B = np.zeros((num_steps, num_steps + 1))
        V = np.eye(num_steps, k=-1)

        A[-1, :] = coeffs[num_steps][::-1]

        B[0, :] = coeffs[num_steps][::-1]

        V[0, 0] = 1.0

        U[-1, 0] = 1.0
        U[np.arange(num_steps), np.arange(num_steps)[::-1]] = 1.0

        starting_method_name = 'RK6'

        starting_coeffs = np.zeros((num_steps, num_steps, 1))
        starting_coeffs[::-1, :, 0] = np.eye(num_steps)

        starting_times = num_steps - 1

        abscissa = np.linspace(-num_steps + 1, 1, num_steps + 1)
        starting_method = (starting_method_name, starting_coeffs, starting_times)

        super(AdamsAlt, self).__init__(A, B, U, V, abscissa, starting_method)


class ABalt(AdamsAlt):

    def __init__(self, order):
        self.order = order

        super(ABalt, self).__init__(order, order, ab_coeffs)


class AMalt(AdamsAlt):

    def __init__(self, order):
        self.order = order

        super(AMalt, self).__init__(order, order - 1, am_coeffs)
