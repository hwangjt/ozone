from __future__ import division

import numpy as np

from ozone.methods.linear_multistep.linear_multistep import LinearMultistep
from ozone.methods.linear_multistep.adams_coeffs import ab_coeffs, am_coeffs


class Adams(LinearMultistep):

    def __init__(self, order, num_steps, coeffs):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For Adams methods, order must be between 2 and 5, inclusive'

        self.order = order

        A = np.zeros((1, 1))
        B = np.zeros((num_steps + 1, 1))
        U = np.zeros((1, num_steps + 1))
        V = np.eye(num_steps + 1, k=-1)

        A[0, 0] = coeffs[num_steps][0]

        B[0, 0] = coeffs[num_steps][0]
        B[1, 0] = 1.0

        U[0, 0] = 1.0
        U[0, 1:] = coeffs[num_steps][1:]

        V[0, 0] = 1.0
        V[0, 1:] = coeffs[num_steps][1:]
        V[1, 0] = 0.0

        starting_method_name = 'RK6ST'

        starting_coeffs = np.zeros((num_steps + 1, num_steps + 1, 2))
        starting_coeffs[0, -1, 0] = 1.0
        for i in range(num_steps):
            starting_coeffs[i + 1, -i - 1, 1] = 1.0

        starting_times = num_steps

        abscissa = np.ones(1)
        starting_method = (starting_method_name, starting_coeffs, starting_times)

        super(Adams, self).__init__(A, B, U, V, abscissa, starting_method)


class AB(Adams):

    def __init__(self, order):
        self.order = order

        super(AB, self).__init__(order, order, ab_coeffs)


class AM(Adams):

    def __init__(self, order):
        self.order = order

        super(AM, self).__init__(order, order - 1, am_coeffs)
