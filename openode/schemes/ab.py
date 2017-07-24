from __future__ import division

import numpy as np
from openode.schemes.scheme import GLMScheme


ab_coeffs = {
    2: np.array([3., -1.]) / 2. ,
    3: np.array([23., -16., 5.]) / 12. ,
    4: np.array([55., -59., 37., -9.]) / 24.,
    5: np.array([1901, -2774., 2616., -1274., 251.]) / 720.,
}


class AB(GLMScheme):
    def __init__(self, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For AB, order must be between 2 and 5, inclusive'

        num_steps = order

        A = np.zeros((1, 1))
        B = np.zeros((num_steps + 1, 1))
        U = np.zeros((1, num_steps + 1))
        V = np.eye(num_steps + 1, k=-1)

        B[1, 0] = 1.0
        U[0, 0] = 1.0
        U[0, 1:] = ab_coeffs[order]
        V[0, 0] = 1.0
        V[0, 1:] = ab_coeffs[order]
        V[1, 0] = 0.0

        starting_scheme_name = 'RK4ST'

        starting_coeffs = np.zeros((num_steps + 1, num_steps + 1, 2))
        starting_coeffs[0, -1, 0] = 1.0
        for i in range(num_steps):
            starting_coeffs[i + 1, -i - 1, 1] = 1.0

        starting_time_steps = num_steps

        super(AB, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.ones(1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))


class ABalt(GLMScheme):
    def __init__(self, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For AB, order must be between 2 and 5, inclusive'

        num_steps = order

        A = np.zeros((num_steps, num_steps))
        U = np.eye(num_steps)[::-1,:]
        B = np.zeros((num_steps, num_steps))
        V = np.eye(num_steps, k=-1)

        B[0, :] = ab_coeffs[order][::-1]
        V[0, 0] = 1.0

        starting_scheme_name = 'RK4'

        starting_coeffs = np.zeros((num_steps, num_steps, 1))
        starting_coeffs[::-1, :, 0] = np.eye(num_steps)

        starting_time_steps = num_steps - 1

        super(ABalt, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.linspace(-num_steps + 1, 0, num_steps),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))
