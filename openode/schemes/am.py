from __future__ import division

import numpy as np
from openode.schemes.scheme import GLMScheme


class AMold(GLMScheme):
    def __init__(self, num_steps):
        assert isinstance(num_steps, int) and 2 <= num_steps <= 4, \
            'For AM, num_steps must be between 2 and 4, inclusive'

        A = np.zeros((num_steps + 1, num_steps + 1))
        U = np.zeros((num_steps + 1, num_steps))
        B = np.zeros((num_steps, num_steps + 1))
        V = np.eye(num_steps, k=-1)

        coeffs = {
            1: np.array([1., 1.]) / 2. ,
            2: np.array([5., 8., -1.]) / 12. ,
            3: np.array([9., 19., -5., 1.]) / 24. ,
            4: np.array([251., 646., -264., 106., -19.]) / 720.,
        }
        A[-1, :] = coeffs[num_steps]
        B[0, :] = coeffs[num_steps]
        V[0, 0] = 1.0
        U[-1, 0] = 1.0
        U[np.arange(num_steps), np.arange(num_steps)[::-1]] = 1.0

        starting_scheme_name = 'RK4'
        starting_coeffs = np.eye(num_steps).reshape((num_steps, num_steps, 1))
        starting_time_steps = num_steps - 1

        super(AM, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.linspace(-num_steps + 1, 1, num_steps + 1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))


class AM(GLMScheme):
    def __init__(self, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For AB, order must be between 2 and 5, inclusive'

        num_steps = order - 1

        A = np.zeros((1, 1))
        B = np.zeros((num_steps + 1, 1))
        U = np.zeros((1, num_steps + 1))
        V = np.eye(num_steps + 1, k=-1)

        coeffs = {
            1: np.array([1., 1.]) / 2. ,
            2: np.array([5., 8., -1.]) / 12. ,
            3: np.array([9., 19., -5., 1.]) / 24. ,
            4: np.array([251., 646., -264., 106., -19.]) / 720.,
        }

        A[0, 0] = coeffs[num_steps][0]
        B[0, 0] = coeffs[num_steps][0]
        B[1, 0] = 1.0
        U[0, 0] = 1.0
        U[0, 1:] = coeffs[num_steps][1:]
        V[0, 0] = 1.0
        V[0, 1:] = coeffs[num_steps][1:]
        V[1, 0] = 0.0

        starting_scheme_name = 'RK4ST'

        starting_coeffs = np.zeros((num_steps + 1, num_steps + 1, 2))
        starting_coeffs[0, -1, 0] = 1.0
        for i in range(num_steps):
            starting_coeffs[i + 1, -i - 1, 1] = 1.0

        starting_time_steps = num_steps

        super(AM, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.ones(1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))
