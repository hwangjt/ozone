from __future__ import division

import numpy as np
from openode.schemes.scheme import GLMScheme


class AM(GLMScheme):
    def __init__(self, num_steps):
        assert isinstance(num_steps, int) and 2 <= num_steps <= 4, \
            'For AM, num_steps must be between 2 and 4, inclusive'

        A = np.zeros((num_steps + 1, num_steps + 1))
        U = np.zeros((num_steps + 1, num_steps))
        B = np.zeros((num_steps, num_steps + 1))
        V = np.eye(num_steps, k=-1)

        coeffs = {
            2: np.array([-1., 8., 5.]) / 12. ,
            3: np.array([1., -5., 19., 9.]) / 24. ,
            4: np.array([-19., 106., -264., 646., 251.]) / 720.,
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
