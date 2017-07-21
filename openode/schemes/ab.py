from __future__ import division

import numpy as np
from openode.schemes.scheme import GLMScheme


class AB(GLMScheme):
    def __init__(self, num_steps):
        assert isinstance(num_steps, int) and 2 <= num_steps <= 5, \
            'For AB, num_steps must be between 2 and 5, inclusive'

        A = np.zeros((num_steps, num_steps))
        U = np.eye(num_steps)[::-1,:]
        B = np.zeros((num_steps, num_steps))
        V = np.eye(num_steps, k=-1)

        coeffs = {
            2: np.array([-1., 3.]) / 2. ,
            3: np.array([5., -16., 23.]) / 12. ,
            4: np.array([-9., 37., -59., 55.]) / 24.,
            5: np.array([251, -1274., 2616., -2774., 1901.]) / 720.,
        }
        B[0, :] = coeffs[num_steps]
        V[0, 0] = 1.0

        starting_scheme_name = 'RK4'
        starting_coeffs = np.eye(num_steps).reshape((num_steps, num_steps, 1))
        starting_time_steps = num_steps - 1

        super(AB, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.linspace(-num_steps + 1, 0, num_steps),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))
