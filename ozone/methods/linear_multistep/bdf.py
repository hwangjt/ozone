from __future__ import division

import numpy as np

from ozone.methods.linear_multistep.linear_multistep import LinearMultistep


f_coeffs = {
    2: 2. / 3.,
    3: 6. / 11.,
    4: 12. / 25.,
    5: 60. / 137.,
    6: 60. / 147.,
}

y_coeffs = {
    2: np.array([4., -1]) / 3.,
    3: np.array([18., -9., 2.]) / 11.,
    4: np.array([48., -36., 16., -3]) / 25.,
    5: np.array([300., -300., 200., -75., 12.]) / 137.,
    6: np.array([360., -450., 400., -225., 72., -10.]) / 147.,
}


class BDF(LinearMultistep):

    def __init__(self, num_steps):
        assert isinstance(num_steps, int) and 2 <= num_steps <= 6, \
            'For BDF, num_steps must be between 2 and 6, inclusive'

        self.order = num_steps

        A = np.zeros((1, 1))
        U = np.zeros((1, num_steps))
        B = np.zeros((num_steps, 1))
        V = np.eye(num_steps, k=-1)

        A[0, 0] = f_coeffs[num_steps]
        B[0, 0] = f_coeffs[num_steps]
        U[0, :] = y_coeffs[num_steps]
        V[0, :] = y_coeffs[num_steps]

        starting_method_name = 'RK6'

        starting_coeffs = np.zeros((num_steps, num_steps, 1))
        starting_coeffs[::-1, :, 0] = np.eye(num_steps)

        starting_times = num_steps - 1

        abscissa = np.ones(1)
        starting_method = (starting_method_name, starting_coeffs, starting_times)

        super(BDF, self).__init__(A, B, U, V, abscissa, starting_method)
