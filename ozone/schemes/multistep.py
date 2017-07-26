from __future__ import division

import numpy as np
from ozone.schemes.scheme import GLMScheme

ab_coeffs = {
    1: np.array([0., 1.]) ,
    2: np.array([0., 3., -1.]) / 2. ,
    3: np.array([0., 23., -16., 5.]) / 12. ,
    4: np.array([0., 55., -59., 37., -9.]) / 24.,
    5: np.array([0., 1901, -2774., 2616., -1274., 251.]) / 720.,
}
am_coeffs = {
    1: np.array([1., 1.]) / 2. ,
    2: np.array([5., 8., -1.]) / 12. ,
    3: np.array([9., 19., -5., 1.]) / 24. ,
    4: np.array([251., 646., -264., 106., -19.]) / 720.,
}


class Adams(GLMScheme):
    def __init__(self, type_, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For Adams methods, order must be between 2 and 5, inclusive'

        if type_ == 'b':
            num_steps = order
            coeffs = ab_coeffs
        elif type_ == 'm':
            num_steps = order - 1
            coeffs = am_coeffs

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

        starting_scheme_name = 'RK4ST'

        starting_coeffs = np.zeros((num_steps + 1, num_steps + 1, 2))
        starting_coeffs[0, -1, 0] = 1.0
        for i in range(num_steps):
            starting_coeffs[i + 1, -i - 1, 1] = 1.0

        starting_time_steps = num_steps

        super(Adams, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.ones(1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))


class AdamsAlt(GLMScheme):
    def __init__(self, type_, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For Adams methods, order must be between 2 and 5, inclusive'

        if type_ == 'b':
            num_steps = order
            coeffs = ab_coeffs
        elif type_ == 'm':
            num_steps = order - 1
            coeffs = am_coeffs

        A = np.zeros((num_steps + 1, num_steps + 1))
        U = np.zeros((num_steps + 1, num_steps))
        B = np.zeros((num_steps, num_steps + 1))
        V = np.eye(num_steps, k=-1)

        A[-1, :] = coeffs[num_steps][::-1]

        B[0, :] = coeffs[num_steps][::-1]

        V[0, 0] = 1.0

        U[-1, 0] = 1.0
        U[np.arange(num_steps), np.arange(num_steps)[::-1]] = 1.0

        starting_scheme_name = 'RK4'

        starting_coeffs = np.zeros((num_steps, num_steps, 1))
        starting_coeffs[::-1, :, 0] = np.eye(num_steps)

        starting_time_steps = num_steps - 1

        super(AdamsAlt, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.linspace(-num_steps + 1, 1, num_steps + 1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))


class AdamsPEC(GLMScheme):
    def __init__(self, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For Adams methods, order must be between 2 and 5, inclusive'

        num_steps = order - 1

        A = np.zeros((1, 1))
        B = np.zeros((num_steps + 1, 1))
        U = np.zeros((1, num_steps + 1))
        V = np.eye(num_steps + 1, k=-1)

        B[0, 0] = am_coeffs[num_steps][0]
        B[1, 0] = 1.0

        U[0, 0] = 1.0
        U[0, 1:] = ab_coeffs[num_steps][1:]

        V[0, 0] = 1.0
        V[0, 1:] = am_coeffs[num_steps][1:]
        V[1, 0] = 0.0

        starting_scheme_name = 'RK4ST'

        starting_coeffs = np.zeros((num_steps + 1, num_steps + 1, 2))
        starting_coeffs[0, -1, 0] = 1.0
        for i in range(num_steps):
            starting_coeffs[i + 1, -i - 1, 1] = 1.0

        starting_time_steps = num_steps

        super(AdamsPEC, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.ones(1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))


class AdamsPECE(GLMScheme):
    def __init__(self, order):
        assert isinstance(order, int) and 2 <= order <= 5, \
            'For Adams methods, order must be between 2 and 5, inclusive'

        num_steps = order - 1

        A = np.zeros((2, 2))
        B = np.zeros((num_steps + 1, 2))
        U = np.zeros((2, num_steps + 1))
        V = np.eye(num_steps + 1, k=-1)

        A[1, 0] = am_coeffs[num_steps][0]

        B[0, 0] = am_coeffs[num_steps][0]
        B[1, 1] = 1.0

        U[0, 0] = 1.0
        U[0, 1:] = ab_coeffs[num_steps][1:]
        U[1, 0] = 1.0
        U[1, 1:] = am_coeffs[num_steps][1:]

        V[0, 0] = 1.0
        V[0, 1:] = am_coeffs[num_steps][1:]
        V[1, 0] = 0.0

        starting_scheme_name = 'RK4ST'

        starting_coeffs = np.zeros((num_steps + 1, num_steps + 1, 2))
        starting_coeffs[0, -1, 0] = 1.0
        for i in range(num_steps):
            starting_coeffs[i + 1, -i - 1, 1] = 1.0

        starting_time_steps = num_steps

        super(AdamsPECE, self).__init__(A=A, B=B, U=U, V=V,
            abscissa=np.ones(2),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))


class AB(Adams):
    def __init__(self, order):
        super(AB, self).__init__('b', order)


class AM(Adams):
    def __init__(self, order):
        super(AM, self).__init__('m', order)


class ABalt(AdamsAlt):
    def __init__(self, order):
        super(ABalt, self).__init__('b', order)


class AMalt(AdamsAlt):
    def __init__(self, order):
        super(AMalt, self).__init__('m', order)


class BDF(GLMScheme):
    def __init__(self, num_steps):
        assert isinstance(num_steps, int) and 2 <= num_steps <= 6, \
            'For BDF, num_steps must be between 2 and 6, inclusive'

        A = np.zeros((1, 1))
        U = np.zeros((1, num_steps))
        B = np.zeros((num_steps, 1))
        V = np.eye(num_steps, k=-1)

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

        A[0, 0] = f_coeffs[num_steps]
        B[0, 0] = f_coeffs[num_steps]
        U[0, :] = y_coeffs[num_steps]
        V[0, :] = y_coeffs[num_steps]

        starting_scheme_name = 'RK4'

        starting_coeffs = np.zeros((num_steps, num_steps, 1))
        starting_coeffs[::-1, :, 0] = np.eye(num_steps)

        starting_time_steps = num_steps - 1

        super(BDF, self).__init__(A=A, B=B, U=U, V=V, abscissa=np.ones(1),
            starting_method=(starting_scheme_name, starting_coeffs, starting_time_steps))
