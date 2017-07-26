from __future__ import division

import numpy as np
from ozone.schemes.scheme import GLMScheme


class RungeKutta(GLMScheme):
    def __init__(self, A, B):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        U = np.ones((A.shape[0], 1))
        V = np.array([[1.]])

        super(RungeKutta, self).__init__(A=A, B=B, U=U, V=V, abscissa=np.sum(A, 1))


class RungeKuttaST(GLMScheme):
    def __init__(self, A, B):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        U = np.zeros((A.shape[0], 2))
        U[:, 0] = 1.0
        V = np.zeros((2, 2))
        V[0, 0] = 1.0

        super(RungeKuttaST, self).__init__(A=A, B=B, U=U, V=V, abscissa=np.sum(A, 1))


class ForwardEuler(RungeKutta):
    def __init__(self):
        super(ForwardEuler, self).__init__(A=0., B=1.)


class ExplicitMidpoint(RungeKutta):
    def __init__(self):
        super(ExplicitMidpoint, self).__init__(
            A=np.array([
                [ 0., 0.],
                [1/2, 0.],
            ]),
            B=np.array([
                [0., 1.],
            ]))


class HeunsMethod(RungeKutta):
    def __init__(self):
        super(HeunsMethod, self).__init__(
            A=np.array([
                [0., 0.],
                [1., 0.],
            ]),
            B=np.array([
                [1/2, 1/2],
            ])
        )


class RalstonsMethod(RungeKutta):
    def __init__(self):
        super(RalstonsMethod, self).__init__(
            A=np.array([
                [0., 0.],
                [2 / 3, 0.],
            ]),
            B=np.array([
                [1 / 4, 3 / 4],
            ])
        )


class KuttaThirdOrder(RungeKutta):
    def __init__(self):
        super(KuttaThirdOrder, self).__init__(
            A=np.array([
                [0., 0., 0.],
                [1 / 2, 0., 0.],
                [-1., 2., 0.],
            ]),
            B=np.array([
                [1 / 6, 4 / 6, 1 / 6],
            ])
        )


class RK4(RungeKutta):
    def __init__(self):
        super(RK4, self).__init__(
            A=np.array([
                [0., 0., 0., 0.],
                [1 / 2, 0., 0., 0.],
                [0., 1 / 2, 0., 0.],
                [0., 0., 1., 0.],
            ]),
            B=np.array([
                [1 / 6, 1 / 3, 1 / 3, 1 / 6],
            ])
        )


class RK4ST(RungeKuttaST):
    def __init__(self):
        A = np.array([
            [0., 0., 0., 0., 0.],
            [1 / 2, 0., 0., 0., 0.],
            [0., 1 / 2, 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [1 / 6, 1 / 3, 1 / 3, 1 / 6, 0.],
        ])
        B = np.array([
            [1 / 6, 1 / 3, 1 / 3, 1 / 6, 0.],
            [0., 0., 0., 0., 1.],
        ])

        super(RK4ST, self).__init__(A=A, B=B)


class BackwardEuler(RungeKutta):
    def __init__(self):
        super(BackwardEuler, self).__init__(A=1., B=1.)


class ImplicitMidpoint(RungeKutta):
    def __init__(self):
        super(ImplicitMidpoint, self).__init__(A=1/2, B=1.)

_gl_coeffs = {
    2: (
        np.array([.5]),
        np.array([1.])
    ),
    4: (
        np.array([[1 / 4, 1 / 4 - np.sqrt(3) / 6],
                  [1 / 4 + np.sqrt(3) / 6, 1 / 4]]),
        np.array([1 / 2, 1 / 2])
    ),
    6: (
        np.array([[5 / 36, 2 / 9 - np.sqrt(15) / 15, 5 / 36 - np.sqrt(15) / 30],
                  [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
                  [5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15) / 15, 5 / 36]]),
        np.array([5 / 18, 4 / 9, 5 / 18])
    )
}

class GaussLegendre(RungeKutta):
    def __init__(self, order=4):
        if order not in _gl_coeffs:
            raise ValueError('GaussLengdre order must be one of the following: {}'.format(
                sorted(_gl_coeffs.keys())
            ))
        A, B = _gl_coeffs[order]
        super(GaussLegendre, self).__init__(A=A, B=B)

_lobatto_coeffs = {
    2: (
        np.array([[0, 0],
                  [1 / 2, 1 / 2]]),
        np.array([1 / 2, 1 / 2])
    ),
    4: (
        np.array([[0, 0, 0],
                  [5 / 24, 1 / 3, -1 / 24],
                  [1 / 6, 2 / 3, 1 / 6]]),
        np.array([1 / 6, 2 / 3, 1 / 6])
    )
}

class LobattoIIIA(RungeKutta):
    def __init__(self, order=4):
        if order not in _lobatto_coeffs:
            raise ValueError('LobattoIIIA order must be one of the following: {}'.format(
                sorted(_lobatto_coeffs.keys())
            ))
        A, B = _lobatto_coeffs[order]
        super(LobattoIIIA, self).__init__(A=A, B=B)

_radau_I_coeffs = {
    3: (
        np.array([[1 / 4, -1 / 4],
                  [1 / 4, 5 / 12]]),
        np.array([1 / 4, 3 / 4])
    ),
    5: (
        np.array([[1 / 9, (-1 - np.sqrt(6)) / 18, (-1 + np.sqrt(6)) / 18],
                  [1 / 9, 11 / 45 + 7*np.sqrt(6) / 360, 11 / 45 - 43*np.sqrt(6) / 360],
                  [1 / 9, 11 / 45 + 43*np.sqrt(6) / 360, 11 / 45 - 7*np.sqrt(6) / 360]]),
        np.array([1 / 9, 4 / 9 + np.sqrt(6) / 36, 4 / 9 - np.sqrt(6) / 36])
    )
}

_radau_II_coeffs = {
    3: (
        np.array([[5 / 12, -1 / 12],
                  [3 / 4, 1 / 4]]),
        np.array([3 / 4, 1 / 4])
    ),
    5: (
        np.array([[11 / 45 - 7*np.sqrt(6) / 360, 37 / 225 - 169 * np.sqrt(6) / 1800, -2 / 225 + np.sqrt(6) / 75],
                  [37 / 225 + 169*np.sqrt(6) / 1800, 11 / 45 + 7*np.sqrt(6) / 360, -2 / 225 - np.sqrt(6) / 75],
                  [4 / 9 - np.sqrt(6) / 36, 4 / 9 + np.sqrt(6) / 36, 1 / 9]]),
        np.array([4 / 9 - np.sqrt(6) / 36, 4 / 9 + np.sqrt(6) / 36, 1 / 9])
    )
}

class Radau(RungeKutta):
    def __init__(self, type_, order=5):
        if type_ == 'I':
            coeffs = _radau_I_coeffs
        elif type_ == 'II':
            coeffs = _radau_II_coeffs

        if order not in coeffs:
            raise ValueError('Radau order must be one of the following: {}'.format(
                sorted(coeffs.keys())
            ))
        A, B = coeffs[order]
        super(Radau, self).__init__(A=A, B=B)
