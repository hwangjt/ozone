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
        self.order = 1

        super(ForwardEuler, self).__init__(A=0., B=1.)


def get_ExplicitMidpoint():
    ExplicitMidpoint_A = np.array([
        [ 0., 0.],
        [1/2, 0.],
    ])

    ExplicitMidpoint_B = np.array([
        [0., 1.],
    ])

    return ExplicitMidpoint_A, ExplicitMidpoint_B


class ExplicitMidpoint(RungeKutta):

    def __init__(self):
        self.order = 2

        ExplicitMidpoint_A, ExplicitMidpoint_B = get_ExplicitMidpoint()

        A = np.array(ExplicitMidpoint_A)
        B = np.array(ExplicitMidpoint_B)

        super(ExplicitMidpoint, self).__init__(A=A, B=B)


class ExplicitMidpointST(RungeKuttaST):

    def __init__(self):
        self.order = 2

        ExplicitMidpoint_A, ExplicitMidpoint_B = get_ExplicitMidpoint()

        A = np.zeros((3, 3))
        B = np.zeros((2, 3))

        A[:2, :2] = ExplicitMidpoint_A
        A[2, :2] = ExplicitMidpoint_B
        B[0, :2] = ExplicitMidpoint_B
        B[1, 2] = 1.

        super(ExplicitMidpointST, self).__init__(A=A, B=B)


class HeunsMethod(RungeKutta):

    def __init__(self):
        self.order = 2

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
        self.order = 2

        super(RalstonsMethod, self).__init__(
            A=np.array([
                [0., 0.],
                [2 / 3, 0.],
            ]),
            B=np.array([
                [1 / 4, 3 / 4],
            ])
        )


def get_KuttaThirdOrder():
    KuttaThirdOrder_A = np.array([
        [0., 0., 0.],
        [1 / 2, 0., 0.],
        [-1., 2., 0.],
    ])

    KuttaThirdOrder_B = np.array([
        [1 / 6, 4 / 6, 1 / 6],
    ])

    return KuttaThirdOrder_A, KuttaThirdOrder_B


class KuttaThirdOrder(RungeKutta):

    def __init__(self):
        self.order = 3

        KuttaThirdOrder_A, KuttaThirdOrder_B = get_KuttaThirdOrder()

        A = np.array(KuttaThirdOrder_A)
        B = np.array(KuttaThirdOrder_B)

        super(KuttaThirdOrder, self).__init__(A=A, B=B)


class KuttaThirdOrderST(RungeKuttaST):

    def __init__(self):
        self.order = 3

        KuttaThirdOrder_A, KuttaThirdOrder_B = get_KuttaThirdOrder()

        A = np.zeros((4, 4))
        B = np.zeros((2, 4))

        A[:3, :3] = KuttaThirdOrder_A
        A[3, :3] = KuttaThirdOrder_B
        B[0, :3] = KuttaThirdOrder_B
        B[1, 3] = 1.

        super(KuttaThirdOrderST, self).__init__(A=A, B=B)


def get_RK4():
    RK4_A = np.array([
        [0., 0., 0., 0.],
        [1 / 2, 0., 0., 0.],
        [0., 1 / 2, 0., 0.],
        [0., 0., 1., 0.],
    ])

    RK4_B = np.array([
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    ])

    return RK4_A, RK4_B


class RK4(RungeKutta):

    def __init__(self):
        self.order = 4

        RK4_A, RK4_B = get_RK4()

        A = np.array(RK4_A)
        B = np.array(RK4_B)

        super(RK4, self).__init__(A=A, B=B)


class RK4ST(RungeKuttaST):

    def __init__(self):
        self.order = 4

        RK4_A, RK4_B = get_RK4()

        A = np.zeros((5, 5))
        B = np.zeros((2, 5))

        A[:4, :4] = RK4_A
        A[4, :4] = RK4_B
        B[0, :4] = RK4_B
        B[1, 4] = 1.

        super(RK4ST, self).__init__(A=A, B=B)


def get_RK6(s):
    r = s * np.sqrt(5)

    RK6_A = np.zeros((7, 7))
    RK6_A[1, :1] = (5 - r) / 10
    RK6_A[2, :2] = [ (-r) / 10 , (5 + 2 * r) / 10]
    RK6_A[3, :3] = [ (-15 + 7 * r) / 20 , (-1 + r) / 4, (15 - 7 * r) / 10]
    RK6_A[4, 0] = (5 - r) / 60
    RK6_A[4, 2:4] = [ 1 / 6 , (15 + 7 * r) / 60 ]
    RK6_A[5, 0] = (5 + r) / 60
    RK6_A[5, 2:5] = [ (9 - 5 * r) / 12 , 1 / 6 , (-5 + 3 * r) / 10 ]
    RK6_A[6, 0] = 1 / 6
    RK6_A[6, 2:6] = [ (-55 + 25 * r) / 12 , (-25 - 7 * r) / 12 , 5 - 2 * r , (5 + r) / 2 ]

    RK6_B = np.zeros((1, 7))
    RK6_B[0, 0] = 1 / 12
    RK6_B[0, 4:7] = [ 5 / 12 , 5 / 12 , 1 / 12 ]

    return RK6_A, RK6_B


class RK6(RungeKutta):

    def __init__(self, s=1.):
        self.order = 6

        RK6_A, RK6_B = get_RK6(s)

        A = np.array(RK6_A)
        B = np.array(RK6_B)

        super(RK6, self).__init__(A=A, B=B)


class RK6ST(RungeKuttaST):

    def __init__(self, s=1.):
        self.order = 6

        RK6_A, RK6_B = get_RK6(s)

        A = np.zeros((8, 8))
        B = np.zeros((2, 8))

        A[:7, :7] = RK6_A
        A[7, :7] = RK6_B
        B[0, :7] = RK6_B
        B[1, 7] = 1.

        super(RK6ST, self).__init__(A=A, B=B)


class BackwardEuler(RungeKutta):

    def __init__(self):
        self.order = 1

        super(BackwardEuler, self).__init__(A=1., B=1.)


class ImplicitMidpoint(RungeKutta):

    def __init__(self):
        self.order = 2

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
        self.order = order

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
        self.order = order

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
        self.order = order

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

class TrapezoidalRule(RungeKutta):

    def __init__(self):
        self.order = 2

        super(TrapezoidalRule, self).__init__(A=np.array([[0., 0.], [1 / 2, 1 / 2]]),
                                              B=np.array([1 / 2, 1 / 2]))
