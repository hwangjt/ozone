from __future__ import division

import numpy as np
from openode.schemes.scheme import GLMScheme


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
