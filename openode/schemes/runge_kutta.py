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

    def starting_method(self, y0):
        """
        Transforms the initial condition into the initial y^[0] vector. Since RK schemes do not
        carry additional information from step to step, this is trivial.

        Parameters
        ----------
        y0 : np.array
            Initial Condition at t=t0.

        Returns
        -------
        np.array
            Initial y^[0] (super)-vector.
        """
        return y0[np.newaxis, ...]

class ExplicitRungeKutta(RungeKutta):
    def __init__(self, A, B):

        A = np.atleast_2d(A)

        if np.any(np.triu(A)):
            raise ValueError('Explicit Runge-Kutta methods must have zeros for the upper triangular '
                             'portion of A.')

        B = np.atleast_2d(B)

        super(ExplicitRungeKutta, self).__init__(A=A, B=B)


class ImplicitRungeKutta(RungeKutta):
    pass


class ForwardEuler(ExplicitRungeKutta):
    def __init__(self):
        super(ForwardEuler, self).__init__(A=0., B=1.)


class ExplicitMidpoint(ExplicitRungeKutta):
    def __init__(self):
        super(ExplicitMidpoint, self).__init__(
            A=np.array([
                [ 0., 0.],
                [1/2, 0.],
            ]),
            B=np.array([
                [0., 1.],
            ]))


class HeunsMethod(ExplicitRungeKutta):
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


class RalstonsMethod(ExplicitRungeKutta):
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


class KuttaThirdOrder(ExplicitRungeKutta):
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


class RK4(ExplicitRungeKutta):
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
