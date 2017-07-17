from __future__ import division
import numpy as np

class GLMScheme(object):
    """
    Base class for all GLM schemes.
    """
    def __init__(self, A, B, U, V, abscissa):

        s, s2 = A.shape

        if s != s2:
            raise ValueError('GLM Matrix A must be square. Received {}x{}'.format(s, s2))

        us, r = U.shape

        if us != s:
            raise ValueError('GLM Matrix U must have {} rows to match A. '
                             'Received B: {}x{}'.format(s, s, s, us, r))

        br, bs = B.shape

        if br != r or bs != s:
            raise ValueError('GLM Matrix B must have {} rows and {} columns to match A and U. '
                             'Received B: {}x{}'.format(s, r, br, bs))

        vr, vr2 = V.shape

        if vr != r or vr2 != r:
            raise ValueError('GLM Matrix V must have {} rows and {} columns to match A and U. '
                             'Received B: {}x{}'.format(s, r, vr, vr2))

        self.num_stages = s
        self.num_values = r
        self.abscissa = abscissa
        self.A = A
        self.B = B
        self.U = U
        self.V = V

        lower = np.tril(A, -1)
        err = np.linalg.norm(lower - A) / np.linalg.norm(A)
        self.explicit = err < 1e-15

    def starting_method(self, y0):
        """
        Transforms the initial condition into the initial y^[0] vector.

        Parameters
        ----------
        y0 : np.array
            Initial Condition at t=t0.

        Returns
        -------
        np.array
            Initial y^[0] (super)-vector.
        """
        raise NotImplementedError('GLM Subclasses must define a starting method.')
