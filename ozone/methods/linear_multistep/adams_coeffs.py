from __future__ import division

import numpy as np


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
