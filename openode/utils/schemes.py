from __future__ import division

import numpy as np


def get_scheme(method_name):
    print(locals())
    return locals()[method_name]()


def forward_euler():
    glm_A = np.array([
        [0.],
    ])

    glm_B = np.array([
        [1.],
    ])

    glm_U = np.array([
        [1.],
    ])

    glm_V = np.array([
        [1.],
    ])

    return glm_A, glm_B, glm_U, glm_V, 1, 1


def explicit_midpoint():
    glm_A = np.array([
        [ 0., 0.],
        [1/2, 0.],
    ])

    glm_B = np.array([
        [0., 1.],
    ])

    glm_U = np.array([
        [1.],
        [1.],
    ])

    glm_V = np.array([
        [1.],
    ])

    return glm_A, glm_B, glm_U, glm_V, 2, 1


def heuns_method():
    glm_A = np.array([
        [ 0., 0.],
        [ 1., 0.],
    ])

    glm_B = np.array([
        [1/2,1/2],
    ])

    glm_U = np.array([
        [1.],
        [1.],
    ])

    glm_V = np.array([
        [1.],
    ])

    return glm_A, glm_B, glm_U, glm_V, 2, 1


def ralstons_method():
    glm_A = np.array([
        [ 0., 0.],
        [2/3, 0.],
    ])

    glm_B = np.array([
        [1/4,3/4],
    ])

    glm_U = np.array([
        [1.],
        [1.],
    ])

    glm_V = np.array([
        [1.],
    ])

    return glm_A, glm_B, glm_U, glm_V, 2, 1


def kutta_third_order():
    glm_A = np.array([
        [ 0., 0., 0.],
        [1/2, 0., 0.],
        [-1., 2., 0.],
    ])

    glm_B = np.array([
        [1/6,4/6,1/6],
    ])

    glm_U = np.array([
        [1.],
        [1.],
        [1.],
    ])

    glm_V = np.array([
        [1.],
    ])

    return glm_A, glm_B, glm_U, glm_V, 3, 1


def RK4():
    glm_A = np.array([
        [ 0.,  0.,  0.,  0.],
        [1/2,  0.,  0.,  0.],
        [ 0., 1/2,  0.,  0.],
        [ 0.,  0.,  1.,  0.],
    ])

    glm_B = np.array([
        [1/6, 1/3, 1/3, 1/6],
    ])

    glm_U = np.array([
        [1.0],
        [1.0],
        [1.0],
        [1.0],
    ])

    glm_V = np.array([
        [1.0],
    ])

    return glm_A, glm_B, glm_U, glm_V, 4, 1
