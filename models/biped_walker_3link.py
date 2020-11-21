import numpy as np # import autograd version of numpy for jacobian calculation in controller.
from numpy.linalg import inv
import scipy
from sympy import *

class BipedWalker3Link(object):
    def __init__(self):
        # link length and mass are both the same for 2 legs by symmetric
        self.r = 1  # distance from leg end to hip --> leg length
        self.l = 0.5  # distance from hip to torso
        self.m = 5  # weight of legs
        self.mh = 15  # weight of hip
        self.mt = 10  # weight of torso
        self.n = 3  # dof
        self.g = 9.8  # gravity coefficient
        self.th1d = np.pi/8
        self.th3d = np.pi/6
        self.swingleg_end_angle = -np.pi/8
        self.hit_threshold = 0.01  # a threshold value that determines the swing leg hits the ground.


    def swing_dynamics(self, x, u):
        D = self.get_D(x, is_sym=False)
        C = self.get_C(x, is_sym=False)
        G = self.get_G(x, is_sym=False)
        B = self.get_B()
        n = self.n
        qdot = x[self.n:]
        qddot = inv(D) @ (-C @ qdot.reshape((n, 1)) - G + B @ u)
        xdot = np.concatenate([qdot, qddot.reshape((n, ))])

        return xdot

    def impact_dynamics(self, x):
        # The impact is modeled as an impulse within an infinitesimally small period of time
        # and can induce instantaneous change in velocity but the positions remain still.
        # Thus the impact map for joint position q is just through relabeling of coordinates.
        # by swaping the swing and stance legs (role has changed)

        x_copy = x.copy()
        # Construct Impact Model Matrix
        n = self.n
        E = self.get_E(x)  # size 2 X 5
        De = self.get_De(x)  # size 5 X 5
        imapct_matrix = np.block([[De, -E.T],
                                  [E, np.zeros((2, 2))]])  # size 7 x 7
        z1dot_minus = z2dot_minus = 0  # the velocity of stance leg before impact. Both are 0s.
        qdote_minus = np.array([x[3], x[4], x[5], z1dot_minus, z2dot_minus])
        after_impact_matrix = inv(imapct_matrix) @ np.block([[(De @ qdote_minus).reshape(5, 1)],
                                                             [np.zeros((2, 1))]])
        x_copy[n: 2*n] = after_impact_matrix[0: n, 0].T  # extract post velocity
        self.relabel(x_copy)  # swap the leg coordinate
        return x_copy

    def relabel(self, x):
        # swap the joint angle for legs
        x[0], x[1] = x[1], x[0]
        # swap the joint velocity for legs
        x[3], x[4] = x[4], x[3]
        return None

    def zero_dynamics(self):
        # TODO: Implement the zero dynamics

        return None

    def get_D(self, x, is_sym):
        # Params:
        # Output: n X n
        m = self.m
        l = self.l
        mh = self.mh
        mt = self.mt
        r = self.r
        n = self.n
        th1, th2, th3 = x[0], x[1], x[2]
        if is_sym == False:
            c12, c13 = np.cos(th1 - th2), np.cos(th1 - th3)
        else:
            c12, c13 = cos(th1 - th2), cos(th1 - th3)

        d00 = ((5/4)*m + mh + mt) * r**2
        d01 = -(1/2) * m * (r**2) * c12
        d02 = mt * r * l * c13
        d10 = -(1/2) * m * (r**2) * c12
        d11 = (1/4) * m * r**2
        d20 = mt * r * l * c13
        d22 = mt * l**2
        # Please be noted. D matrix must be defined in this way as required by sympy package
        # for jacobian calculation purpose in hzd_controller. Don't use index assignment!!!
        D = np.array([[d00, d01, d02],
                      [d10, d11, 0],
                      [d20, 0, d22]])

        return D


    def get_C(self, x, is_sym):
        # Params:
        # Output: n X n

        m = self.m
        l = self.l
        mh = self.mh
        mt = self.mt
        r = self.r
        n = self.n
        th1, th2, th3 = x[0], x[1], x[2]
        th1dot, th2dot, th3dot = x[3], x[4], x[5]
        if is_sym == False:
            s12, s13 = np.sin(th1 - th2), np.sin(th1 - th3)
        else:
            s12, s13 = sin(th1 - th2), sin(th1 - th3)

        c01 = -(1/2) * m * (r**2) * s12 * th2dot
        c02 = mt * r * l * s13 * th3dot
        c10 = (1/2) * m * (r**2) * s12 * th1dot
        c20 = -mt * r * l * s13 * th1dot

        # Please be noted. C matrix must be defined in this way as required by sympy package
        # for jacobian calculation purpose in hzd_controller. Don't use index assignment!!!
        C = np.array([[0, c01, c02],
                      [c10, 0, 0],
                      [c20, 0, 0]])

        return C


    def get_G(self, x, is_sym):
        # Params:
        # Output: n X 1

        m = self.m
        l = self.l
        mh = self.mh
        mt = self.mt
        r = self.r
        n = self.n
        g = self.g
        th1 = x[0]
        th2 = x[1]
        th3 = x[2]
        if is_sym == False:
            s1, s2, s3 = np.sin(th1), np.sin(th2), np.sin(th3)
        else:
            s1, s2, s3 = sin(th1), sin(th2), sin(th3)

        G = np.zeros(n,)
        g0 = -(1/2) * g * (2 * mh + 3 * m + 2 * mt) * r * s1
        g1 = (1/2) * g * m * r * s2
        g2 = -g * mt * l * s3

        # Please be noted. G matrix must be defined in this way as required by sympy package
        # for jacobian calculation purpose in hzd_controller. Don't use index assignment!!!
        G = np.array([[g0],
                      [g1],
                      [g2]])

        return G

    def get_B(self):
        n = self.n
        B = np.zeros((n, n - 1))
        B[0, 0] = -1
        B[1, 1] = -1
        B[2, 0] = 1
        B[2, 1] = 1

        return B


    def get_De(self, x):
        m = self.m
        l = self.l
        mh = self.mh
        mt = self.mt
        r = self.r
        n = self.n
        g = self.g
        th1, th2, th3 = x[0], x[1], x[2]
        c12, c13 = np.cos(th1 - th2), np.cos(th1 - th3)

        De = np.zeros((n + 2, n + 2))
        De[0, 0] = ((5/4) * m + mh + mt) * r**2
        De[0, 1] = -(1/2) * m * (r**2) * c12
        De[0, 2] = mt * r * l * c13
        De[0, 3] = ((3/2) * m + mh + mt) * r * np.cos(th1)
        De[0, 4] = -((3/2) * m + mh + mt) * r * np.sin(th1)
        De[1, 1] = (1/4) * m * r**2
        De[1, 2] = 0
        De[1, 3] = -(1/2) * m * r * np.cos(th2)
        De[1, 4] = (1/2) * m * r * np.sin(th2)
        De[2, 2] = mt * l**2
        De[2, 3] = mt * l * np.cos(th3)
        De[2, 4] = -mt * l * np.sin(th3)
        De[3, 3] = 2 * m + mh + mt
        De[3, 4] = 0
        De[4, 4] = 2 * m + mh + mt

        # turn it into a symmetric matrix
        De = De + De.T - np.diag(De.diagonal())

        return De

    def get_E(self, x):
        # TODO: Implement E matrix for impact model
        r = self.r
        th1, th2 = x[0], x[1]
        c1, c2 = np.cos(th1), np.cos(th2)
        s1, s2 = np.sin(th1), np.sin(th2)
        E = np.array([[r*c1, -r*c2, 0, 1, 0],
                      [-r*s1, r*s2, 0, 0, 1]])

        return E



