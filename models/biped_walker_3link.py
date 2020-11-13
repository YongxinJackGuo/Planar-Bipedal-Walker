from optimization.opt_traj import get_opt_coeff
import numpy as np
from numpy.linalg import inv

class BipedWalker3Link(object):
    def __init__(self):
        # link length and mass are both the same for 2 legs by symmetric
        self.r = 1  # distance from leg end to hip
        self.l = 0.5  # distance from hip to torso
        self.m = 5  # weight of legs
        self.mh = 15  # weight of hip
        self.mt = 10  # weight of torso
        self.epsilon = 0.1
        self.alpha = 0.9
        self.n = 3  # dof
        self.g = 9.8  # gravity coefficient
        self.opt_coeff = get_opt_coeff()

    def h_d(self):

        return None

    def swing_dynamics(self, x, u):
        D = self.get_D(x)
        C = self.get_C(x)
        G = self.get_G(x)
        B = self.get_B()
        qdot = x[self.n:]
        qddot = inv(D) @ (-C @ qdot - G + B @ u)
        xdot = np.concatenate([qdot, qddot])
        return xdot

    def impact_dynamics(self):

        return None

    def zero_dynamics(self):

        return None

    def get_D(self, x):
        m = self.m
        l = self.l
        mh = self.mh
        mt = self.mt
        r = self.r
        n = self.n
        th1, th2, th3 = x[0], x[1], x[2]
        c12, c13 = np.cos(th1 - th2), np.cos(th1 - th3)

        D = np.zeros((n, n))
        D[0, 0] = ((5/4)*m + mh + mt) * r**2
        D[0, 1] = -(1/2) * m * (r**2) * c12
        D[0, 2] = mt * r * l * c13
        D[1, 0] = -(1/2) * m * (r**2) * c12
        D[1, 1] = (1/4) * m * r**2
        D[2, 0] = mt * r * l * c13
        D[2, 2] = mt * l**2

        return D


    def get_C(self, x):
        m = self.m
        l = self.l
        mh = self.mh
        mt = self.mt
        r = self.r
        n = self.n
        th1, th2, th3 = x[0], x[1], x[2]
        th1dot, th2dot, th3dot = x[3], x[4], x[5]
        s12, s13 = np.sin(th1 - th2), np.sin(th1 - th3)

        C = np.zeros((n, n))
        C[0, 1] = -(1/2) * m * (r**2) * s12 * th2dot
        C[0, 2] = mt * r * l * s13 * th3dot
        C[1, 0] = (1/2) * m * (r**2) * s12 * th1dot
        C[2, 0] = -mt * r * l * s13 * th1dot

        return C


    def get_G(self, x):
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
        s1, s2, s3 = np.sin(th1), np.sin(th2), np.sin(th3)

        G = np.zeros(n,)
        G[0] = -(1/2) * g * (2 * mh + 3 * m + 2 * mt) * r * s1
        G[1] = (1/2) * g * m * r * s2
        G[2] = -g * mt * l * s3

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


    def compute_cls_feedback_u(self, x):

        return None


    def compute_hzd_feedback_u(self):

        return None


    def compute_L2fy(self, x):

        return None

    def compute_LgLfy(self, x):

        return None

    def compute_phi(self):

        return None

