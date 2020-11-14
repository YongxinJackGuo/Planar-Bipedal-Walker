from optimization.opt_traj import get_opt_coeff
import numpy as np
from pydrake.forwarddiff import jacobian
from numpy.linalg import inv

class HybridZeroDynamicsController(object):
    def __init__(self, model):
        self.opt_coeff = get_opt_coeff()
        self.model = model
        self.th1d = model.th1d
        self.th3d = model.th3d
        self.epsilon = 0.1
        self.alpha = 0.9
        self.n = model.n

    def compute_cls_feedback_u(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: u --> (n-1)X1. In 3-link case --> 2X1

        LgLfy = self.compute_LgLfy(x)
        L2fy = self.compute_L2fy(x)
        Lfy = self.compute_Lfy(x)
        y = self.get_y(x)
        psi = self.compute_psi_vec(y, Lfy)

        u = inv(LgLfy) @ (psi - L2fy)

        return u

    def compute_hzd_feedback_u(self):
        return None

    def compute_L2fy(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: Lfy size = (n-1) X 1. In 3-link case --> 2X1

        dLfy_dx = jacobian(self.compute_Lfy, x)
        L2fy = dLfy_dx @ self.get_f(x)

        return L2fy

    def compute_LgLfy(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: LgLfy size = (n-1) X 2. In 3-link case --> 2X2

        dLfy_dx = jacobian(self.compute_Lfy, x)
        LgLfy = dLfy_dx @ self.get_g(x)

        return LgLfy

    def compute_Lfy(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: Lfy size = (n-1) X 1. In 3-link case --> 2X1

        dydx = jacobian(self.get_y, x)
        # multiply f, which is xdot
        Lfy = dydx @ self.get_f(x)

        return Lfy

    def get_f(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: f size = (n*2) X 1. In 3-link case --> 6X1

        n = self.n
        model = self.model
        D = model.get_D(x)
        C = model.get_C(x)
        G = model.get_G(x)
        f = np.block([x[0: n], inv(D) @ (-C @ x[0: n] - G)])

        return f

    def get_g(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: g size = (n*2) X 2. In 3-link case --> 6X2

        model = self.model
        D = model.get_D(x)
        B = model.get_B()
        n = self.n

        g = np.block([[np.zeros((n, n-1))],
                      [inv(D) @ B]])
        return g


    def compute_psi_vec(self, y, Lfy):
        # Params:
        # Input: y --> (n-1) X 1. In 3-link case --> 2 X 1
        # Input: Lfy --> (n-1) X 1. In 3-link case --> 2 X 1
        # Output: psi size = (n-1) X 1. In 3-link case --> 2X1

        n = self.n
        psi_vec = np.zeros((n-1,))
        for i in range(n-1):
            psi_vec[i] = self.compute_psi_scalar(y[i], Lfy[i])

        return psi_vec

    def compute_psi_scalar(self, x1, x2):
        alpha = self.alpha
        phi = x1 + (0.5 - alpha) * np.sign(x2) * np.abs(x2)**(2 - alpha)
        psi_scalar = -np.sign(x2) * np.abs(x2)**alpha - np.sign(phi) * np.abs(phi)**(alpha/2 - alpha)
        return psi_scalar

    def get_y(self, x):
        # y size = (n-1) X 1. In 3-link case --> 2X1
        #TODO: generalize to multiple n-dof model

        n = self.n
        y = np.zeros((n-1,))
        th1d = self.th1d
        opt_coeff = self.opt_coeff
        a00, a01, a02, a03 = opt_coeff[0], opt_coeff[1], opt_coeff[2], opt_coeff[3]
        a10, a11, a12, a13 = opt_coeff[4], opt_coeff[5], opt_coeff[6], opt_coeff[7]
        th1, th2, th3 = x[0], x[1], x[2]
        h1d = a00 + a01 * th1 + a02 * th1**2 + a03 * th1**3
        h2d = -th1 + (a10 + a11 * th1 + a12 * th1**2 + a13 * th1**3) * (th1 - th1d) * (th1 + th1d)
        h1 = th3 - h1d
        h2 = th2 - h2d
        # define the system output
        y = np.array([h1, h2])

        return y