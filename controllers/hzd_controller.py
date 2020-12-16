from optimization.opt_traj import get_opt_coeff
import numpy as np
from numpy.linalg import inv
from sympy import *

class HybridZeroDynamicsController(object):
    def __init__(self, model, effort_limit=None):
        self.opt_coeff = get_opt_coeff()
        self.model = model
        self.th1d = model.th1d
        self.th3d = model.th3d
        self.epsilon = 0.1
        self.alpha = 0.9
        self.n = model.n
        self.effort_limit = effort_limit
        # Compute all the required closed form during instantiation.
        self.Lfy, self.L2fy, self.LgLfy = self.get_symbolic_eqns()
        print("Control input closed form function derivation finished")

    def compute_cls_feedback_u(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: u --> (n-1)X1. In 3-link case --> 2X1
        # Motivation for using sympy. Calculate the closed form just once and use that
        # for evaluating any state values is faster than computing the jacobian everytime.
        # Ode takes in control input in every infinitesimal interval.

        th1, th2, th3, th1dot, th2dot, th3dot = x[0], x[1], x[2], x[3], x[4], x[5]
        LgLfy = self.LgLfy
        L2fy = self.L2fy
        Lfy = self.Lfy

        y = self.get_y(x)
        Lfy_value = Lfy(th1, th2, th3, th1dot, th2dot, th3dot)  # data type: list
        psi = (1/self.epsilon**2) * self.compute_psi_vec(y, self.epsilon * np.asarray(Lfy_value))

        L2fy_value = L2fy(th1, th2, th3, th1dot, th2dot, th3dot)  # data type: list
        LgLfy_value = LgLfy(th1, th2, th3, th1dot, th2dot, th3dot)  # data type: list
        LgLfy_inv_value = inv(np.asarray(LgLfy_value))
        u = LgLfy_inv_value @ (psi - np.asarray(L2fy_value))
        # print("My LgLfy value is: ", np.asarray(LgLfy_value))

        # Clip the control effort if it is required
        effort_limit = self.effort_limit
        if effort_limit is not None:
            u = np.clip(u, a_min=effort_limit[0], a_max=effort_limit[1])

        return u

    def compute_hzd_feedback_u(self):
        return None


    def get_f(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: f size = (n*2) X 1. In 3-link case --> 6X1

        n = self.n
        model = self.model
        D = Matrix(model.get_D(x, is_sym=True))
        C = Matrix(model.get_C(x, is_sym=True))
        G = Matrix(model.get_G(x, is_sym=True))

        f = np.block([[x[n: 2*n].reshape(n, 1)], [D.inv() @ ((-C @ x[n: 2*n]).reshape(n, 1) - G)]])

        return f

    def get_g(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: g size = (n*2) X 2. In 3-link case --> 6X2

        model = self.model
        D = Matrix(model.get_D(x, is_sym=True))
        B = Matrix(model.get_B())
        n = self.n

        g = np.block([[np.zeros((n, n-1))],
                      [D.inv() @ B]])
        return g

    def get_Lfy(self, x):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: Lfy size = (n-1) X 1. In 3-link case --> 2X1
        y = Matrix(self.get_y(x))
        dydx = y.jacobian(x)
        Lfy = dydx @ self.get_f(x)

        return Lfy

    def get_dLfydx(self, x, Lfy):
        # Params:
        # Input: Lfy must be in Matrix type defined in sympy

        dLfydx = Lfy.jacobian(x)

        return dLfydx

    def get_L2fy(self, x, dLfydx):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: Lfy size = (n-1) X 1. In 3-link case --> 2X1
        L2fy = dLfydx @ self.get_f(x)

        return L2fy

    def get_LgLfy(self, x, dLfydx):
        # Parmas:
        # Input: x --> (n*2) X 1. In 3-link case --> 6X1
        # Output: LgLfy size = (n-1) X 2. In 3-link case --> 2X2

        LgLfy = dLfydx @ self.get_g(x)

        return LgLfy

    def get_symbolic_eqns(self):
        # define state symbols
        th1, th2, th3, th1dot, th2dot, th3dot = symbols('th1 th2 th3 th1dot th2dot th3dot')
        x = np.array([th1, th2, th3, th1dot, th2dot, th3dot])

        # Get dLfydx expression for later used in L2fy and LgLfy. Only evaluated once here.
        Lfy_eqn = self.get_Lfy(x)
        Lfy = lambdify(x, Lfy_eqn)
        dLfydx = self.get_dLfydx(x, Lfy_eqn)

        # Get L2fy equation expression and function
        L2fy_eqn = self.get_L2fy(x, dLfydx)  # get the expression
        L2fy = lambdify(x, L2fy_eqn)  # get the function for L2fy

        # Get LgLfy equation expression and function
        LgLfy_eqn = self.get_LgLfy(x, dLfydx)
        LgLfy = lambdify(x, LgLfy_eqn)

        return Lfy, L2fy, LgLfy


    def compute_psi_vec(self, y, Lfy):
        # Params:
        # Input: y --> (n-1) X 1. In 3-link case --> 2 X 1
        # Input: Lfy --> (n-1) X 1. In 3-link case --> 2 X 1
        # Output: psi size = (n-1) X 1. In 3-link case --> 2X1

        n = self.n
        psi_vec = np.zeros((n-1,))
        for i in range(n-1):
            psi_vec[i] = self.compute_psi_scalar(y[i], Lfy[i])

        psi_vec = psi_vec.reshape(n-1, 1)
        return psi_vec

    def compute_psi_scalar(self, x1, x2):
        alpha = self.alpha
        phi = x1 + (1 / (2 - alpha)) * np.sign(x2) * np.abs(x2)**(2 - alpha)
        psi_scalar = -np.sign(x2) * np.abs(x2)**alpha - np.sign(phi) * np.abs(phi)**(alpha/(2 - alpha))

        return psi_scalar

    def get_y(self, x):
        # y size = (n-1) X 1. In 3-link case --> 2X1
        #TODO: generalize to multiple n-dof model

        n = self.n
        y = np.zeros((n-1,))
        th1d = self.th1d
        th3d_bias_coeff = 2  # if torso angle is too small, no motivation for going forward

        opt_coeff = self.opt_coeff
        a00, a01, a02, a03 = opt_coeff[0], opt_coeff[1], opt_coeff[2], opt_coeff[3]
        a10, a11, a12, a13 = opt_coeff[4], opt_coeff[5], opt_coeff[6], opt_coeff[7]
        th1, th2, th3 = x[0], x[1], x[2]
        h1d = a00 + a01 * th1 + a02 * th1**2 + a03 * th1**3
        h2d = -th1 + (a10 + a11 * th1 + a12 * th1**2 + a13 * th1**3) * (th1 - th1d) * (th1 + th1d)
        # Use the following two desired trajectory when doing the LgLfy checking.
        # h1d = 0
        # h2d = -th1
        h1 = th3 - h1d * th3d_bias_coeff
        h2 = th2 - h2d
        # define the system output
        y = np.array([h1, h2])

        return y