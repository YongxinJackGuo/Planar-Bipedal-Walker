import numpy as np # import autograd version of numpy for jacobian calculation in controller.
from numpy.linalg import inv
import scipy
from sympy import *
from utils import side_tools


class BipedWalker5Link(object):
    def __init__(self):
        pass

    def swing_dynamics(self, x, u):

        xdot = None
        #TODO: Write continuous dynamics
        return xdot

    def impact_dynamics(self, x):
        x_copy = x.copy()

        #TODO: Write discrete dynamics

        return x_copy

    def relabel(self, x):

        #TODO: Perform relabelling

        return None

    def get_D(self, x, is_sym):

        #TODO: Write D matrix
        D = None

        return D

    def get_C(self, x, is_sym):

        #TODO: Write D matrix
        C = None

        return C

    def get_G(self, x, is_sym):

        #TODO: Write D matrix
        G = None

        return G

    def get_B(self):
        
        #TODO: Write B matrix
        B = None

        return B

    def get_De(self, x):

        # TODO: Write De matrix
        De = None

        return De


