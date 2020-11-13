from models.biped_walker_3link import BipedWalker3Link
import numpy as np


# Test Scripts
pi = np.pi
x0 = np.array([pi/8, -pi/8, pi/6, 0, 0, 0])
u = np.array([1,1])
walker = BipedWalker3Link()
xdot = walker.swing_dynamics(x0, u)
print(xdot)