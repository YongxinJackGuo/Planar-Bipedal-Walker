from models.biped_walker_3link import BipedWalker3Link
import numpy as np
from controllers.hzd_controller import HybridZeroDynamicsController
from utils import side_tools

# Test Scripts --> Test control output u


#xdot = walker.swing_dynamics(x0, u)
# m = 5
# mh = 15
# mt = 10
# r = 1
# l = 0.5
# LgLfy = side_tools.test_LgLfy(x0, m, mh, mt, l, r)
# print("The correct LgLfy is: ", LgLfy)

pi = np.pi
x0 = np.array([pi/8, -pi/8, pi/6, 1, 0.2, 0.2])
# u = np.array([1,1])
walker = BipedWalker3Link()
controller = HybridZeroDynamicsController(walker)
u = controller.compute_cls_feedback_u(x0)
print(u)





