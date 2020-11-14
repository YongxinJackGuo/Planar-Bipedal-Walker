from models.biped_walker_3link import BipedWalker3Link
import numpy as np
from controllers.hzd_controller import HybridZeroDynamicsController

# Test Scripts --> Test control output u
pi = np.pi
x0 = np.array([pi/8, -pi/8, pi/6, 0, 0, 0])
# u = np.array([1,1])
walker = BipedWalker3Link()
controller = HybridZeroDynamicsController(walker)
u = controller.compute_cls_feedback_u(x0)
#xdot = walker.swing_dynamics(x0, u)
print(u)





