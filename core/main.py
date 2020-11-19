from models.biped_walker_3link import BipedWalker3Link
import numpy as np
from controllers.hzd_controller import HybridZeroDynamicsController
from utils import side_tools
from sandbox.biped_walker_sim import Simulator

# Test Scripts --> Test control output u


#xdot = walker.swing_dynamics(x0, u)


pi = np.pi
x0 = np.array([pi/8, -pi/8, pi/6, 2, 2, 0])
tf = 3
# u = np.array([1,1])
walker = BipedWalker3Link()
hzd_controller = HybridZeroDynamicsController(walker)
walkerSim = Simulator(walker, hzd_controller)
x, u, stanceleg_coord, t = walkerSim.simulate_full_model(x0, tf)
print("stance leg coordinate are: ", stanceleg_coord)

# print("States Simulated: ", x)
# print("Control Imposed:", u)


# m = 5
# mh = 15
# mt = 10
# r = 1
# l = 0.5
# LgLfy = side_tools.test_LgLfy(x0, m, mh, mt, l, r)
# print("The correct LgLfy is: ", LgLfy)





