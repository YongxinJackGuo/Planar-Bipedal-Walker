from models.biped_walker_3link import BipedWalker3Link
import numpy as np
from controllers.hzd_controller import HybridZeroDynamicsController
from utils import side_tools
from sandbox.biped_walker_sim import Simulator



pi = np.pi
x0 = np.array([pi/8, -pi/8, pi/4, 5, -5, 0])
tf = 3
walker = BipedWalker3Link()
hzd_controller = HybridZeroDynamicsController(walker)
walkerSim = Simulator(walker, hzd_controller, walker_type="3link")
x, u, stanceleg_coord, t = walkerSim.simulate_full_model(x0, tf)
walkerSim.animate(mass_center_size=0.06, mass_center_color='b', link_width=2.0, link_color='g', save=False)







