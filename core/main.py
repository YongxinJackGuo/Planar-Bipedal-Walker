from models.biped_walker_3link import BipedWalker3Link
import numpy as np
from controllers.hzd_controller import HybridZeroDynamicsController
from utils import side_tools
from sandbox.biped_walker_sim import Simulator
from matplotlib import pyplot as plt
from utils import plotter


pi = np.pi
torso_disturbance = pi/6
x0 = np.array([pi/8, -pi/8, pi/4, 5, -5, 0])
tf = 3
u_limit = (-150, 150)

#----------------Simulate the dynamics----------------
walker = BipedWalker3Link()
hzd_controller = HybridZeroDynamicsController(walker, effort_limit=u_limit)
walkerSim = Simulator(walker, hzd_controller, walker_type="3link")
x, u, x_minus, stanceleg_coord, t = walkerSim.simulate_full_model(x0, tf)

#----------------Plot or Animation-------------------
# Create the animation for display or save
walkerSim.animate(mass_center_size=0.06, mass_center_color='b', link_width=2.0,
                  link_color='g', save=False, display=True)

# plot the return map
# plotter.plot_poincare_map(x_minus)







