import numpy as np
from scipy.integrate import solve_ivp
from utils import side_tools
from .animator import WalkerAnimator
from matplotlib import pyplot as plt

class Simulator(object):
    def __init__(self, model, controller, walker_type):
        self.model = model
        self.controller = controller
        self.x = []
        self.u = []
        self.stanceleg_coord = []
        self.t = []
        self.tf = 0
        self.E = []
        self.v = []
        self.cost = []
        self.walker_type = walker_type
        self.step_interval_sample_count = []

    def simulate_full_model(self, x0, tf):
        model = self.model
        controller = self.controller
        swingleg_end_angle = model.swingleg_end_angle
        r = model.r
        hit_threshold = model.hit_threshold

        impact_times = 0
        t0 = 0.0
        dt = 0.01
        x = [x0]
        u = []  # get the controller value at x
        t = [t0]
        E = [(0, 0, 0)]  # energy list storing tuples of (kinetic, potential and total energy)
        v = [0]  # walking speed store list
        cost = [0]
        stanceleg_coord = [(0, 0)]
        step_interval_sample_count = []  # counts how many data points at each step interval
        x_minus = []  # states right before impact, used for plotting return map

        curr_x = x0
        curr_t = t0
        curr_stanceleg_coord = (0, 0)

        while curr_t < tf:
            hit_flag = False  # turn off the flag until impact comes
            curr_u = controller.compute_cls_feedback_u(curr_x)  # Feedback Control

            # define an autonomous ODE
            def f(t, x):
                return model.swing_dynamics(x, curr_u)

            def hit_ground(t, x):
                return side_tools.hit_ground(t, x, swingleg_end_angle, hit_threshold)
            #  the events should terminate the ODE solver when event is reached
            hit_ground.terminal = 1

            # integrate the dynamics within dt period
            sol = solve_ivp(f, (0, dt), curr_x, events=hit_ground)

            curr_t += sol.t[-1]  # update time step
            curr_x = sol.y[:, -1]  # update initial state for next ode integration
            t.append(curr_t)
            x.append(curr_x)
            u.append(curr_u)
            E.append(self.compute_energy(curr_x))
            v.append(model.get_walking_speed(curr_x))
            cost.append(sum(curr_u**2))
            # TODO: impact model when events happened (define the resets)
            if len(sol.t_events[0]) != 0:  # impact happens
                z1_stance = side_tools.get_swingleg_end_coord(curr_x, curr_stanceleg_coord, r)[0]
                curr_stanceleg_coord = (z1_stance, 0)
                stanceleg_coord.append(curr_stanceleg_coord)
                step_interval_sample_count.append(len(x))
                x_minus.append(x[-1])  # the last state of current state sequence is the x_minus
                # impact map, resets the state
                curr_x = model.impact_dynamics(curr_x)
                impact_times += 1
                hit_flag = True  # turn on flag before it gets into next continuous dynamics phase.
                print("Impact happens", impact_times, "time(s) at", curr_t, "seconds")

        # incoporate the last interval if it did not encounter impact in last dt second
        if hit_flag == False:
            step_interval_sample_count.append(len(x))
        self.step_interval_sample_count = step_interval_sample_count  # for animation

        # update class member
        self.x, self.u, self.stanceleg_coord, self.t = x, u, stanceleg_coord, t
        self.tf, self.E, self.v, self.cost = tf, E, v, cost

        # print some meaningful information
        print("stance leg coordinate are: ", stanceleg_coord)

        return x, u, x_minus, stanceleg_coord, t

    def compute_energy(self, x):
        # a function that returns the kinetic, potential and total energy of the robot at current moment
        model = self.model
        energy = model.get_energy(x)  # get kinetic, potential and total energy

        return energy

    def animate(self, mass_center_size, mass_center_color, link_width, link_color, save, display):
        biped_walker_animator = WalkerAnimator(self.model, self.walker_type, mass_center_size,
                                               mass_center_color, link_width, link_color)
        biped_walker_animator.animate(self.x, self.stanceleg_coord, self.step_interval_sample_count,
                                      self.tf, self.E, self.v, self.cost, save, display)

        return None


    def simulate_zero_dynamics(self):

        return None

