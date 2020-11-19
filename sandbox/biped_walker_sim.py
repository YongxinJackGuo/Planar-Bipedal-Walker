import numpy as np
from scipy.integrate import solve_ivp
from utils import side_tools
from .animation import animate

class Simulator(object):
    def __init__(self, model, controller):
        self.model = model
        self.controller = controller

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
        stanceleg_coord = [(0, 0)]

        curr_x = x0
        curr_t = t0
        curr_stanceleg_coord = (0, 0)

        while curr_t < tf:
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
            # TODO: impact model when events happened (define the resets)
            if len(sol.t_events[0]) != 0:  # impact happens
                z1_stance = side_tools.get_swingleg_end_coord(curr_x, curr_stanceleg_coord, r)[0]
                curr_stanceleg_coord = (z1_stance, 0)
                stanceleg_coord.append(curr_stanceleg_coord)
                # impact map, resets the state
                curr_x = model.impact_dynamics(curr_x)
                impact_times += 1
                print("Impact happens", impact_times, "time(s) at", curr_t, "seconds")

        return x, u, stanceleg_coord, t

    def simulate_zero_dynamics(self):

        return None

