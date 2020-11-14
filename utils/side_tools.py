import numpy as np


# TODO: Define events for ODE
def hit_ground(t, x, stanceleg_coord, r):
    #  define guards (a set for triggering hybrid system transition).
    #  Define the events when the horizontal coordinate of swing leg
    #  is larger than stance leg and vertial coordinate reaches zero
    z1_stance = stanceleg_coord[0]
    z2_stance = stanceleg_coord[1]
    swingleg_coord = get_swingleg_end_coord(x, stanceleg_coord, r)
    z1_swing = swingleg_coord[0]
    z2_swing = swingleg_coord[1]

    is_swingleg_front = (z1_swing - z1_stance) > 0
    is_swingleg_ground = z2_swing == 0
    detect_signal = is_swingleg_front & is_swingleg_ground

    return detect_signal

def get_swingleg_end_coord(x, stanceleg_coord, r):
    th1 = x[0]
    th2 = x[1]
    swingleg_end_coord_z1 = stanceleg_coord[0] + r * (np.sin(th1) - np.sin(th2))
    swingleg_end_coord_z2 = stanceleg_coord[1] + r * (np.cos(th1) - np.cos(th1))

    return (swingleg_end_coord_z1, swingleg_end_coord_z2)