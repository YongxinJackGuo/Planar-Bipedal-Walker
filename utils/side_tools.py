import numpy as np


# TODO: Define events for ODE
def hit_ground(t, x, swingleg_end_angle):
    #  define guards (a set for triggering hybrid system transition).
    #  Define the events when the horizontal coordinate of swing leg
    #  is larger than stance leg and vertial coordinate reaches zero
    #  However, the method is not used here since the swing leg will
    #  always touch or penetrate the ground. Instead, we monitor the
    #  end angle of the swing leg has been reached or not.

    is_hit = x[1] == swingleg_end_angle
    detect_signal = not is_hit

    return detect_signal

def get_swingleg_end_coord(x, stanceleg_coord, r):
    th1 = x[0]
    th2 = x[1]
    swingleg_end_coord_z1 = stanceleg_coord[0] + r * (np.sin(th1) - np.sin(th2))
    swingleg_end_coord_z2 = stanceleg_coord[1] + r * (np.cos(th1) - np.cos(th1))

    return (swingleg_end_coord_z1, swingleg_end_coord_z2)

def test_LgLfy(x, m, mh, mt, l, r):
    # Compute the guaranteed correct LgLfy value from closed form dynamics based on Dr.Grizzles' Literature
    # This functions serves as a helper tool that checks the sympy implementation in hzd_controller is correct
    th1, th2, th3 = x[0], x[1], x[2]
    c12 = np.cos(th1 - th2)
    c13 = np.cos(th1 - th3)

    r11 = ((m*r**3)/4) * ((5/4)*m*r + mh*r + mt*r - m*r*c12**2 + mt*l*c13)
    r12 = ((m*r**3)/4) * ((5/4)*m*r + mh*r + mt*r - m*r*c12**2 + 2*mt*l*c12*c13)
    r21 = (-m*mt*l*r**2/4) * (1 + 2*c12) * (r*c13 + l)
    r22 = (-mt*l*r**2/4) * (5*m*l + 4*mh*l + 4*mt*l + m*r*c13 + 2*m*r*c12*c13 - 4*mt*l*c13**2 + 2*m*l*c12)

    detD = (m*mt*r**4*l**2/4) * ((5/4)*m + mh + mt - m*c12**2 - mt*c13**2)

    LgLfy = (1/detD) * np.array([[r11, r12],
                                 [r21, r22]])

    return LgLfy


def hit_ground_for_5link():
    # z1_stance = stanceleg_coord[0]
    # z2_stance = stanceleg_coord[1]
    # swingleg_coord = get_swingleg_end_coord(x, stanceleg_coord, r)
    # z1_swing = swingleg_coord[0]
    # z2_swing = swingleg_coord[1]

    # is_swingleg_front = (z1_swing - z1_stance) > 0
    # is_swingleg_ground = z2_swing == 0
    # legs_not_same_place = x[0] != x[1]
    # print("z1 swing is: ", sz1_swing, "z1 stance is: ", z1_stance, " is swingleg front: ", is_swingleg_front)
    # print("is swingleg_ground: ", is_swingleg_ground)
    # print("are legs not at the same postion: ", legs_not_same_place)
    # print("======================================================")

    return None
