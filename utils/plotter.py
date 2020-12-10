from matplotlib import pyplot as plt
import numpy as np

def plot_poincare_map(x_minus):
    # params:
    # input: x, a list of ndarray

    # convert list of ndarray to 2D ndarray
    x_minus = np.asarray(x_minus)
    q1dot_minus = x_minus[:, 3]

    # plot the return map
    fig = plt.figure()
    plt.plot(q1dot_minus[:-2], q1dot_minus[1:-1], color='r')
    plt.plot(q1dot_minus[:-2], q1dot_minus[:-2], color='k')
    plt.legend(['q1dot prior to impact', 'reference line'])
    plt.xlabel('joint velocity of stance leg q1dot (rad/s)')
    plt.ylabel('return map function P(q1dot) (rad/s)')
    plt.title('Return Map of Stance Leg Joint Velocity')
    plt.show()

    return None