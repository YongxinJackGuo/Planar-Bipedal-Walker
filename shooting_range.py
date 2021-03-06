# from pydrake.forwarddiff import jacobian
from pydrake.autodiffutils import AutoDiffXd

import math
import autograd.numpy as np
from autograd import jacobian
from numpy.linalg import inv
from matplotlib import pyplot as plt
import matplotlib.animation as animation



def f(t):
    f = inv(np.array([[1.5 - 0.5 * (t - 2.5), 2 * t[0]],
              [2 * t[1] ** 3, t[0] ** 2]]))
    return f
def g(t):
    t1 = np.sin(t[0])
    t2 = np.cos(t[1])
    g = np.array([])
    g = np.array([0.5 - 0.5 *(t1 - 2.5) + t2**3, 5.0 + t1**2 + t2**3, 2.0 + t2**2])
    return g

x = np.array([np.pi/3, np.pi/3, 0, 0])

def gg(t):
    a = np.array([[2*t[0]**3, 3 * np.cos(t[1]), 3],
                  [12+t[1]**2, 2, 3],
                  [t[0]**3, np.cos(t[0])*t[1], 2]]);
    gg = jacobian(g)
    return gg(t) @ inv(a) @ np.array([2,3,4])

def ggg(t):
    a = np.array([2 * t[0] ** 3, 3 * np.cos(t[1]), 3, 4]);
    temp = jacobian(gg)
    return temp(t) @ a





def calc(th1):
    return 0.512 + 0.073 * th1 + 0.035 * th1**2 - 0.819*th1**3

th1 = np.linspace(-np.pi/8, np.pi/8, 100)

th3 = calc(th1)

fig1 = plt.figure()



def update(frame):
    line = plt.Line2D((2*frame, 8), (6*frame, 6), lw=2.5, color='r')
    line1 = plt.Line2D((-2*frame, 8), (-6*frame, 6), lw=2.5, color='g')
    plt.gca().add_line(line)
    plt.gca().add_line(line1)
    plt.legend("test")
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    return plt


ani = animation.FuncAnimation(fig1, update, frames=[i for i in range(10)], blit=False, repeat=False, interval=10)
plt.show()