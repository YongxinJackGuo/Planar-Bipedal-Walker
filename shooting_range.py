from pydrake.forwarddiff import jacobian
from pydrake.autodiffutils import AutoDiffXd

import math
import numpy as np
from matplotlib import pyplot as plt


def f(t):
    return np.array([1.5 - 0.5 *(t - 2.5)])
def g(t):
    t1 = np.sin(t[0])
    t2 = np.cos(t[1])
    return np.array([1.5 - 0.5 *(t1 - 2.5) + t2**3, 5.0 + t1**2 + t2**3, 2.0 + t2**2])

x = np.array([np.pi/3, np.pi/3, 0, 0])
def z(x):
    print(x)
    z = jacobian(g, x)

j = jacobian(z, x)
print(j)


# def calc(th1):
#     return 0.512 + 0.073 * th1 + 0.035 * th1**2 - 0.819*th1**3
#
# th1 = np.linspace(-np.pi/8, np.pi/8, 100)
#
# th3 = calc(th1)
#
# plt.figure()
# plt.ylim([0, 180])
# plt.plot(th1 * 180/np.pi, th3 * 180/np.pi)
# plt.show()