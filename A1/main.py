from scipy.optimize import fsolve
from scipy.integrate import solve_ivp, solve_bvp
import numpy as np
import matplotlib.pyplot as plt

# Length of section
L = 0.01
# Number of nodes
n = 10
# Division size
dx = L / (n - 1)
# Constants
D = 1E-3 / 100 ** 2
k = 0.01 / 1000
CA0 = 1 * 1000


def f(c):
    # Initialize list to store values
    z = np.zeros(n - 1)
    # At first known point:
    z[0] = D / dx ** 2 * (c[1] - 2 * c[0] + CA0) - k * c[0] ** 2
    # For middle points
    for i in range(1, n - 2):
        z[i] = D / dx ** 2 * (c[i + 1] - 2 * c[i] + c[i - 1]) - k * c[i] ** 2
    # For end point
    z[-1] = 1 / (2 * dx) * (3 *c[-1] - 4 * c[-2] + c[-3])
    return z

# Solve system of equations with fsolve
t = np.ones(n - 1)
y = fsolve(f, t)
C = np.concatenate(([1], y))
x = np.linspace(0, L, n)


# Solve DE with BVP
def system(x, C):
    C1 = C[1]
    C2 = k / D * C[0] ** 2
    return C1, C2

def BC(A, B):
    return np.array([A[0] - CA0, B[1]])

# Solve BVP
y0 = np.zeros((2, n))
y0[0] = 1
sol_bvp = solve_bvp(system, BC, np.linspace(0, 0.01, n), y0)

# # Solve IVP
# sol_ivp = solve_ivp(system, [0, 0.01], [CA0, 0], t_eval=np.linspace(0, 0.01, n))
#
# print(sol_bvp.y[0])

plt.plot(x, C, label='FD Method')
plt.plot(np.linspace(0, 0.01, n), sol_bvp.y[0], '--', label='solve_bvp')
plt.legend()
plt.grid()
plt.show()
