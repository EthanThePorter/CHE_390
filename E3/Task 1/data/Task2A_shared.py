import numpy as np
import pandas as pd
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt

# Define constants - Subscript W for water, G for ethylene glycol.
# Fractions
x_G = 0.6
x_W = 0.4
# Densities (kg/m3)
p_G = 1113
p_W = 998
density = p_G * x_G + p_W * x_W
# Viscosity (Pa.s)
u_G = 0.0161
u_W = 0.0010
viscosity = u_G * x_G + u_W * x_W
# Roughness (m)
roughness = 5.01e-5
# Pipe lengths (m)
L = np.array([260, 450, 450, 450, 450, 400, 400, 400, 400, 600])
# Pipe diameter (m)
D = 1 * 2.54 / 100
# Flow rates (L/min) to (m3/s)
F_in = 78 * 4 / 1000 / 60
F_out = 78 / 1000 / 60


def Pressure(q, L):
    Re = 4 * density * q / np.pi / viscosity / D
    A = (2.457 * np.log(1 / ((7 / Re) ** 0.9 + 0.27 * (roughness / D)))) ** 16
    B = (37530 / Re) ** 16
    f = 2 * ((8/Re)**12 + 1/(A + B)**(3/2))**(1/12)
    return 2 * f * density * (4 * q / np.pi / D ** 2) ** 2 * L / D


def FlowRate(q, D):
    # Initialize empty list of values for overall flow rates
    F = np.zeros(10)
    # Define node mass balances
    F[0] = q[0] - q[1] - q[2]
    F[1] = q[1] - q[3] - q[4]
    F[2] = q[3] - q[5] - F_out
    F[3] = q[2] + q[4] - q[6] - q[7]
    F[4] = q[5] + q[6] - q[8] - F_out
    F[5] = q[7] - q[9] - F_out
    F[6] = q[8] + q[9] - F_out

    # Get pressure drops
    P = Pressure(q, L)
    # Parse
    P12 = P[1]
    P14 = P[2]
    P23 = P[3]
    P24 = P[4]
    P35 = P[5]
    P45 = P[6]
    P46 = P[7]
    P57 = P[8]
    P67 = P[9]

    F[7] = P12 + P24 - P14
    F[8] = P23 + P35 - P24 - P45
    F[9] = P45 + P57 - P46 - P67

    return F


#initial guesses
q0 = [0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]

q = fsolve(FlowRate, q0, args=(D), xtol= 1e-8)

print(q)









