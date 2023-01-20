import numpy as np
import pandas as pd
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt

# Define constants - Subscript W for water, G for ethylene glycol.
# Fractions
x_G = 0.6
x_W = 0.4
# Densities (kg/m3)1
p_G = 1113
p_W = 998
p = p_G * x_G + p_W * x_W
# Viscosity (Pa.s)
u_G = 0.0161
u_W = 0.0010
u = u_G * x_G + u_W * x_W
# Roughness (m)
e = 5.01e-5
# Pipe lengths (m)
L_01 = 260
L_12, L_23, L_45, L_67 = 450, 450, 450, 450
L_24, L_35, L_46, L_57 = 400, 400, 400, 400
L_14 = 600
L = [L_01, L_12, L_14, L_23, L_24, L_35, L_45, L_46, L_57, L_67]
# Pipe diameter (m)
D = 1 * 2.54 / 100
# Flow rates (L/min) to (L/s)
F_in = 78 * 4 / 60
F_out = 78 / 60


def FrictionFactor(v):
    # Calculate Reynolds numbers
    Re = p * v * D / u
    # Churchill Equation Parameters
    A_ = (2.457 * np.log(1/((7/Re)**0.9 + 0.27*(e/D))))**16
    B_ = (37530/Re)**16
    # Solve Churchill equation and return f
    return 2 * ((8/Re)**12 + 1/(A_ + B_)**(3/2))**(1/12)


def P(q, L):
    """
    Pressure drop equation
    :param q: Flowrate in (m3/s)
    :param L: Length of pipe (m)
    :return: Pressure drop in pascal
    """
    # Convert q to m3/min
    q = q * 0.001
    # Get cross-sectional area of pipe
    A = np.pi * (D / 2) ** 2
    # Get fluid velocity in m/s
    v = q / A
    # Get friction factor from equation
    f = FrictionFactor(v)
    # Calculate pressure drop
    return 2 * f * p * v**2 * L / D


def FlowRate(q, D):
    # Initialize empty list of values for overall flow rates
    F = np.zeros(10)
    # Define node mass balances
    F[0] = q[0] - q[1] - q[2]
    F[1] = q[1] - q[3] - q[4]
    F[2] = q[3] - q[5] - F_out
    F[3] = q[2] + q[4] - q[6] - q[7]
    F[4] = q[5] + q[6] - q[8] - F_out
    F[5] = q[8] - q[9] - F_out
    F[6] = q[8] + q[9] - F_out

    # Initialize list to store pressure drops
    # Get pressure drops
    F[7] = P(q[1], L[1]) + P(q[4], L[4]) - P(q[2], L[2])
    F[8] = P(q[4], L[4]) + P(q[5], L[5]) - P(q[4], L[4]) - P(q[6], L[6])
    F[9] = P(q[6], L[6]) + P(q[8], L[8]) + P(q[7], L[7]) + P(q[9], L[9])

    return F


#initial guesses
q0 = [0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]

q = fsolve(FlowRate,q0,args=(D), xtol= 1e-8)

print(q)









