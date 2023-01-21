import numpy as np
import pandas as pd
from scipy.optimize import fsolve, curve_fit

# Define constants - Subscript W for water, G for ethylene glycol.
# Fractions
x_G = 0.6
x_W = 0.4
# Densities (kg/m3)
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
L = np.array([260, 450, 450, 450, 450, 400, 400, 400, 400, 600])
# Pipe diameter (m)
D = 1 * 2.54 / 100
# Flow rates (L/min) to (m3/s)
F_in = 78 * 4 / 1000 / 60
F_out = 78 / 1000 / 60


def FrictionFactor(q, D):
    # Calculate Reynolds numbers
    Re = 4 * p * q / np.pi / u / D
    # Define Colebrook white equation and initial guess
    CW = lambda f, Re_: -4*np.log10(e/D + 4.647/(Re_ * np.sqrt(f))) + 2.28 - 1 / np.sqrt(f)
    f0 = [0.001]
    # If Reynolds number less than 3000, use Hagen-Poiseuille equation, else use CW equation
    return np.array([16 / Re_ if Re_ < 3000 else fsolve(CW, f0, args=(Re_,))[0] for Re_ in Re])


def Pressure(q, L, D):
    # Get friction factor from equation
    f = FrictionFactor(q, D)
    # Calculate pressure drop
    return 2 * f * p * (4 * q / np.pi / D ** 2) ** 2 * L / D


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
    P = Pressure(q, L, D)
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
    # Calculate energy balance for each loop
    F[7] = P12 + P24 - P14
    F[8] = P23 + P35 - P24 - P45
    F[9] = P45 + P57 - P46 - P67
    return F


# Define initial guesses
q0 = [0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
# Solve system of equations
q = fsolve(FlowRate, q0, args=(D), xtol=1e-8)
# Calculate pressure drop
DP = Pressure(q, L, D) / 1000  # pressure drop (kPa)

# Output results
df = pd.DataFrame()
df['Pipe length (m)'] = L
df['Flow rate (m3/s)'] = q
df['Pressure drop (kPa)'] = DP
print(df)
