import numpy as np
import pandas as pd
from scipy.optimize import fsolve, curve_fit

xG = 0.6
xW = 0.4
pG = 1113
pW = 998
p = pG * xG + pW * xW
uG = 0.0161
uW = 0.0010
viscosity = uG * xG + uW * xW
epsilon = 5.01e-5
L = np.array([260, 450, 600, 450, 400, 400, 450, 400, 400, 450])
D = 1 * 2.54 / 100
qef = 78 / 1000 / 60


def Pressure(q, L):
    Re = 4 * p * q / np.pi / viscosity / D
    A = (2.457 * np.log(1 / ((7 / Re) ** 0.9 + 0.27 * (epsilon / D)))) ** 16
    B = (37530 / Re) ** 16
    f = 2 * ((8 / Re) ** 12 + 1 / (A + B) ** (3 / 2)) ** (1 / 12)
    return 2 * f * p * (4 * q / np.pi / D ** 2) ** 2 * L / D


def FlowRate(q, D):
    F = np.zeros(10)
    F[0] = q[0] - q[1] - q[2]
    F[1] = q[1] - q[3] - q[4]
    F[2] = q[3] - q[5] - qef
    F[3] = q[2] + q[4] - q[6] - q[7]
    F[4] = q[5] + q[6] - q[8] - qef
    F[5] = q[7] - q[9] - qef
    F[6] = q[8] + q[9] - qef

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


q0 = [0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
q = fsolve(FlowRate, q0, args=(D), xtol=1e-8)
DP = Pressure(q, L) / 1000  # pressure drop (kPa)

df = pd.DataFrame()
df['Pipe length (m)'] = L
df['Flow rate (m3/s)'] = q
df['Pressure drop (kPa)'] = DP
print(df)