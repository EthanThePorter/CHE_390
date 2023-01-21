import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt

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


# Define list of diameters to calculate variables for (inches) to (m)
D_array_inches = np.arange(1, 4.25, 0.25)
D_array = D_array_inches * 2.54 / 100
# Initialize variables to store values for AOC, ACC, TOC
AOC_list = []
ACC_list = []
TAC_list = []
# Iterate through diameter array to calculate economic values
for D in D_array:
    # Define initial guesses
    q0 = [0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
    # Solve system of equations for section flow rates
    q = fsolve(FlowRate, q0, args=(D), xtol=1e-8)

    # Define constants for pump efficiency calculations
    # Pump efficiencies
    pump_efficiency = 0.6
    pump_motor_efficiency = 0.4
    # Calculate total power requirement (W) to (kW)
    W = np.sum(q * Pressure(q, L, D)) / pump_efficiency / pump_motor_efficiency / 1000

    # Economic variables
    # Total number of operation hours (hours/year)
    N_op = 8420
    # Price of electricity ($/kWh)
    C_e = 0.105
    # Purchase price of pipe ($/ft) to ($/m)
    C_o = 5.92 * 3.281
    # Factor to account for fittings, installation, finance, etc.
    F = 1.0
    # Standard size of pipe (inches) to (m)
    D_o = 1 * 2.54 / 100
    # Pipe size cost exponent
    m = 1.25
    # Factors for maintenance and repairs
    a = 0.12
    b = 0.12
    # Calculate annualized operation cost (AOC)
    AOC = W * N_op * C_e
    # Calculate annualized capital cost (ACC)
    ACC = (1 + F) * C_o * (D / D_o) ** m * (a + b) * np.sum(L)
    # Calculate total annualized cost (TAC)
    TAC = AOC + ACC

    # Append values to main lists
    AOC_list.append(AOC)
    ACC_list.append(ACC)
    TAC_list.append(TAC)


# Output minimum TAC
min_TAC = np.min(TAC_list)
economic_diameter = D_array_inches[TAC_list.index(min_TAC)]
print(f'Minimum TAC is: {np.round(min_TAC, 2)} when {economic_diameter}" pipe is used.')

# Initialize plots
fig, axs = plt.subplots(3, sharex=True)
# Plot AOC
axs[0].plot(D_array_inches, AOC_list)
axs[0].set_ylabel('$')
axs[0].title.set_text('AOC')
axs[0].grid()
# Plot ACC
axs[1].plot(D_array_inches, ACC_list)
axs[1].set_ylabel('$')
axs[1].title.set_text('ACC')
axs[1].grid()
# Plot TAC
axs[2].plot(D_array_inches, TAC_list)
axs[2].set_xlabel('Pipe Diameter (inches)')
axs[2].set_ylabel('$')
axs[2].title.set_text('TAC')
axs[2].axvline(economic_diameter, color='red')
axs[2].grid()
plt.show()
