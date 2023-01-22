import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt
import openpyxl


def getWilsonParameters():
    # Read data
    data = pd.read_excel('./data/VLEData_EthanolWater1.xlsx', 'Sheet1')
    # Parse data
    Td = data.iloc[1:21, 0]
    xd = data.iloc[1:21, 1]
    yd = data.iloc[1:21, 2]
    # Convert data to arrays
    T = np.asarray(Td, dtype=float)
    x = np.asarray(xd, dtype=float)
    y = np.asarray(yd, dtype=float)

    # Total pressure in kPa
    Pt = 101.325
    # Vapor pressure of pure ethanol
    P1 = 10 ** (7.28781 - 1623.22 / (T - 44.170))
    # Vapor pressure of pure water
    P2 = 10 ** (7.19621 - (1730.63 / (T - 39.724)))
    # Calculate activity coefficients
    gamma1 = Pt * y / (x * P1)
    gamma2 = Pt * (1 - y) / ((1 - x) * P2)
    # Calculating experimental excess energy for regression
    qexpt = x * np.log(gamma1) + (1 - x) * np.log(gamma2)
    # Define gas constant
    R = 8.314

    # function for calculating excess energy based on Wilson equation
    def Wilson(x, G12, G21):
        ln_gamma1 = -np.log(x + (1 - x) * G12) + (1 - x) * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21))
        ln_gamma2 = -np.log((1 - x) + x * G21) - x * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21))
        qpred = x * ln_gamma1 + (1 - x) * ln_gamma2
        return qpred

    # Initial guessed values for parameters
    G0 = [0.1, 0.1]
    # Fit data to get parameters for Wilson equation
    G, covariance = curve_fit(Wilson, x, qexpt, G0)
    return G


def WilsonEquations(x, G12, G21):
    gamma1 = np.exp(-np.log(x + (1 - x) * G12) + (1 - x) * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21)))
    gamma2 = np.exp(-np.log((1 - x) + x * G21) - x * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21)))
    return gamma1, gamma2


def get_VLE_data(parameters, P_total, G, x):
    # Get parameters to solve for
    T = parameters[0]
    y = parameters[1]
    # Get activity coefficients
    gamma1, gamma2 = WilsonEquations(x, *G)
    # Get mole fractions
    x1 = x
    x2 = 1 - x
    y1 = y
    y2 = 1 - y
    # Vapor pressure of pure ethanol
    P1 = 10 ** (7.28781 - 1623.22 / (T - 44.170))
    # Vapor pressure of pure water
    P2 = 10 ** (7.19621 - (1730.63 / (T - 39.724)))
    return P_total - gamma1 * x1 * P1 / y1, P_total - gamma2 * x2 * P2 / y2


# Define total system pressure (kPa)
P_total = 32.86
# Define points to plot T-xy and xy data
x_array = np.arange(0.001, 0.999, 0.001)
# Get parameters for Wilson equation
G = getWilsonParameters()
# Initialize list to save data to
T_list = []
y_list = []
# Get VLE data
for x in x_array:
    T, y = fsolve(get_VLE_data, x0=(344, x), args=(P_total, G, x))
    T_list.append(T)
    y_list.append(y)

# Get VLE data from literature
data = pd.read_excel('./data/VLEData_EthanolWater2.xlsx', 'Sheet1')
# Parse data
Td = data.iloc[1:21, 0]
xd = data.iloc[1:21, 1]
yd = data.iloc[1:21, 2]
# Convert data to arrays
T_actual = np.asarray(Td, dtype=float)
x_actual = np.asarray(xd, dtype=float)
y_actual = np.asarray(yd, dtype=float)

# Initialize plots
fig, axs = plt.subplots(1, 2)
# Plot TXY
axs[0].plot(x_array, T_list, label='Predicted Bubble Point Curve')
axs[0].plot(y_list, T_list, label='Predicted Dew Point Curve')
axs[0].plot(x_actual, T_actual, label='Actual Bubble Point Curve')
axs[0].plot(y_actual, T_actual, label='Actual Dew Point Curve')
axs[0].set_title('T-XY Diagram for Ethanol-Water System at 32.86kPa')
axs[0].set_xlabel('X or Y')
axs[0].set_ylabel('T (K)')
axs[0].set_xlim(0, 1)
axs[0].legend()
axs[0].grid()
# Plot XY
axs[1].plot([0, 1], [0, 1], '--', label='Y=X')
axs[1].plot(x_array, y_list, label='Eqm. Data')
axs[1].set_title('XY Diagram at 32.86kPa')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].legend()
plt.show()
