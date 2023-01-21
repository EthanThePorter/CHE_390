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
    gamma1 = np.log(-np.log(x + (1 - x) * G12) + (1 - x) * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21)))
    gamma2 = np.log(-np.log((1 - x) + x * G21) - x * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21)))
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
    print(gamma1, gamma2)
    return P_total - gamma1 * x1 * P1 / y1, P_total - gamma2 * x2 * P2 / y2



# Define total system pressure (kPa)
P_total = 32.86
# Define points to plot T-xy and xy data
x_array = np.arange(0, 1, 0.05)
# Get parameters for Wilson equation
G = getWilsonParameters()
# Initialize list to save data to
T_list = []
y_list = []
# Initialize guess for y and T
initial_guess = [325, 0.1]
# # Get VLE data
# for x in x_array:
#     T, y = fsolve(get_VLE_data, x0=initial_guess, args=(P_total, G, x))
#     T_list.append(T)
#     y_list.append(y)
# # Plot
# plt.plot(x_array, T)
# plt.plot(y_list, T)
# plt.show()

x = 0.063
print(x)
parameters = fsolve(get_VLE_data, x0=initial_guess, args=(P_total, G, x))
print(get_VLE_data(parameters, P_total, G, x))
print(parameters)
