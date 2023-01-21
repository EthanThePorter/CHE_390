import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
import matplotlib.pyplot as plt
import openpyxl

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
    ln_gamma2 = -np.log((1-x) + x * G21) - x * (G12 / (x + (1 - x) * G12) - G21 / ((1 - x) + x * G21))
    qpred = x * ln_gamma1 + (1 - x) * ln_gamma2
    return qpred


# Initial guessed values for parameters
G0 = [0.1, 0.1]
# Fit data
popt, pcov = curve_fit(Wilson, x, qexpt, G0)
print(popt, np.diag(pcov))


# Compute 95% confidence intervals for A
alpha = 0.05
# Number of data points
n = len(x)
# Number of parameters
p = len(popt)
# Number of degrees of freedom
dof = max(0, n - p)
# Student test value for the dof and confidence level
tval = t.ppf(1.0 - alpha / 2.0, dof)

for i, p, var in zip(range(n), popt, np.diag(pcov)):
    sigma = var ** 0.5
    print('A{0}: {1} [{2}  {3}]'.format(i, p, p - sigma * tval, p + sigma * tval))

# Calculate R2
qpred = Wilson(x, popt[0], popt[1])
resid = qexpt - qpred
SSE = np.sum((qpred - qexpt) ** 2)
qm = np.mean(qexpt)
SST = np.sum((qexpt - qm) ** 2)
R2 = 1 - SSE / SST
print('R-squared: ', R2)
print('SSE: ', SSE)

fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(x, resid, 'o')
ax1.set_xlabel('Liquid mole fraction x')
ax1.set_ylabel('Residual')
ax1.set_title('Residual as a function of liquid mole fraction')
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(-0.01, 0.01)

ax2.plot(x, qexpt, 'o', x, qpred, '-')
ax2.set_xlabel('Liquid mole fraction x')
ax2.set_ylabel('Excess energy')
ax2.set_title('Measured and predicted excess energy as a function of liquid mole fraction')
ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 0.4)
ax2.legend(['Measured', 'Predicted'])
plt.show()
