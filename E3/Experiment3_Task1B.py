# Experiment3_Task1a_v1
# Function to use curve_fit to determine Wilson model coefficients 

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
import matplotlib.pyplot as plt
import openpyxl

data = pd.read_excel('./data/VLEData_EthanolWater1.xlsx', 'Sheet1')

Td = data.iloc[1:21, 0]
xd = data.iloc[1:21, 1]
yd = data.iloc[1:21, 2]

# convert objects to np arrays
T = np.asarray(Td, dtype=float)
x = np.asarray(xd, dtype=float)
y = np.asarray(yd, dtype=float)

Pt = 101.325  # Total pressure in kPa

# Vapor pressure of pure ethanol
P1 = 10 ** (7.28781 - 1623.22 / (T - 44.170))
# Vapor pressure of pure water
P2 = 10 ** (7.19621 - (1730.63 / (T - 39.724)))

gamma1 = Pt * y / (x * P1)  # calculating measured activity coefficient of ethanol
gamma2 = Pt * (1 - y) / ((1 - x) * P2)

# Calculating experimental excess energy for regression
qexpt = x * np.log(gamma1) + (1 - x) * np.log(gamma2)

R = 8.314


# function for calculating excess energy based on Wilson equation
def NRTL(x, a, g12, g21):
    t12 = g12 / (R * T)
    t21 = g21 / (R * T)

    G12 = np.exp(-a * t12)
    G21 = np.exp(-a * t21)

    x1 = x
    x2 = 1 - x

    y1 = x2 ** 2 * (t21 * (G21 / (x1 + x2 * G21)) ** 2 + t12 * G12 / (x2 + x1 * G12) ** 2)
    y2 = x1 ** 2 * (t12 * (G12 / (x2 + x1 * G12)) ** 2 + t21 * G21 / (x1 + x2 * G21) ** 2)

    return x1 * y1 + x2 * y2


# Initial guessed values gor Parameters
G0 = [0.1, 0.1]

popt, pcov = curve_fit(NRTL, x, qexpt, G0)
print(popt)

# compute 95% confidence intervals for A
alpha = 0.05  # 95% confidence interval

n = len(x)  # number of data points
p = len(popt)  # number of parameters

dof = max(0, n - p)  # number of degrees of freedom

tval = t.ppf(1.0 - alpha / 2.0, dof)  # student test value for the dof and confidence level

for i, p, var in zip(range(n), popt, np.diag(pcov)):
    sigma = var ** 0.5
    print('A{0}: {1} [{2}  {3}]'.format(i, p,
                                        p - sigma * tval,
                                        p + sigma * tval))

# Calcualte R2
qpred = NRTL(x, popt[0], popt[1])
resid = qexpt - qpred

SSE = np.sum((qpred - qexpt) ** 2)
qm = np.mean(qexpt)
SST = np.sum((qexpt - qm) ** 2)
R2 = 1 - SSE / SST

fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(x, resid, 'o')
ax1.set_xlabel('Liquid mole fraction x')
ax1.set_ylabel('Residual')
ax1.set_title('Residual as a function of liquid mole fraction')
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(-0.06, 0.06)

ax2.plot(x, qexpt, 'o', x, qpred, '-')
ax2.set_xlabel('Liquid mole fraction x')
ax2.set_ylabel('Excess energy')
ax2.set_title('Measured and predicted excess energy as a function of liquid mole fraction')
ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 0.4)
ax2.legend(['Measured', 'Predicted'])
plt.show()
