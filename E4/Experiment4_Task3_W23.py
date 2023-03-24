# Experiment4_Task1a_W23

# Function SteamReforming computes the equilibrium composition of multiple
# reactions


import numpy as np
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt
import pandas as pd

v = np.array([[-1, -1, 3, 1, 0],
              [0, -1, 1, -1, 1],
              [-1, -2, 4, 0, 1]])  # Stoichiometry Matrix

e = np.array([
    [1, 0, 0, 1, 1],
    [4, 2, 2, 0, 0],
    [0, 1, 0, 1, 2],
])

r, c = np.shape(v)

Gf = np.array([-50.84, -228.60, 0, -137.28, -394.38])   # Free energy of formation in J/mol.
Hf = np.array([-74.85, -241.80, 0, -110.54, -393.51])  # Enthalpy of formation in kJ/mol.

# Heat capacity in J/mol K
Cp = np.array([[34.942, -3.9957e-2, 1.9184e-4, -1.5303e-7, 3.9321e-11],  # CH4
               [33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12],  # H2O
               [25.399, 2.0178e-2, -3.8549e-5, 3.1880e-8, -8.7585e-12],  # H2
               [29.556, -6.5807e-3, 2.0130e-5, -1.2227e-8, 2.2617e-12],  # CO
               [27.437, 4.2315e-2, -1.9555e-5, 3.9968e-9, -2.9872e-13]])  # CO2

R = 8.314
P = 1.20  # Reaction pressure, atm
To = 298.15  # Reference temperature in K
nCH4f = 1  # methane in feed in moles.
Ratio = 3.5  # ratio of H2O to CH4 in feed
nH2Of = Ratio * nCH4f  # Water in feed in moles
nf = np.array([nCH4f, nH2Of, 0, 0, 0])  # Initial feed concentration in mole.
nko = 0  # Inert gas in moles


def GRT(n, T):
    y = n / np.sum(n)
    u = Gf * 1000 / To \
        - (Cp[:, 0] * np.log(T / To) + Cp[:, 1] / 2 * (T - To) + Cp[:, 2] / 6 * (T**2 - To**2) + Cp[:, 3] / 12 * (T**3 - To**3) + Cp[:, 4] / 20 * (T**4 - To**4)) \
        + (Hf * 1000 - Cp[:, 0] * To - Cp[:, 1] / 2 * To**2 - Cp[:, 2] / 3 * To**3 - Cp[:, 3] / 4 * To**4 - Cp[:, 4] / 5 * To**5) * (1/T - 1/To) \
        * T
    return np.sum(n * u / (R * T)) + np.sum(n * (np.log(P) + np.log(y)))


def constraints(n, T):
    return np.dot(e, n) - np.dot(e, nf)


fp = np.zeros(r)
x = np.zeros(c)
DeltaGo = np.zeros(r)
Ko = np.zeros(r)
DeltaHo = np.zeros(r)
RS1 = np.zeros(c)
RS2 = np.zeros(c)
sum_RS = np.zeros(r)
K = np.zeros(r)
prod = np.zeros(r)

Top = np.arange(400, 720, 20)
nT = len(Top)
xCH4 = np.zeros(nT)
xH2O = np.zeros(nT)
xH2 = np.zeros(nT)
xCO = np.zeros(nT)
xCO2 = np.zeros(nT)
Conv_CH4 = np.zeros(nT)
Yield_H2 = np.zeros(nT)
Yield_CO2 = np.zeros(nT)
Results = np.zeros((nT, 9))

for i in range(0, nT, 1):
    T = Top[i] + 273.2  # Temperature at reactor inlet in K.

    # Get bounds
    lb = [0, 0, 0, 0, 0]
    ub = [nCH4f, nH2Of, 2 * nH2Of, nCH4f, nCH4f]
    bounds = [(lb[i], ub[i]) for i in range(len(lb))]

    n0 = [0.15, 0.6, 0.2, 0.001, 0.05]  # Solution is very sensitive to initial guess!!!!!!!!!
    xsol = fmin_slsqp(GRT, n0, f_eqcons=constraints, bounds=bounds, acc=1e-6, args=(T,))
    print()
    print(xsol)
    print(GRT(xsol, T))
    print(constraints(xsol, T))

    xo = nf / (sum(nf) + nko)
    xc = xsol
    x = (xo + xc) / (1 + sum(xc))
    mT = (sum(nf) + nko) * (1 + sum(xc))  # Equation 4.6


    xCH4[i] = x[0]
    xH2O[i] = x[1]
    xH2[i] = x[2]
    xCO[i] = x[3]
    xCO2[i] = x[4]

    Conv_CH4[i] = (nf[0] - x[0] * mT) / nf[0]  # Methane conversion
    Yield_H2[i] = x[2] * mT / (nf[0] - x[0] * mT)
    Yield_CO2[i] = x[4] * mT / (nf[0] - x[0] * mT)

    Results[i, 0:] = [Top[i], xCH4[i], xH2O[i], xH2[i], xCO[i], xCO2[i], Conv_CH4[i], Yield_H2[i], Yield_CO2[i]]



# Plotting
Results = np.round(Results, 3)
index = np.arange(1, nT + 1)
pd.set_option("display.max_rows", None, "display.max_columns", None)  # option to display all rows and columns
columns = ['Temp', 'xCH4', 'xH2O', 'xH2', 'xCO', 'xCO2', 'Conv_CH4', 'Yield_H2', 'Yield_CO2']
Results1 = pd.DataFrame(Results, index, columns)
print(Results1)
Results1.to_excel('ChE390_Lab4_Task1.xlsx')

fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(Top, Conv_CH4, 'ro-')
ax1.set_xlabel('Temperature (C))')
ax1.set_ylabel('CH4 conversion')
ax1.set_ylim(0, 1)
ax1_y2 = ax1.twinx()
ax1_y2.plot(Top, Yield_H2, 'd-', Top, Yield_CO2, '*-')
ax1_y2.set_xlabel('Temperature (C))')
ax1_y2.set_ylabel('Yield of H2 or CO2')
ax1_y2.set_ylim(0, 4.5)

ax2.plot(Top, Conv_CH4, 'o-', Top, xCH4, 'd-', Top, xH2O, 's-', Top, xH2, '+-', Top, xCO, 'x-', Top, xCO2, '*-')
ax2.set_xlabel('Temperature (C))')
ax2.set_ylabel('Mole fraction or methane conversion')
ax2.legend(('Conv_CH4', 'xCH4', 'xH2O', 'xH2', 'xCO', 'xCO2'), loc='best')
plt.show()