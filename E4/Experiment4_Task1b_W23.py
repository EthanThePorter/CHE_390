# Experiment4_Task1a_W23

# Function SteamReforming computes the equilibrium composition of multiple
# reactions


import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd

v = np.array([[-1, -1, 3, 1, 0],
              [0, -1, 1, -1, 1],
              [-1, -2, 4, 0, 1]])  # Stoichiometry Matrix

r, c = np.shape(v)

Gf = np.array([-50.84, -228.60, 0, -137.28, -394.38])  # Free energy of formation in kJ/mol.
Hf = np.array([-74.85, -241.80, 0, -110.54, -393.51])  # Enthalpy of formation in kJ/mol.

# Heat capacity in J/mol K

Cp = np.array([[34.942, -3.9957e-2, 1.9184e-4, -1.5303e-7, 3.9321e-11],  # CH4
               [33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12],  # H2O
               [25.399, 2.0178e-2, -3.8549e-5, 3.1880e-8, -8.7585e-12],  # H2
               [29.556, -6.5807e-3, 2.0130e-5, -1.2227e-8, 2.2617e-12],  # CO
               [27.437, 4.2315e-2, -1.9555e-5, 3.9968e-9, -2.9872e-13]])  # CO2

R = 8.314
P = 1.55  # Reaction pressure, atm
To = 298.15  # Reference temperature in K
nCH4f = 1  # methane in feed in moles.
Ratio = 3.5  # ratio of H2O to CH4 in feed
nH2Of = Ratio * nCH4f  # Water in feed in moles
nf = np.array([nCH4f, nH2Of, 0, 0, 0])  # Initial feed concentration in mole.
nko = 0  # Inert gas in moles

# Equilibrium Equation function

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


def EquiEquations(beta):
    # Equilibrium Equations
    for i in range(r):
        DeltaGo[i] = np.dot(v[i], Gf)
        Ko[i] = np.exp(-DeltaGo[i] * 1000 / R / To)

        DeltaHo[i] = np.dot(v[i], Hf)

        # RS1 and Rs2 are the first and second terms of the right-hand side of Equation 4.18
        RS1 = Cp[:, 0] * np.log(T / To) + Cp[:, 1] / 2 * (T - To) + Cp[:, 2] / 6 * (T ** 2 - To ** 2) + Cp[:,
                                                                                                        3] / 12 * (
                          T ** 3 - To ** 3) + Cp[:, 4] / 20 * (T ** 4 - To ** 4)
        RS2 = (Hf * 1000 - (
                    Cp[:, 0] * To + Cp[:, 1] / 2 * To ** 2 + Cp[:, 2] / 3 * To ** 3 + Cp[:, 3] / 4 * To ** 4 + Cp[:,
                                                                                                               4] / 5 * To ** 5)) * (
                          1 / T - 1 / To)

        sum_RS[i] = np.dot(v[i], RS1) - np.dot(v[i], RS2)
        K[i] = Ko[i] * np.exp(sum_RS[i] / R)
        xo = nf / (sum(nf) + nko)  # Initial mole fraction vector
        xc = np.dot(beta, v)
        x = (xo + xc) / (1 + sum(xc))  # Equation 4.6
        prod = 1
        for j in range(c):
            prod = x[j] ** (v[i, j]) * prod

        fp[i] = K[i] - P ** (sum(v[i, :])) * prod  # Equation 4.9

    ft = fp ** 2
    return ft


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
    lb = [-1.0, -1.0, -1.0]  # It is a reversible reaction, so the extent of reactions can be -1

    ub = [1.0, 1.0, 1.0]  # Upper bound for the extents of the reactions

    n0 = [0.1, 0.02, 0.02]  # Solution is very sensitive to initial guess!!!!!!!!!
    xsol = least_squares(EquiEquations, n0, bounds=(lb, ub), method='trf', ftol=1.0e-6, loss='soft_l1', f_scale=0.5,
                         max_nfev=8000)

    beta = xsol.x[0:c]  # solution for the extents of the reactions
    xo = nf / (sum(nf) + nko)
    xc = np.dot(beta, v)
    x = (xo + xc) / (1 + sum(xc))
    mT = (sum(nf) + nko) * (1 + sum(xc))  # Equation 4.6

    print(xc)
    print(beta)
    print(mT)
    print()

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
