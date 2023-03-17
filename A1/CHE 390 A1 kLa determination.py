import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve, minimize
import pandas as pd

class BioreactorOptimization:


    def __init__(self):
        # GENERAL CONSTANTS
        # Atmospheric Pressure (Pa)
        self.P_0 = 101325
        # Density of media - Assume to be similar to water at 30C (kg/m3)
        self.p = 995.65
        # Gravitational constant
        self.g = 9.81
        # Efficiency of electric motors
        self.eff = 0.8

        # REACTOR CONSTANTS
        # Volume of Reactor (L)
        self.V = 1.75
        # Diameter of reactor (m)
        self.d_T = 0.125
        # Diameter of impeller (m)
        self.d_i = 0.0524
        # Impeller blade thickness (m)
        self.b_t = 2 / 1000
        # Cross-Sectional Area of bioreactor (m2)
        self.A = np.pi * (self.d_T / 2) ** 2
        # Height of fluid in reactor (m) - converts V from L to m3
        self.h = (self.V/1000) / self.A


    def CompressorPower(self, Q):
        """
        Function to calculate power required to operate air compressor for the given bioreactor
        :param Q: Gas flow rate in (SLPM)
        :return: Returns the power required in (W)
        """
        # Gas flow in (L/min) to (m3/s)
        Q = Q / 1000 / 60
        # Cp/Cv = Gamma
        gamma = 1.4
        # Hydrostatic pressure at bottom of vessel near sparger
        P_1 = self.p * self.g * self.h
        # Equation 5: https://learn.uwaterloo.ca/d2l/le/content/881002/viewContent/4749879/View
        a_1 = gamma / (1 - gamma) * self.P_0 * ((P_1/self.P_0) ** ((gamma-1) / gamma) - 1)
        # Compressor Power = Equation 4: https://learn.uwaterloo.ca/d2l/le/content/881002/viewContent/4749879/View
        P_compressor = a_1 * Q
        # Apply power efficiency
        P_compressor /= self.eff
        return P_compressor


    def AgitatorPower(self, Q, RPM):
        """
        Function to calculate Agitator Power
        :param Q: Gas flow rate in (SLPM)
        :param RPM: Agitator rotational speed in (RPM)
        :return: Returns gassed power required to run agitator (W)
        """
        # Gas flow in (L/min) to (m3/s)
        Q = Q / 1000 / 60
        # Get rotational speed in (rad/s)
        N = RPM * 2 * np.pi / 60
        # Power Number - Equation 3: https://learn.uwaterloo.ca/d2l/le/content/881002/viewContent/4749881/View
        N_p = 6.57 - 54.771 * (self.b_t / self.d_i)
        # Ungassed Power - Equation 4: https://learn.uwaterloo.ca/d2l/le/content/881002/viewContent/4749881/View
        P_ug = N_p * self.p * N ** 3 * self.d_i ** 5
        # Gassed Power - Equation 56: https://learn.uwaterloo.ca/d2l/le/content/881002/viewContent/4749880/View
        P_g = 1.224 * (P_ug ** 2 * N * self.d_i ** 3 / Q ** 0.56) ** 0.432
        # Apply power efficiency
        P_g /= self.eff
        return P_g


    def PowerOptimization(self):
        """
        Output results for small scale power optimization
        """
        # Initialize list of flow rates
        Q_list = np.linspace(0.5,10, 1000)
        # Initialize value for agitation speed
        RPM = 250

        # Get array of compressor power
        # Increase compressor power requirements by 20% to account for any pressure losses
        P_compressor_array = self.CompressorPower(Q_list) * 1.2
        P_agitator_array = self.AgitatorPower(Q_list, RPM)
        P_total_array = P_compressor_array + P_agitator_array


        # Output minimum total power
        minimum_total_power = min(P_total_array)
        index = list(P_total_array).index(minimum_total_power)
        Q_min = Q_list[index]
        print(f'Minimum total power required is: {round(minimum_total_power,1)}W at {round(Q_min, 1)}SLPM')

        # Plot
        plt.plot(Q_list, P_compressor_array, label='Compressor')
        plt.plot(Q_list, P_agitator_array, label='Agitator')
        plt.plot(Q_list, P_total_array, label='Total')
        plt.xlabel('GasFlo (SLMP)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid()
        plt.show()

BR = BioreactorOptimization()

kLa = [0.001952629, 0.004069727, 0.010424924, 0.014129501, 0.005163069, 0.018212969]
Q = np.array([1, 1, 2, 3, 2, 2])
RPM = np.array([150, 250, 250, 250, 150, 350])
P_g = np.array(BR.AgitatorPower(Q, RPM))

# Get Superficial gas velocities
U_s = Q / 1000 / 60 / BR.A
# Get volume of reactor in m3
V = BR.V / 1000

# Define P_g/V
Pg_V = P_g / V

def f(p):
    # Parse parameters
    a = p[0]
    b = p[2]
    c = p[4]

    # Sum
    summation = np.zeros(int(len(Q)))
    # Iterate through
    for i in range(int(len(Q))):
        summation[i] = c * (Pg_V[i])**a *U_s[i]**b - kLa[i]

    return summation

x0 = [1, 1, 1, 1, 1, 1]
p = fsolve(f, x0)

a = p[0]
b = p[1]
c = p[2]

print(p)
print('[0.47542035 1.47204663 0.580347  ]')

for i in range(len(Q)):
    print(c * (Pg_V[i]) ** a * U_s[i] ** b - kLa[i])


