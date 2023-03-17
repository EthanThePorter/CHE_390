import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, fsolve


class BioreactorOptimization:


    def __init__(
            self,
            reactor_volume,
            reactor_diameter,
            impeller_diameter,
            impeller_blade_thickness,
    ):
        """
        Class for simulating the performance of a bioreactor.
        :param reactor_volume: Fluid Volume of Reactor (L)
        :param reactor_diameter: Units: (m)
        :param impeller_diameter: Units: (m)
        :param impeller_blade_thickness: Units: (m)
        """
        # GENERAL CONSTANTS
        # Atmospheric Pressure (Pa)
        self.P_0 = 101325
        # Density of media - Assume to be similar to water at 30C (kg/m3) - https://byjus.com/physics/density-of-water/
        self.p = 995.65
        # Gravitational constant
        self.g = 9.81
        # Efficiency of electric motors
        self.eff = 0.8

        # REACTOR CONSTANTS
        # Volume of Reactor (L)
        self.V = reactor_volume
        # Diameter of reactor (m)
        self.d_T = reactor_diameter
        # Diameter of impeller (m)
        self.d_i = impeller_diameter
        # Impeller blade thickness (m)
        self.b_t = impeller_blade_thickness

        # Cross-Sectional Area of bioreactor (m2)
        self.A = np.pi * (self.d_T / 2) ** 2
        # Height of fluid in reactor (m) - converts V from L to m3
        self.h = (self.V/1000) / self.A

        # kLa correlation values
        self.a = 0.525
        self.b = 0.746
        self.c = 0.00005


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


    def PowerOptimization(
            self,
            RPM=250.0,
            flowrate_range=(0.5, 20),
            use_superficial_velocity=False
    ):
        """
        Optimize power based on fixed RPM and variable gas flowrate. Outputs plot of
        :param use_superficial_velocity: Divide Q by A to get superficial gas velocity
        :param RPM: Fixed RPM value
        :param flowrate_range: Range of values to plot
        """
        # Initialize list of flow rates
        Q_list = np.linspace(flowrate_range[0], flowrate_range[1], 500)


        # Get array of compressor power
        # Increase compressor power requirements by 20% to account for any pressure losses
        P_compressor_array = self.CompressorPower(Q_list) * 1.2
        P_agitator_array = self.AgitatorPower(Q_list, RPM)
        P_total_array = P_compressor_array + P_agitator_array


        # Output minimum total power
        minimum_total_power = min(P_total_array)
        index = list(P_total_array).index(minimum_total_power)
        Q_min = Q_list[index]

        if use_superficial_velocity:
            # Update Q to be the superficial gas velocity
            Q_list = Q_list / 1000 / 60 / self.A
            Q_min = Q_min / 1000 / 60 / self.A
            print(f'Minimum total power required is: {round(minimum_total_power, 1)}W at {round(Q_min, 6)}m/s')
            # Plot
            plt.plot(Q_list, P_compressor_array, label='Compressor')
            plt.plot(Q_list, P_agitator_array, label='Agitator')
            plt.plot(Q_list, P_total_array, label='Total')
            plt.xlabel('Superficial Gas Velocity (m/s)')
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print(f'Minimum total power required is: {round(minimum_total_power, 1)}W at {round(Q_min, 3)}SLPM')
            # Plot
            plt.plot(Q_list, P_compressor_array, label='Compressor')
            plt.plot(Q_list, P_agitator_array, label='Agitator')
            plt.plot(Q_list, P_total_array, label='Total')
            plt.xlabel('GasFlo (SLMP)')
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid()
            plt.show()

    def kLa(self, Q, RPM):
        """
        Function to calculate kLa Values for reaction
        :param Q: Flowrate in (SLPM)
        :param RPM: Agitation speed in (RPM)
        :return: kLa value in (s^-1)
        """
        P_g = self.AgitatorPower(Q, RPM)
        U = Q / 1000 / 60 / self.A
        print(P_g, U)
        return self.c * (P_g / self.V) ** self.a * U ** self.b


# Initialize simulation for small scale reactor
small_bioreactor = BioreactorOptimization(
    reactor_volume=1.75,
    reactor_diameter=0.125,
    impeller_diameter=0.0524,
    impeller_blade_thickness=2/1000,
)

# Initialize simulation for large scale reactor
large_bioreactor = BioreactorOptimization(
    reactor_volume=860,
    reactor_diameter=0.98642366,
    impeller_diameter=0.413508798,
    impeller_blade_thickness=0.015782779,
)

print(small_bioreactor.kLa(2, 250))


# # Find minimum superficial gas velocity to required to get
# Q_list = np.linspace(0.5, 100000, 1000)
# kLa_list = large_bioreactor.kLa(Q_list, 31.6801)
# plt.plot(Q_list / 1000 / 60 / large_bioreactor.A, kLa_list)
# plt.show()
#

