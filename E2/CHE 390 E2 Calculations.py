import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FluidData:

    def __init__(self):

        # CONSTANTS
        PD1_zeroshift = 10
        PD2_zeroshift = 22
        self.pipe_labels = ['Glass DIN 15', 'AISI 304 DIN 15', 'Venturi', 'U-Tube']

        # Get data
        self.pressure = pd.read_excel('CHE 390 E2 Data.xlsx', sheet_name='Pressure Data')
        self.flowrate = pd.read_excel('CHE 390 E2 Data.xlsx', sheet_name='Flowrate Data')
        self.temperature = pd.read_excel('CHE 390 E2 Data.xlsx', sheet_name='Temperature Data')

        # Convert flowrate from power percent to m3/s
        for pipe in self.pipe_labels:
            self.flowrate[pipe] = self.flowrate[pipe] / 100 * 10 / 3600
        # Apply zero shift to pressure, then convert from mmH2O to Pa
        for pipe in self.pipe_labels:
            for i, value in enumerate(self.pressure[pipe]):
                # Apply Zero Shift and Conversion
                if value < 1000:
                    self.pressure.at[i, pipe] = (value - PD1_zeroshift) * 9.80665
                else:
                    self.pressure.at[i, pipe] = (value - PD2_zeroshift) * 9.80665

        # Get viscosity data
        self.viscosity = pd.DataFrame()
        for pipe in self.pipe_labels:
            self.viscosity[pipe] = self.viscosity_function(self.temperature[pipe])

        # Get density data
        self.density = pd.DataFrame()
        for pipe in self.pipe_labels:
            self.density[pipe] = self.density_function(self.temperature[pipe])

        # Pipe diameters
        diameters = [15/1000, 18/1000, 24/1000, 31/1000]
        self.diameters = pd.DataFrame()
        for i, pipe in enumerate(self.pipe_labels):
            self.diameters[pipe] = diameters[i]



    @staticmethod
    def viscosity_function(T):
        """
        https://www.fxsolver.com/browse/formulas/dynamic+viscosity+of+water+%28as+a+function+of+temperature+temperature%29
        :param T: Temperature in Celsius
        :return: Viscosity of water in Pa.s
        """
        return 2.414E-5 * 10 ** (247.8 / (T - 140 + 273.15))

    @staticmethod
    def density_function(T):
        """
        https://www.omnicalculator.com/physics/water-density
        :param T: Temperature in Celsius
        :return: Density of water in kg/m3
        """
        return 999.83311 + 0.0752 * T - 0.0089 * T**2 + 7.36413E-5 * T**3 - 4.74639E-7 * T**4 + 1.34888E-9* T**5


# Initialize Object for Fluid Data
data = FluidData()

