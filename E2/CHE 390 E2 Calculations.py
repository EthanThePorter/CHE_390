import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONSTANTS
PD1_zeroshift = 10
PD2_zeroshift = 22
pipe_labels = ['Glass DIN 15', 'AISI 304 DIN 15', 'Venturi', 'U-Tube']

# Get data
pressure = pd.read_excel('CHE 390 E2 Data.xlsx', sheet_name='Pressure Data')
flowrate = pd.read_excel('CHE 390 E2 Data.xlsx', sheet_name='Flowrate Data')
temperature = pd.read_excel('CHE 390 E2 Data.xlsx', sheet_name='Temperature Data')

print(pressure)

# Convert flowrate from power percent to m3/s
for pipe in pipe_labels:
    flowrate[pipe] = flowrate[pipe] / 100 * 10 / 3600
# Apply zero shift to pressure, then convert from mmH2O to Pa
for pipe in pipe_labels:
    for i, value in enumerate(pressure[pipe]):
        # Apply Zero Shift and Conversion
        if value < 1000:
            pressure.at[i, pipe] = (value - PD1_zeroshift) * 9.80665
        else:
            pressure.at[i, pipe] = (value - PD2_zeroshift) * 9.80665




print(pressure)


