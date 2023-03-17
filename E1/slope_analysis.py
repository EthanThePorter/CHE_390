import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define constants
# Length of sample (mm)
gauge_length = 50.4

# Get data
tensile_data = pd.read_excel('CHE 390 E1 Data.xlsx', sheet_name='Tensile Data', header=[0, 1])
specimen_data = pd.read_excel('CHE 390 E1 Data.xlsx', sheet_name='Specimen Data')

# Format specimen data to have the sample label as the index
specimen_data.index = specimen_data['Sample']

# Define sample id
sample_id = '(1)'

# Define lists to save values to
youngs_modulus_list = []
yield_strength_list = []
tensile_strength_list = []


# Get sample width and thickness (m) and area in m2
width = specimen_data['Width (mm)'][sample_id] / 1000
thickness = specimen_data['Thickness (mm)'][sample_id] / 1000
cross_sectional_area = width * thickness
# Convert specimen extension in mm to percent
extension_percent = tensile_data[sample_id]['Extension (mm)'] / gauge_length
# Calculate pressure (MPa) from load (N)
load = tensile_data[sample_id]['Load (N)']
pressure = load / cross_sectional_area / 1000000

# Calculate Young's modulus by performing linear regression on the first 5 points and getting the slope coefficient
x_data = np.array(extension_percent[:5]).reshape(-1, 1)
y_data = np.array(pressure[:5])
model = LinearRegression()
model.fit(x_data, y_data)
youngs_modulus = model.coef_[0]

print(f'{sample_id}: {round(youngs_modulus, 2)} MPa')

x1 = np.concatenate([extension_percent, [0]])
x2 = np.concatenate([[0], extension_percent])
y1 = np.concatenate([pressure, [0]])
y2 = np.concatenate([[0], pressure])
slope = ((y2 - y1) / (x2 - x1))[1:]
break_index = [x > -50 for x in slope].index(False)

print(break_index)

plt.axvline(extension_percent[break_index] * 100, color = 'red')

# Plot slope
plt.plot(extension_percent * 100, slope, label='Slope')


# Plot data
plt.plot(extension_percent * 100, pressure, label=sample_id)


plt.ylabel('Pressure (MPa)')
plt.xlabel('Percent Extension (%)')
plt.grid()
plt.legend()
plt.show()


