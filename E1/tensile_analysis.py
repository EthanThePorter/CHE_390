import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define constants
# Length of sample (mm)
gauge_length = 50.8

# Get data
tensile_data = pd.read_excel('CHE 390 E1 Calculations.xlsx', sheet_name='G15 Tensile Data', header=[0, 1])
specimen_data = pd.read_excel('CHE 390 E1 Calculations.xlsx', sheet_name='G15 Specimen Data')

# Format specimen data to have the sample label as the index
specimen_data.index = specimen_data['Sample']

# Define sample id
sample_ids = ['(1)', 'A', 'B', 'AB', 'C', 'AC', 'BC', 'ABC']

# Define lists to save values to
youngs_modulus_list = []
yield_strength_list = []
tensile_strength_list = []

for sample_id in sample_ids:

    # Get sample width and thickness (m) and area in m2
    width = specimen_data['Width (mm)'][sample_id] / 1000
    thickness = specimen_data['Thickness (mm)'][sample_id] / 1000
    cross_sectional_area = width * thickness
    # Convert specimen extension in mm to percent
    extension_percent = tensile_data[sample_id]['Extension (mm)'] / gauge_length
    # Calculate pressure (MPa) from load (N)
    load = tensile_data[sample_id]['Load (N)']
    pressure = load / cross_sectional_area / 1000000

    # Get actual length of arrays by checking for first occurance of NaN, pressure is just length for first column
    try:
        N = [np.isnan(x) for x in pressure].index(True)
    except ValueError:
        N = len(pressure)

    # Calculate Young's modulus by performing linear regression on the first 5 points and getting the slope coefficient
    x_data = np.array(extension_percent[:5]).reshape(-1, 1)
    y_data = np.array(pressure[:5])
    model = LinearRegression()
    model.fit(x_data, y_data)
    youngs_modulus = model.coef_[0]
    # Calculate Tensile strength calculating
    x1 = np.concatenate([extension_percent[int(N/2):], [0]])
    x2 = np.concatenate([[0], extension_percent[int(N/2):]])
    y1 = np.concatenate([pressure[int(N/2):], [0]])
    y2 = np.concatenate([[0], pressure[int(N/2):]])
    slope = ((y2 - y1) / (x2 - x1))[1:]
    break_index = [x > -15 for x in slope].index(False) + int(N/2)
    tensile_strength = pressure[break_index]
    # Get yield strength as the max os the pressure
    yield_strength = max(pressure)

    # Append values to lists
    yield_strength_list.append(yield_strength)
    youngs_modulus_list.append(youngs_modulus)
    tensile_strength_list.append(tensile_strength)
    # Plot first 90% of data
    plt.plot(extension_percent[:break_index] * 100, pressure[:break_index], label=sample_id)


# Create data frame and save
df = pd.DataFrame()
df["Young's Modulus (MPa)"] = youngs_modulus_list
df['Yield Strength (MPa)'] = yield_strength_list
df['Tensile Strength (MPa)'] = tensile_strength_list
df.to_excel('data.xlsx', index=False)
print(df)


plt.ylabel('Pressure (MPa)')
plt.xlabel('Percent Extension (%)')
plt.grid()
plt.legend()
plt.show()


