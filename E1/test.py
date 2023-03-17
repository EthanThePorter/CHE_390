import numpy as np
import pandas as pd

data = pd.read_excel('CHE 390 E1.xlsx', sheet_name='G16 Specimen Data')

print(data)

width = data['Width (mm)']
thickness = data['Thickness (mm)']

x = width * thickness

print(x)

print([1, 2, 3, 4][0:1])