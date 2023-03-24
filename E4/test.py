import numpy as np

# Heat capacity in J/mol K
Cp = np.array([[34.942, -3.9957e-2, 1.9184e-4, -1.5303e-7, 3.9321e-11],  # CH4
               [33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12],  # H2O
               [25.399, 2.0178e-2, -3.8549e-5, 3.1880e-8, -8.7585e-12],  # H2
               [29.556, -6.5807e-3, 2.0130e-5, -1.2227e-8, 2.2617e-12],  # CO
               [27.437, 4.2315e-2, -1.9555e-5, 3.9968e-9, -2.9872e-13]])  # CO2


print(Cp[:, 0])