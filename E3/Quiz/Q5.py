import numpy as np
from scipy.optimize import fsolve

# Define equation
N = lambda f, Re: 4 * np.log10(Re * np.sqrt(f)) - 0.4 - 1 / np.sqrt(f)
# Define initial guess
f0 = 0.005
# Use fsolve to get root
f = fsolve(N, f0, args=(3000,))


