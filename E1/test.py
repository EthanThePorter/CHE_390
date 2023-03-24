import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

plt.plot(x, y, label='Run 1')
plt.ylabel('ln(C*/C)')
plt.xlabel('t (s)')
plt.grid()
plt.legend()
plt.show()