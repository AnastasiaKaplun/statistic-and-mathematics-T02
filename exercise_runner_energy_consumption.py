# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 15:52:10 2024

@author: rbaia
"""

# Exercise runner energy consumption
# The script below provides an example of plotting a function
# Modify this code to plot the function

import matplotlib.pyplot as plt
import numpy as np

# Define the full energy expenditure function
def C(s):
    return 5*s + 0.2*s**3

# Generate datapoints for plotting
s_values = np.linspace(0, 10, 500)
C_values = C(s_values)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(s_values, C_values, label='C(s) = 5s + 0.2s^3')
plt.title('Runner Energy Consumption')
plt.xlabel('Speed (m/s)')
plt.ylabel('Energy Expenditure (calories)')
plt.xlim([0, 10])
plt.ylim([0, 250])
plt.grid(True)
plt.legend(loc='lower right')
plt.show()

# Energy expenditure at 5 m/s
print("Energy expenditure at 5 m/s:", C(5))
