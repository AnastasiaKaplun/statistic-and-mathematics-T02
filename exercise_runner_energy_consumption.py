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
# Exercise runner energy consumption (2)
# Compute the derivative to find when energy consumption starts to accelerate

from sympy import symbols, diff, solve

s = symbols('s')

# Define the energy expenditure function
C_function = 5*s + 0.2*s**3

# Compute the derivative
dC = diff(C_function, s)
print("The derivative of C(s) is:", dC)

# Find when the quadratic term in the derivative equals the constant term
acceleration_speed = solve(0.6*s**2 - 5, s)

# Keep the positive speed
positive_speed = [value for value in acceleration_speed if value > 0][0]

print("Energy expenditure starts to increase rapidly at approximately:",
      float(positive_speed), "m/s")
      