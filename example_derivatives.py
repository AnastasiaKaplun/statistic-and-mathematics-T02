# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:58:43 2024

@author: rbaia
"""

# Computing derivatives



# Plot the original function
import numpy as np
import matplotlib.pyplot as plt

# Define the full energy expenditure function
def f(x):
    return -x**2 + 8*x + 20

# Generate datapoints for plotting 
x_values = np.linspace(0, 10, 500)    # Generate values of x (500 values of x between 0 and 10)
f_values = f(x_values)                # Compute the y values for the x values generated

# Plot the full function and the linear term
plt.figure(figsize=(10, 6))
plt.plot(x_values, f_values, label=r'$f(x)=-x^2 + 8x + 20$', color='r')
plt.title(r'A function $f(x)=-x^2 + 8x + 20$')
plt.xlabel('x values')
plt.ylabel('f values')
#plt.xlim([0,10])
plt.ylim([0,40])
plt.grid(True)
plt.show()


# Computing the derivative
from sympy import symbols, diff, solve

# Specify the variable x
x = symbols('x')

# Define the function
f = -x**2 + 8*x + 20

# Compute the derivative
dx_f = diff(f)

print('The derivative of the function is:', dx_f)



# How steep is the function at x=2? And at x=6?
# Use the subs() function to replace x by the value 2 in the derivative function dx_f
dx_f_2 = dx_f.subs(x, 2)
print(f'The value of the derivative at x = 2 is: {dx_f_2}.')

dx_f_6 = dx_f.subs(x, 6)
print(f'The value of the derivative at x = 2 is: {dx_f_6}.')



# What is the maximum value of the function (i.e., derivative=0)
x_values_maxima = solve(dx_f, x)
print("The derivative is zero at the following x-values:", x_values_maxima)

y_values_maxima = [f.subs(x, points) for points in x_values_maxima]
print("The y-values of these x-values are:", y_values_maxima)