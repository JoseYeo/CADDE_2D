# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:14:24 2023

@author: nagarajan
"""
# import libraries for numerical functions and plotting
import numpy as np
import matplotlib.pyplot as plt

# plot exact solution first
time = np.linspace(0, 3)
y_exact = (-6*np.exp(-3*time) + 7*np.exp(-2*time) + np.sin(time) - np.cos(time))
plt.plot(time, y_exact, label='Exact')

omega = 1

#Step Size
h = 0.1

time = np.arange(0, 3.001, h)

f = lambda t,z1,z2: 10*np.sin(omega*t) - 5*z2 - 6*z1

z1 = np.zeros_like(time)
z2 = np.zeros_like(time)

# initial conditions
z1[0] = 0
z2[0] = 5

# Forward Euler iterations
for idx, t in enumerate(time[:-1]):
    z1[idx+1] = z1[idx] + h*z2[idx]
    z2[idx+1] = z2[idx] + h*f(t, z1[idx], z2[idx])

plt.plot(time, z1, 'o--', label='Forward Euler solution')
plt.xlabel('time')
plt.ylabel('displacement')
plt.legend()
plt.grid(True)
plt.show()