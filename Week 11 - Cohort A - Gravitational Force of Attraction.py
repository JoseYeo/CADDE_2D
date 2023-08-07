# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 03:27:45 2023

@author: nagarajan
"""

import numpy as np
import matplotlib.pyplot as plt

# Defined constant(s)
G = 6.67384e-11  # SI units

# Define values to sample from
mean_m_1 = 40.e4
sigma_m_1 = 0.05e4
mean_m_2 = 30.e4
sigma_m_2 = 0.1e4
mean_r = 3.2
sigma_r = 0.01

# Compute mean and uncertainty in the force using standard error propagation (ANALYTICAL):
    
mean_f = G * mean_m_1 * mean_m_2 / mean_r ** 2
sigma_f = mean_f * np.sqrt((sigma_m_1 / mean_m_2) ** 2
                           + (sigma_m_2 / mean_m_2) ** 2
                           + 4. * (sigma_r / mean_r) ** 2)
print(mean_f, sigma_f)

# Now compute this using Monte-Carlo error propagation. We sample N initial values that are drawn from the initial distributions:
    
N = 1000
m_1 = np.random.normal(mean_m_1, sigma_m_1, N)
m_2 = np.random.normal(mean_m_2, sigma_m_2, N)
r = np.random.normal(mean_r, sigma_r, N)

# For each sample, we can compute the force value:
F = G * m_1 * m_2 / r ** 2

# We can print for these the mean and standard deviation:
print(np.mean(F), np.std(F))

# This is similar to the values found above, but in fact we have the full distribution of values.
# Which we can plot a histogram for, along with a curve showing the Gaussian function for the result found from standard error propagation (analytically):

# Define range of output values for plotting
xmin = 0.75
xmax = 0.82

# Define Gaussian function

def gaussian(x, mu, sigma):
    norm = 1. / (sigma * np.sqrt(2. * np.pi))
    return norm * np.exp(-(x - mu) ** 2. / (2. * sigma ** 2))

x = np.linspace(xmin, xmax, 1000)
y = gaussian(x, mean_f, sigma_f)

plt.hist(F, bins=50, range=[xmin, xmax], density=True)
plt.plot(x, y, color='red', lw=3)
plt.xlabel("Force (N)")
plt.ylabel("Relative Probability")
plt.xlim(xmin, xmax)

# The distribution of sampled points is now significantly non-Gaussian, which is normal 
# because the uncertainties are large, and standard error propagation only works for 
# small errors. The conclusion is that Monte-Carlo propagation is easier to code 
# (because one doesn't need to remember all the propagation equations) and is also more 
# correct because it can take into account non-Gaussian distributions (of input or output).
    
    
    