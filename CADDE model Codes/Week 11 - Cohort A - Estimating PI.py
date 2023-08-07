# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 02:52:11 2023

@author: nagarajan
"""
#Define the input
# Let us estimate the value of PI using Monte Carlo Simulations

import numpy as np

n_simulations = 1000
n_points_circle = 0
n_points_square = 0

# Generate points by sampling from the probability distribution
for _ in range(n_simulations):    
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    
    # Check whether each point falls within the circle (red) or not
    
    dist_from_origin = x**2 + y**2
    if dist_from_origin <= 1:
        n_points_circle += 1
    n_points_square += 1
    
# Use the pi formula derived above to simulate pi

pi = 4 * n_points_circle / n_points_square
print(pi)