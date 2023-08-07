# -*- coding: utf-8 -*-
"""
Created on Fri May 27 03:01:12 2022
@author: Nagarajan Raghavan
Course: 30.100 CaDDE

"""

import matplotlib.pyplot as plt
import numpy as np

# Independent Variable: t
# Dependent Variable: y

# Define Your Functional Form of dy/dt

def dy_dt(t,y_t):
       return t**3 - 5*t - 2

# Define Your Functional Form of the Solution y(t)
# Here, let us first assume that we can solve the ODE, so we know the exact solution to it

def y(t):
         return t**4/4 - 5/2*t**2 - 2*t + 20

# Define the time range (range of independent variable) for the analysis

t_range = (2,4.5)

# Define the starting point of the analysis: (t0, y0)

initial_value_pair = (t_range[0],10)

# Make figure background white in color

figure = plt.figure(facecolor="white")

# Let us first plot out the exact solution (because we know it!)

# Breakdown the t-axis into fine points and compute the TRUE VALUE of y at each point

exact_t_range = np.arange(t_range[0],t_range[1],(t_range[1]-t_range[0])/10)
exact_y_vals = y(exact_t_range)

# Plot out the exact solution
plt.plot(exact_t_range,exact_y_vals, linestyle='--', marker='o', markersize=10, linewidth=2.5, label="Exact Solution")  

# NUMERICAL ANALYSIS BEGINS HERE!
# Let us explore the FORWARD EULER Approach

# Define the step size of analysis, this will DIRECTLY IMPACT on the ACCURACY of NUMERICAL METHOD
euler_stepsize = 0.5
backeuler_stepsize = 0.5
midpoint_stepsize = 0.5
heun_stepsize = 0.5

h=euler_stepsize

# Breakdown the t-axis into fine points (you can ideally use the already defined time range earlier)
euler_t_range = np.arange(t_range[0],t_range[1]+h/1000., h)

# Initialize the first value of predicted y at (t0, y0)
euler_y_vals = np.array([initial_value_pair[1]])

# Initialize the first value of predicted y at (t0, y0)
euler_y_vals_2nd_Order = np.array([initial_value_pair[1]])

# Let us start from left end and go to the right end of the time range (independent variable)
for k in euler_t_range:
        if k == t_range[0]:
                # Do not do anything at first time step    
                continue
        else:
                 # EULER Formula here - compute new y from previous y (latest known value of y at previous time point (k-h))
                 # Array with [-1] refers to the last value currently in the array
                 # (k-h) refers to the immediate previous point of analysis
                 
                 # FIRST ORDER
                 next_entry = euler_y_vals[-1] + h*dy_dt(k-h,euler_y_vals[-1])
                 
                 # Add in newly computed estimate of y(t=k) at time stamp (k) to the array
                 euler_y_vals = np.append (euler_y_vals, next_entry)
                 
                 # SECOND ORDER
                 next_entry = euler_y_vals[-1] + h*dy_dt(k-h,euler_y_vals[-1]) + h*h/2*(3*(k-h)**2-5)
                 
                 # Add in newly computed estimate of y(t=k) at time stamp (k) to the array
                 euler_y_vals_2nd_Order = np.append (euler_y_vals_2nd_Order, next_entry)

# Plot out the numerical solution using EULER Method
plt.plot(euler_t_range,euler_y_vals, linewidth=4, linestyle = 'dotted', label="Euler Approx 1st Order")
plt.plot(euler_t_range,euler_y_vals_2nd_Order, linewidth=4, linestyle = 'dashdot', label="Euler Approx 2nd Order")

#Draw Backward Euler solution

h=backeuler_stepsize
backeuler_t_range = np.arange(t_range[0],t_range[1]+h/1000., h)
backeuler_y_vals = np.array([initial_value_pair[1]])

for k in backeuler_t_range:
        if k == t_range[0]:
                continue
        else:
                next_entry = backeuler_y_vals[-1] + h*dy_dt(k,euler_y_vals[-1]+ h*dy_dt(k-h,euler_y_vals[-1]))
                backeuler_y_vals = np.append (backeuler_y_vals, next_entry)

plt.plot(backeuler_t_range,backeuler_y_vals, marker='*',linewidth=2.5, label="Backward Euler Approx")
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

#Draw Heun's Predictor-Corrector solution

h=heun_stepsize
heun_t_range=np.arange(t_range[0],t_range[1]+h/1000.,h)
heun_y_vals=np.array([initial_value_pair[1]])

for k in heun_t_range:
        if k == t_range[0]:
                continue
        else:
                next_entry = heun_y_vals[-1] + h/2*((dy_dt(k-h,heun_y_vals[-1])+(dy_dt(k,heun_y_vals[-1] + h*dy_dt(k-h,heun_y_vals[-1])))))
                heun_y_vals = np.append(heun_y_vals, next_entry)

plt.plot(heun_t_range, heun_y_vals, marker='*', linewidth=2.5, label="Heun Method")
#plt.xlim([0, 1])
#plt.ylim([0, 6])
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

# Draw Midpoint solution

h=midpoint_stepsize
midpoint_t_range=np.arange(t_range[0],t_range[1]+h/1000.,h)
midpoint_y_vals=np.array([initial_value_pair[1]])

for k in midpoint_t_range:
        if k == t_range[0]:
                continue
        else:
                next_entry = midpoint_y_vals[-1] + h*dy_dt(k-h+h/2,midpoint_y_vals[-1] + h/2*dy_dt(k-h,midpoint_y_vals[-1]))
                midpoint_y_vals = np.append(midpoint_y_vals, next_entry)

plt.plot(midpoint_t_range, midpoint_y_vals, marker='*', linewidth=2.5, label="Midpoint Method")
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

plt.xlabel("Time")
plt.ylabel("Function Value")
plt.title("Comparison of First and Second Order Taylor Series Approx")
plt.legend(loc="upper left")
plt.show()