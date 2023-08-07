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
       return 4*np.exp(0.8*t) - 0.5*y_t

# Define Your Functional Form of the Solution y(t)
# Here, let us first assume that we can solve the ODE, so we know the exact solution to it

def y(t):
         return 4/(1.3)*(np.exp(0.8*t)-np.exp(-0.5*t)) + 2*np.exp(-0.5*t)

# Define the time range (range of independent variable) for the analysis

t_range = (0,4)

# Define the starting point of the analysis: (t0, y0)

initial_value_pair = (t_range[0],2)

# Make figure background white in color

figure = plt.figure(facecolor="white")

# Let us first plot out the exact solution (because we know it!)

# Breakdown the t-axis into fine points and compute the TRUE VALUE of y at each point

#exact_t_range = np.arange(t_range[0],t_range[1],(t_range[1]-t_range[0])/5)
exact_t_range = np.array([0,1,2,3,4])
exact_y_vals = y(exact_t_range)

# Plot out the exact solution
plt.plot(exact_t_range,exact_y_vals, linestyle='--', marker='o', markersize=10, linewidth=2.5, label="Exact Solution")  

# NUMERICAL ANALYSIS BEGINS HERE!
# Let us explore the FORWARD EULER Approach

# Define the step size of analysis, this will DIRECTLY IMPACT on the ACCURACY of NUMERICAL METHOD
euler_stepsize = 1
heun_stepsize = 1
runge_kutta_3_stepsize = 1
runge_kutta_4_stepsize = 1

#Draw First Order Runge-Kutta Solution (Forward Euler)

h=euler_stepsize

# Breakdown the t-axis into fine points (you can ideally use the already defined time range earlier)
euler_t_range = np.arange(t_range[0],t_range[1]+h/1000., h)

# Initialize the first value of predicted y at (t0, y0)
euler_y_vals = np.array([initial_value_pair[1]])


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
                 
# Plot out the numerical solution using EULER Method
plt.plot(euler_t_range,euler_y_vals, linewidth=4, linestyle = 'dotted', label="RK-1 Runge Kutta (Forward Euler)")

#Draw Second Order Runge-Kutta solution (Heun's Method)

h=heun_stepsize
heun_t_range=np.arange(t_range[0],t_range[1]+h/1000.,h)
heun_y_vals=np.array([initial_value_pair[1]])

for k in heun_t_range:
        if k == t_range[0]:
                continue
        else:
                next_entry = heun_y_vals[-1] + h/2*((dy_dt(k-h,heun_y_vals[-1])+(dy_dt(k,heun_y_vals[-1] + h*dy_dt(k-h,heun_y_vals[-1])))))
                heun_y_vals = np.append(heun_y_vals, next_entry)

plt.plot(heun_t_range, heun_y_vals, marker='*', linewidth=2.5, label="RK-2 Runge Kutta (Heun)")
#plt.xlim([0, 1])
#plt.ylim([0, 6])
plt.legend(loc="upper left")

#Draw Third Order Runge-Kutta solution

h=runge_kutta_3_stepsize
rk_t_range_3 = np.arange(t_range[0],t_range[1]+h/1000.,h)
rk_y_vals_3 = np.array([initial_value_pair[1]])

for k in rk_t_range_3:
        if k==t_range[0]:
                continue
        else:
                k_1 = dy_dt(k-h,rk_y_vals_3[-1])
                k_2 = dy_dt(k-h + h/2, rk_y_vals_3[-1] + h/2*k_1)
                k_3 = dy_dt(k-h + h, rk_y_vals_3[-1] - h*k_1 + 2*k_2*h)
                next_entry = rk_y_vals_3[-1] + 1./6*h*(k_1 + 4*k_2 + k_3)
                rk_y_vals_3 = np.append(rk_y_vals_3, next_entry)

plt.plot(rk_t_range_3, rk_y_vals_3, linewidth=2.5, label="RK-3 Runge Kutta")


#Draw Fourth Order Runge-Kutta solution

h=runge_kutta_4_stepsize
rk_t_range_4 = np.arange(t_range[0],t_range[1]+h/1000.,h)
rk_y_vals_4 = np.array([initial_value_pair[1]])

for k in rk_t_range_4:
        if k==t_range[0]:
                continue
        else:
                k_1 = dy_dt(k-h,rk_y_vals_4[-1])
                k_2 = dy_dt(k-h + h/2, rk_y_vals_4[-1] + h/2*k_1)
                k_3 = dy_dt(k-h + h/2, rk_y_vals_4[-1] + h/2*k_2)
                k_4 = dy_dt(k-h + h, rk_y_vals_4[-1] + h*k_3)
                next_entry = rk_y_vals_4[-1] + 1./6*h*(k_1 + 2*k_2 + 2*k_3 + k_4)
                rk_y_vals_4 = np.append(rk_y_vals_4, next_entry)

plt.plot(rk_t_range_4, rk_y_vals_4, linewidth=2.5, label="RK-4 Runge Kutta")

plt.xlabel("Time")
plt.ylabel("Function Value")
plt.title("Comparison of Different Orders of Runge Kutta Methods")
plt.legend(loc="upper left")
plt.show()

#Error in prediction between 1st Order RK and True Solution
mape_RK1 = np.mean(np.abs((exact_y_vals - euler_y_vals) / exact_y_vals))*100

#Error in prediction between 2nd Order RK and True Solution
mape_RK2 = np.mean(np.abs((exact_y_vals - heun_y_vals) / exact_y_vals))*100

#Error in prediction between 3rd Order RK and True Solution
mape_RK3 = np.mean(np.abs((exact_y_vals - rk_y_vals_3) / exact_y_vals))*100

#Error in prediction between 4th Order RK and True Solution
mape_RK4 = np.mean(np.abs((exact_y_vals - rk_y_vals_4) / exact_y_vals))*100