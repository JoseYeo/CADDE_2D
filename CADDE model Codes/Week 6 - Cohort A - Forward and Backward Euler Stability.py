# -*- coding: utf-8 -*-
"""
Created on Fri May 27 03:01:12 2022
@author: Nagarajan Raghavan
Course: 30.100 Part II

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# TRUE SOLUTION
# function that returns dy/dt
def model(y,tt):
 #   dydt = y*tt + 1.0
 #   dydt = 1 + y + tt**5
 dydt = -10*y
 return dydt

# initial condition
y0 = 1

# time points
tt = np.linspace(0,1)

# solve ODE
y = odeint(model,y0,tt)

# plot results
figure = plt.figure(facecolor="white")

plt.plot(tt,y, label="True Solution", linestyle='--', marker='o')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

# NUMERICAL ANALYSIS

def dy_dt(t,y_t):
#        return 1 + y_t + t**5
         return -10*y_t
#t*y_t + 1

t_range = (0,1)

initial_value_pair = (t_range[0],1)

euler_stepsize = 0.05
# backeuler_stepsize = 0.25

#Draw Forward Euler solution

h=euler_stepsize
euler_t_range = np.arange(t_range[0],t_range[1]+h/1000., h)
euler_y_vals = np.array([initial_value_pair[1]])

for k in euler_t_range:
        if k == t_range[0]:
                continue
        else:
                next_entry = euler_y_vals[-1] + h*dy_dt(k-h,euler_y_vals[-1])
                euler_y_vals = np.append (euler_y_vals, next_entry)

plt.plot(euler_t_range,euler_y_vals, linewidth=2.5, label="Forward Euler Approx")
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

#Draw Backward Euler solution

# h=backeuler_stepsize
# backeuler_t_range = np.arange(t_range[0],t_range[1]+h/1000., h)
# backeuler_y_vals = np.array([initial_value_pair[1]])

# for k in backeuler_t_range:
#         if k == t_range[0]:
#                 continue
#         else:
#                 next_entry = backeuler_y_vals[-1] + h*dy_dt(k,euler_y_vals[-1]+ h*dy_dt(k-h,euler_y_vals[-1]))
#                 backeuler_y_vals = np.append (backeuler_y_vals, next_entry)

# plt.plot(backeuler_t_range,backeuler_y_vals, linewidth=2.5, label="Backward Euler Approx")
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")


plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")
plt.show()