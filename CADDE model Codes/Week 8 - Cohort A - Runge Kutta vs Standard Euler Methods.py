# -*- coding: utf-8 -*-
"""
Created on June 22, 2023
@author: Nagarajan Raghavan
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# TRUE SOLUTION
# function that returns dy/dt
def model(y,tt):
    dydt = y*tt + 1.0
    return dydt

# initial condition
y0 = 1

# time points
tt = np.linspace(0,1)

# solve ODE
y = odeint(model,y0,tt)

# plot results
figure = plt.figure(facecolor="white")

plt.plot(tt,y, label="True Solution", linestyle='--', marker='o', markersize=5)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

# NUMERICAL ANALYSIS

def dy_dt(t,y_t):
        return t*y_t + 1
        
t_range = (0,1.5)

initial_value_pair = (t_range[0],1)

euler_stepsize = 0.25
backeuler_stepsize = 0.25
midpoint_stepsize = 0.25
heun_stepsize = 0.25
runge_kutta_stepsize = 0.25

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

plt.plot(euler_t_range,euler_y_vals, marker='*', linewidth=2.5, label="Forward Euler Approx")
plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")

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
plt.xlim([0, 1])
plt.ylim([0, 6])
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

#Draw Runge-Kutta solution

h=runge_kutta_stepsize
rk_t_range = np.arange(t_range[0],t_range[1]+h/1000.,h)
rk_y_vals = np.array([initial_value_pair[1]])

for k in rk_t_range:
        if k==t_range[0]:
                continue
        else:
                k_1 = dy_dt(k-h,rk_y_vals[-1])
                k_2 = dy_dt(k-h + h/2, rk_y_vals[-1] + h/2*k_1)
                k_3 = dy_dt(k-h + h/2, rk_y_vals[-1] + h/2*k_2)
                k_4 = dy_dt(k-h + h, rk_y_vals[-1] + h*k_3)
                next_entry = rk_y_vals[-1] + 1./6*h*(k_1 + 2*k_2 + 2*k_3 + k_4)
                rk_y_vals = np.append(rk_y_vals, next_entry)

plt.plot(rk_t_range, rk_y_vals, linewidth=2.5, label="Runge Kutta Method")

plt.xlabel('time')
plt.ylabel('y(t)')
plt.title("Comparison of True and Approx Solution to the ODE")
plt.legend(loc="upper left")
plt.show()