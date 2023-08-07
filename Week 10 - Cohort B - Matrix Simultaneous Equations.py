# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:55:35 2022

@author: smith
"""
import numpy as np

a = np.array([[-39, 24, 0], 
              [16, -39, 24], 
              [0, 16, -39]])

b = np.array([-3.1375, 0.25, -18.6375])

from scipy import linalg

x = linalg.solve(a, b)
print (x)