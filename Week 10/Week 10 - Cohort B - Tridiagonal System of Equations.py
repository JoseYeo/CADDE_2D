# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:24:36 2022

@author: smith
"""

import numpy as np
N = 3
diagonals = np.zeros((3, N))   # 3 diagonals
diagonals[0,:] = np.linspace(-1, -N, N)
diagonals[1,:] = -39
diagonals[2,:] = np.linspace(1, N, N)

import scipy.sparse
A = scipy.sparse.spdiags(diagonals, [-1,0,1], N, N, format='csc')
A.toarray()    # look at corresponding dense matrix
[[-39,  24,  0],
 [16,  -39,  24],
 [0 ,   16, -39]]

b = [-3.1375, 0.25, -18.6375]              # sparse matrix vector product

import scipy.sparse.linalg
x = scipy.sparse.linalg.spsolve(A, b)
print (x)

A_d = A.toarray()            # corresponding dense matrix
b = np.dot(A_d, x)           # standard matrix vector product
x = np.linalg.solve(A_d, b)  # standard Ax=b algorithm
print (x)