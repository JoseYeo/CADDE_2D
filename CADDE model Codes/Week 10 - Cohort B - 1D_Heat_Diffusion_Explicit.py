# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 04:24:12 2022

@author: smith
"""

#! /usr/bin/env python3
#
def fd1d_heat_explicit_cfl ( k, t_num, t_min, t_max, x_num, x_min, x_max ):

#*****************************************************************************80
#
## fd1d_heat_explicit_cfl(): compute the Courant-Friedrichs-Loewy coefficient.
#
#  Discussion:
#
#    The equation to be solved has the form:
#
#      dUdT - k * d2UdX2 = F(X,T)
#
#    over the interval [X_MIN,X_MAX] with boundary conditions
#
#      U(X_MIN,T) = U_X_MIN(T),
#      U(X_MAX,T) = U_X_MAX(T),
#
#    over the time interval [T_MIN,T_MAX] with initial conditions
#
#      U(X,T_MIN) = U_T_MIN(X)
#
#    The code uses the finite difference method to approximate the
#    second derivative in space, and an explicit forward Euler approximation
#    to the first derivative in time.
#
#    The finite difference form can be written as
#
#      U(X,T+dt) - U(X,T)                  ( U(X-dx,T) - 2 U(X,T) + U(X+dx,T) )
#      ------------------  = F(X,T) + k *  ------------------------------------
#               dt                                   dx * dx
#
#    or, assuming we have solved for all values of U at time T, we have
#
#      U(X,T+dt) = U(X,T) + cfl * ( U(X-dx,T) - 2 U(X,T) + U(X+dx,T) ) + dt * F(X,T) 
#
#    Here "cfl" is the Courant-Friedrichs-Loewy coefficient:
#
#      cfl = k * dt / dx / dx
#
#    In order for accurate results to be computed by this explicit method,
#    the CFL coefficient must be less than 0.5!
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    24 January 2012
#
#  Author:
# 
#    John Burkardt
#
#  Reference:
#
#    George Lindfield, John Penny,
#    Numerical Methods Using MATLAB,
#    Second Edition,
#    Prentice Hall, 1999,
#    ISBN: 0-13-012641-1,
#    LC: QA297.P45.
#
#  Input:
#
#    real K, the heat conductivity coefficient.
#
#    integer T_NUM, the number of time values, including the initial
#    value.
#
#    real T_MIN, T_MAX, the minimum and maximum times.
#
#    integer X_NUM, the number of nodes.
#
#    real X_MIN, X_MAX, the minimum and maximum spatial coordinates.
#
#  Output:
#
#    real CFL, the Courant-Friedrichs-Loewy coefficient.
#
  x_step = ( x_max - x_min ) / ( x_num - 1 )
  t_step = ( t_max - t_min ) / ( t_num - 1 )
#
#  Check the CFL condition, print out its value, and quit if it is too large.
#
  cfl = k * t_step / x_step / x_step

  if ( 0.5 <= cfl ):
    print ( '' )
    print ( 'fd1d_heat_explicit_cfl - Fatal error!' )
    print ( '  CFL condition failed.' )
    print ( '  0.5 <= K * dT / dX / dX = %f' % ( cfl ) )
    raise Exception ( 'fd1d_heat_explicit_cfl - Fatal error!' )

  return cfl

def fd1d_heat_explicit ( x_num, x, t, dt, cfl, rhs, bc, h ):

#*****************************************************************************80
#
## fd1d_heat_explicit(): Finite difference solution of 1D heat equation.
#
#  Discussion:
#
#    This program takes one time step to solve the 1D heat equation 
#    with an explicit method.
#
#    This program solves
#
#      dUdT - k * d2UdX2 = F(X,T)
#
#    over the interval [A,B] with boundary conditions
#
#      U(A,T) = UA(T),
#      U(B,T) = UB(T),
#
#    over the time interval [T0,T1] with initial conditions
#
#      U(X,T0) = U0(X)
#
#    The code uses the finite difference method to approximate the
#    second derivative in space, and an explicit forward Euler approximation
#    to the first derivative in time.
#
#    The finite difference form can be written as
#
#      U(X,T+dt) - U(X,T)                  ( U(X-dx,T) - 2 U(X,T) + U(X+dx,T) )
#      ------------------  = F(X,T) + k *  ------------------------------------
#               dt                                   dx * dx
#
#    or, assuming we have solved for all values of U at time T, we have
#
#      U(X,T+dt) = U(X,T) + cfl * ( U(X-dx,T) - 2 U(X,T) + U(X+dx,T) ) + dt * F(X,T) 
#
#    Here "cfl" is the Courant-Friedrichs-Loewy coefficient:
#
#      cfl = k * dt / dx / dx
#
#    In order for accurate results to be computed by this explicit method,
#    the CFL coefficient must be less than 0.5!
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    31 January 2012
#
#  Author:
# 
#    John Burkardt
#
#  Reference:
#
#    George Lindfield, John Penny,
#    Numerical Methods Using MATLAB,
#    Second Edition,
#    Prentice Hall, 1999,
#    ISBN: 0-13-012641-1,
#    LC: QA297.P45.
#
#  Input:
#
#    integer X_NUM, the number of points to use in the spatial dimension.
#
#    real X(X_NUM,1), the coordinates of the nodes.
#
#    real T, the current time.
#
#    real DT, the size of the time step.
#
#    real CFL, the Courant-Friedrichs-Loewy coefficient,
#    computed by fd1d_heat_explicit_cfl.
#
#    real H(X_NUM,1), the solution at the current time.
#
#    @RHS, the function which evaluates the right hand side.
#
#    @BC, the function which evaluates the boundary conditions.
#
#  Output:
#
#    real H_NEW(X_NUM,1), the solution at time T+DT.
#
  import numpy as np

  h_new = np.zeros ( x_num )

  f = rhs ( x_num, x, t )

  for c in range ( 1, x_num - 1 ):
    l = c - 1
    r = c + 1
    h_new[c] = h[c] + cfl * ( h[l] - 2.0 * h[c] + h[r] ) + dt * f[c]

  h_new = bc ( x_num, x, t + dt, h_new )

  return h_new

def fd1d_heat_explicit_test01 ( ):

#*****************************************************************************80
#
## fd1d_heat_explicit_test01() does a simple test problem
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt
  import numpy as np
  import platform
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter

  print ( '' )
  print ( 'fd1d_heat_explicit_test01:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Compute an approximate solution to the time-dependent' )
  print ( '  one dimensional heat equation:' )
  print ( '' )
  print ( '    dH/dt - K * d2H/dx2 = f(x,t)' )
  print ( '' )
  print ( '  Run a simple test case.' )
#
#  Heat coefficient.
#
  k = k_test01 ( )
#
#  X_NUM is the number of equally spaced nodes to use between 0 and 1.
#
  x_num = 21
  x_min = 0.0
  x_max = 1.0
  dx = ( x_max - x_min ) / ( x_num - 1 )
  x = np.linspace ( x_min, x_max, x_num )
#
#  T_NUM is the number of equally spaced time points between 0 and 10.0.
#
  t_num = 501
  t_min = 0.0
  t_max = 150.0
  dt = ( t_max - t_min ) / ( t_num - 1 )
  t = np.linspace ( t_min, t_max, t_num )
#
#  Get the CFL coefficient.
#
  cfl = fd1d_heat_explicit_cfl ( k, t_num, t_min, t_max, x_num, x_min, x_max )

  print ( '' )
  print ( '  Number of X nodes = %d' % ( x_num ) )
  print ( '  X interval is [%f,%f]' % ( x_min, x_max ) )
  print ( '  X spacing is %f' % ( dx ) )
  print ( '  Number of T values = %d' % ( t_num ) )
  print ( '  T interval is [%f,%f]' % ( t_min, t_max ) )
  print ( '  T spacing is %f' % ( dt ) )
  print ( '  Constant K = %g' % ( k ) )
  print ( '  CFL coefficient = %g' % ( cfl ) )
#
#  Running the code produces an array H of temperatures H(t,x),
#  and vectors x and t.
#
  hmat = np.zeros ( ( x_num, t_num ) )

  for j in range ( 0, t_num ):
    if ( j == 0 ):
      h = ic_test01 ( x_num, x, t[j] )
      h = bc_test01 ( x_num, x, t[j], h )
    else:
      h = fd1d_heat_explicit ( x_num, x, t[j-1], dt, cfl, rhs_test01, bc_test01, h )
    for i in range ( 0, x_num ):
      hmat[i,j] = h[i]
#
#  Plot the data.
#
  tmat, xmat = np.meshgrid ( t, x )
  fig = plt.figure ( )
# ax = fig.add_subplot ( 111, projection = '3d' )
  ax = Axes3D ( fig )
  surf = ax.plot_surface ( xmat, tmat, hmat )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---T--->' )
  plt.title ( 'U(X,T)' )
  plt.savefig ( 'plot_test01.png' )
  plt.show ( block = False )
  plt.close ( )
#
#  Write the data to files.
#
  r8mat_write ( 'h_test01.txt', x_num, t_num, hmat )
  r8vec_write ( 't_test01.txt', t_num, t )
  r8vec_write ( 'x_test01.txt', x_num, x )

  print ( '' )
  print ( '  H(X,T) written to "h_test01.txt"' )
  print ( '  T values written to "t_test01.txt"' )
  print ( '  X values written to "x_test01.txt"' )
#
#  Terminate.
#
  print ( '' )
  print ( 'fd1d_heat_explicit_test01:' )
  print ( '  Normal end of execution.' )
  return

def bc_test01 ( x_num, x, t, h ):

#*****************************************************************************80
#
## bc_test01() evaluates the boundary conditions for problem 1.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the current time.
#
#    real H(X_NUM), the current heat values.
#
#  Output:
#
#    real H(X_NUM), the current heat values, after boundary
#    conditions have been imposed.
#
  h[0]       = 90.0
  h[x_num-1] = 70.0

  return h

def ic_test01 ( x_num, x, t ):

#*****************************************************************************80
#
## ic_test01() evaluates the initial condition for problem 1.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM), the node coordinates.
#
#    real T, the initial time.
#
#  Output:
#
#    real H(X_NUM), the heat values at the initial time.
#
  import numpy as np

  h = np.zeros ( x_num )

  for i in range ( 0, x_num ):
    h[i] = 50.0

  return h

def k_test01 ( ):

#*****************************************************************************80
#
## k_test01() evaluates the conductivity for problem 1.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Output:
#
#    real K, the conductivity.
#
  k = 0.002

  return k

def rhs_test01 ( x_num, x, t ):

#*****************************************************************************80
#
## rhs_test01() evaluates the right hand side for problem 1.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the current time.
#
#  Output:
#
#    real VALUE(X_NUM,1), the source term.
#
  import numpy as np

  value = np.zeros ( x_num )

  return value

def fd1d_heat_explicit_test02 ( ):

#*****************************************************************************80
#
## fd1d_heat_explicit_test02() does a problem with known solution.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt
  import numpy as np
  import platform
  from math import sqrt
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter

  print ( '' )
  print ( 'fd1d_heat_explicit_test02:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Compute an approximate solution to the time-dependent' )
  print ( '  one dimensional heat equation for a problem where we' )
  print ( '  know the exact solution.' )
  print ( '' )
  print ( '    dH/dt - K * d2H/dx2 = f(x,t)' )
  print ( '' )
  print ( '  Run a simple test case.' )
#
#  Heat coefficient.
#
  k = k_test02 ( )
#
#  X_NUM is the number of equally spaced nodes to use between 0 and 1.
#
  x_num = 21
  x_min = 0.0
  x_max = 1.0
  dx = ( x_max - x_min ) / ( x_num - 1 )
  x = np.linspace ( x_min, x_max, x_num )
#
#  T_NUM is the number of equally spaced time points between 0 and 10.0.
#
  t_num = 26
  t_min = 0.0
  t_max = 10.0
  dt = ( t_max - t_min ) / ( t_num - 1 )
  t = np.linspace ( t_min, t_max, t_num )
#
#  Get the CFL coefficient.
#
  cfl = fd1d_heat_explicit_cfl ( k, t_num, t_min, t_max, x_num, x_min, x_max )

  print ( '' )
  print ( '  Number of X nodes = %d' % ( x_num ) )
  print ( '  X interval is [%f,%f]' % ( x_min, x_max ) )
  print ( '  X spacing is %f' % ( dx ) )
  print ( '  Number of T values = %d' % ( t_num ) )
  print ( '  T interval is [%f,%f]' % ( t_min, t_max ) )
  print ( '  T spacing is %f' % ( dt ) )
  print ( '  Constant K = %g' % ( k ) )
  print ( '  CFL coefficient = %g' % ( cfl ) )
#
#  Running the code produces an array H of temperatures H(t,x),
#  and vectors x and t.
#
  gmat = np.zeros ( ( x_num, t_num ) )
  hmat = np.zeros ( ( x_num, t_num ) )

  print ( '' )
  print ( '  Step            Time       RMS Error' )
  print ( '' )

  for j in range ( 0, t_num ):

    if ( j == 0 ):
      h = ic_test02 ( x_num, x, t[j] )
      h = bc_test02 ( x_num, x, t[j], h )
    else:
      h = fd1d_heat_explicit ( x_num, x, t[j-1], dt, cfl, rhs_test02, bc_test02, h )

    g = exact_test02 ( x_num, x, t[j] )

    e = 0.0
    for i in range ( 0, x_num ):
      e = e + ( h[i] - g[i] ) ** 2
    e = sqrt ( e ) / sqrt ( x_num )

    print ( '  %4d  %14.6g  %14.6g' % ( j, t[j], e ) )
    for i in range ( 0, x_num ):
      gmat[i,j] = g[i]
      hmat[i,j] = h[i]
#
#  Plot the data.
#
  tmat, xmat = np.meshgrid ( t, x )
  fig = plt.figure ( )
  ax = Axes3D ( fig )
  surf = ax.plot_surface ( xmat, tmat, hmat )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---T--->' )
  plt.title ( 'U(X,T)' )
  plt.savefig ( 'plot_test02.png' )
  plt.show ( block = False )
  plt.close ( )
#
#  Write the data to files.
#
  r8mat_write ( 'g_test02.txt', x_num, t_num, gmat )
  r8mat_write ( 'h_test02.txt', x_num, t_num, hmat )
  r8vec_write ( 't_test02.txt', t_num, t )
  r8vec_write ( 'x_test02.txt', x_num, x )

  print ( '' )
  print ( '  G(X,T) written to "g_test02.txt"' )
  print ( '  H(X,T) written to "h_test02.txt"' )
  print ( '  T values written to "t_test02.txt"' )
  print ( '  X values written to "x_test02.txt"' )
#
#  Terminate.
#
  print ( '' )
  print ( 'fd1d_heat_explicit_test02:' )
  print ( '  Normal end of execution.' )
  return

def bc_test02 ( x_num, x, t, h ):

#*****************************************************************************80
#
## bc_test02() evaluates the boundary conditions for problem 2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the current time.
#
#    real H(X_NUM,1), the current heat values.
#
#  Output:
#
#    real H(X_NUM,1), the current heat values, after boundary
#    conditions have been imposed.
#
  import numpy as np

  h = np.zeros ( x_num )
#
#  Because exact_test02 expects an array as an input argument,
#  we can't simply pass the scalar x[0], but have to create an
#  array.
#
  x_array = np.zeros ( 1 )

  x_array[0] = x[0]
  h[0]       = exact_test02 ( 1, x_array, t )

  x_array[0] = x[x_num-1]
  h[x_num-1] = exact_test02 ( 1, x_array, t )

  return h

def exact_test02 ( x_num, x, t ):

#*****************************************************************************80
#
## exact_test02() evaluates the exact solution for problem 2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM), the node coordinates.
#
#    real T, the initial time.
#
#  Output:
#
#    real H(X_NUM), the exact solution at X and T.
#
  from math import exp
  from math import sin
  from math import sqrt
  import numpy as np

  k = k_test02 ( )

  h = np.zeros ( x_num )

  for i in range ( 0, x_num ):
    h[i] = exp ( - t ) * sin ( sqrt ( k ) * x[i] )

  return h

def ic_test02 ( x_num, x, t ):

#*****************************************************************************80
#
## ic_test02() evaluates the initial condition for problem 2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the initial time.
#
#  Output:
#
#    real H(X_NUM,1), the heat values at the initial time.
#
  h = exact_test02 ( x_num, x, t )

  return h

def k_test02 ( ):

#*****************************************************************************80
#
## k_test02() evaluates the conductivity for problem 2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Output:
#
#    real K, the conductivity.
#
  k = 0.002

  return k

def rhs_test02 ( x_num, x, t ):

#*****************************************************************************80
#
## rhs_test02() evaluates the right hand side for problem 2.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the current time.
#
#  Output:
#
#    real VALUE(X_NUM,1), the source term.
#
  import numpy as np

  value = np.zeros ( x_num )

  return value

def fd1d_heat_explicit_test03 ( ):

#*****************************************************************************80
#
## fd1d_heat_explicit_test03() does a simple test problem.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt
  import numpy as np
  import platform
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter

  print ( '' )
  print ( 'fd1d_heat_explicit_test03:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Compute an approximate solution to the time-dependent' )
  print ( '  one dimensional heat equation:' )
  print ( '' )
  print ( '    dH/dt - K * d2H/dx2 = f(x,t)' )
  print ( '' )
  print ( '  Run a simple test case.' )
#
#  Heat coefficient.
#
  k = k_test03 ( )
#
#  X_NUM is the number of equally spaced nodes to use between 0 and 1.
#
  x_num = 21
  x_min = -10.0
  x_max = +5.0
  dx = ( x_max - x_min ) / ( x_num - 1 )
  x = np.linspace ( x_min, x_max, x_num )
#
#  T_NUM is the number of equally spaced time points between 0 and 10.0.
#
  t_num = 2001
  t_min = 0.0
  t_max = 50.0
  dt = ( t_max - t_min ) / ( t_num - 1 )
  t = np.linspace ( t_min, t_max, t_num )
#
#  Get the CFL coefficient.
#
  cfl = fd1d_heat_explicit_cfl ( k, t_num, t_min, t_max, x_num, x_min, x_max )

  print ( '' )
  print ( '  Number of X nodes = %d' % ( x_num ) )
  print ( '  X interval is [%f,%f]' % ( x_min, x_max ) )
  print ( '  X spacing is %f' % ( dx ) )
  print ( '  Number of T values = %d' % ( t_num ) )
  print ( '  T interval is [%f,%f]' % ( t_min, t_max ) )
  print ( '  T spacing is %f' % ( dt ) )
  print ( '  Constant K = %g' % ( k ) )
  print ( '  CFL coefficient = %g' % ( cfl ) )
#
#  Running the code produces an array H of temperatures H(t,x),
#  and vectors x and t.
#
  hmat = np.zeros ( ( x_num, t_num ) )

  for j in range ( 0, t_num ):
    if ( j == 0 ):
      h = ic_test03 ( x_num, x, t[j] )
      h = bc_test03 ( x_num, x, t[j], h )
    else:
      h = fd1d_heat_explicit ( x_num, x, t[j-1], dt, cfl, rhs_test03, bc_test03, h )
    for i in range ( 0, x_num ):
      hmat[i,j] = h[i]
#
#  Plot the data.
#
  tmat, xmat = np.meshgrid ( t, x )
  fig = plt.figure ( )
  ax = Axes3D ( fig )
  surf = ax.plot_surface ( xmat, tmat, hmat )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---T--->' )
  plt.title ( 'U(X,T)' )
  plt.savefig ( 'plot_test03.png' )
  plt.show ( block = False )
  plt.close ( )
#
#  Write the data to files.
#
  r8mat_write ( 'h_test03.txt', x_num, t_num, hmat )
  r8vec_write ( 't_test03.txt', t_num, t )
  r8vec_write ( 'x_test03.txt', x_num, x )

  print ( '' )
  print ( '  H(X,T) written to "h_test03.txt"' )
  print ( '  T values written to "t_test03.txt"' )
  print ( '  X values written to "x_test3.txt"' )
#
#  Terminate.
#
  print ( '' )
  print ( 'fd1d_heat_explicit_test03:' )
  print ( '  Normal end of execution.' )
  return

def bc_test03 ( x_num, x, t, h ):

#*****************************************************************************80
#
## bc_test03() evaluates the boundary conditions for problem 3.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the current time.
#
#    real H(X_NUM,1), the current heat values.
#
#  Output:
#
#    real H(X_NUM,1), the current heat values, after boundary
#    conditions have been imposed.
#
  h[0]       = 15.0
  h[x_num-1] = 25.0

  return h

def ic_test03 ( x_num, x, t ):

#*****************************************************************************80
#
## ic_test03() evaluates the initial condition for problem 3.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM,1), the node coordinates.
#
#    real T, the initial time.
#
#  Output:
#
#    real H(X_NUM,1), the heat values at the initial time.
#
  import numpy as np

  h = np.zeros ( x_num )

  for i in range ( 0, x_num ):
    if ( x[i] < 0.0 ):
      h[i] = 15.0
    elif ( x[i] == 0.0 ):
      h[i] = 20.0
    else:
      h[i] = 25.0

  return h

def k_test03 ( ):

#*****************************************************************************80
#
## k_test03() evaluates the conductivity for problem 3.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Output:
#
#    real K, the conductivity.
#
  k = 2.0

  return k

def rhs_test03 ( x_num, x, t ):

#*****************************************************************************80
#
## rhs_test03() evaluates the right hand side for problem 3.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    integer X_NUM, the number of nodes.
#
#    real X(X_NUM), the node coordinates.
#
#    real T, the current time.
#
#  Output:
#
#    real VALUE(X_NUM), the source term.
#
  import numpy as np

  value = np.zeros ( x_num )

  return value

def fd1d_heat_explicit_test ( ):

#*****************************************************************************80
#
## fd1d_heat_explicit_test() tests fd1d_heat_explicit().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
  import platform

  print ( '' )
  print ( 'fd1d_heat_explicit_test():' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test fd1d_heat_explicit().' )

  fd1d_heat_explicit_test01 ( )
  fd1d_heat_explicit_test02 ( )
  fd1d_heat_explicit_test03 ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'fd1d_heat_explicit_test():' )
  print ( '  Normal end of execution.' )
  return

def r8mat_write ( filename, m, n, a ):

#*****************************************************************************80
#
## r8mat_write() writes an R8MAT to a file.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 October 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    string FILENAME, the name of the output file.
#
#    integer M, the number of rows in A.
#
#    integer N, the number of columns in A.
#
#    real A(M,N), the matrix.
#
  output = open ( filename, 'w' )

  for i in range ( 0, m ):
    for j in range ( 0, n ):
      s = '  %g' % ( a[i,j] )
      output.write ( s )
    output.write ( '\n' )

  output.close ( )

  return

def r8mat_write_test ( ):

#*****************************************************************************80
#
## r8mat_write_test() tests r8mat_write().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 October 2014
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'r8mat_write_test:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test r8mat_write, which writes an R8MAT to a file.' )

  filename = 'r8mat_write_test.txt'
  m = 5
  n = 3
  a = np.array ( (  \
    ( 1.1, 1.2, 1.3 ), \
    ( 2.1, 2.2, 2.3 ), \
    ( 3.1, 3.2, 3.3 ), \
    ( 4.1, 4.2, 4.3 ), \
    ( 5.1, 5.2, 5.3 ) ) )
  r8mat_write ( filename, m, n, a )

  print ( '' )
  print ( '  Created file "%s".' % ( filename ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'r8mat_write_test:' )
  print ( '  Normal end of execution.' )
  return
  
def r8vec_write ( filename, n, a ):

#*****************************************************************************80
#
## r8vec_write() writes an R8VEC to a file.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    string FILENAME, the name of the output file.
#
#    integer N, the number of entries in A.
#
#    real A(N), the matrix.
#
  output = open ( filename, 'w' )

  for i in range ( 0, n ):
    s = '  %g\n' % ( a[i] )
    output.write ( s )

  output.close ( )

  return

def r8vec_write_test ( ):

#*****************************************************************************80
#
## r8vec_write_test() tests r8vec_write().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    06 November 2014
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'r8vec_write_test:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test r8vec_write, which writes an R8VEC to a file.' )
  filename = 'r8vec_write_test.txt'
  n = 5
  a = np.array ( ( 1.1, 2.2, 3.3, 4.4, 5.5 ) )
  r8vec_write ( filename, n, a )

  print ( '' )
  print ( '  Created file "%s".' % ( filename ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'r8vec_write_test:' )
  print ( '  Normal end of execution.' )
  return
  
def timestamp ( ):

#*****************************************************************************80
#
## timestamp() prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

if ( __name__ == '__main__' ):
  timestamp ( )
  fd1d_heat_explicit_test ( )
  timestamp ( )
