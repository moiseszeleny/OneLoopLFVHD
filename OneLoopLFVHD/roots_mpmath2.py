#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17  13:49:53 2021

@author: Moises Zeleny
"""
from mpmath import sqrt, mpf
import mpmath as mp
#mp.dps = 35; mp.pretty = True

#from numba import jit, njit

#@jit()
def rootsx2_inv(a,b,c):
    '''
    Solutions of quadratic equation witb an alternative form of the solutions:
    x_{1,2} = -(2*c)/(b \mp sqrt(b**2 - 4*a*c))
    
    Parameters:
        a,b,c: float
    
    Returns x1 and x2 solutions of quadratic equation.
    '''
    r = b**2 - 4*a*c
    x1 = -(2*c)/(b - sqrt(r))
    x2 = -(2*c)/(b + sqrt(r))
    return x1,x2

#@jit()
def rootsx2(a,b,c):
    '''
    Solutions of quadratic equation :
    x_{1,2} = (-b \mp sqrt(b**2 - 4*a*c))/(2*a)
    
    Parameters:
        a,b,c: float
    
    Returns x1 and x2 solutions of quadratic equation.
    '''
    r = b**2 - 4*a*c
    x1 = (-b - mp.sqrt(r))/(2*a)
    x2 = (-b + mp.sqrt(r))/(2*a)
    return x1,x2

#@jit()
def rootx2_estable(a,b,c):
    '''
    Solutions of quadratic equation stable numerically:
    x_{1} = (-b - sqrt(b**2 - 4*a*c))/(2*a)
    x_{2} = -(2*c)/(b + sqrt(r,dtype=complex256))
    
    Parameters:
        a,b,c: float
    
    Returns x1 and x2 solutions of quadratic equation.
    '''
    x1 = (-b - sqrt(b**2 - 4*a*c))/(2*a)
    x2 = -(2*c)/(b + sqrt(b**2 - 4*a*c))
    return x1,x2


#######################3
#### Roots of x1, x2
########################
#@jit()
def x1(ma,M1,M2):
#     ma = mpf(str(ma))
#     M1 = mpf(str(M1))
#     M2 = mpf(str(M2))
    a = mpf('1.0')
    b = - ((mp.power(ma,2) -mp.power(M1,2) + mp.power(M2,2))/mp.power(ma,2))
    c = mp.power(M2,2)/mp.power(ma,2)
    root1 = mp.polyroots([a,b,c],extraprec=100)[0]#roots([a,b,c])[0]
    return root1

#@jit()
def x2(ma,M1,M2):
    a = mpf('1.0') #1.0
    b = - ((mp.power(ma,2) -mp.power(M1,2) + mp.power(M2,2))/mp.power(ma,2))#- ((ma**2 -M1**2 + M2**2)/ma**2)
    c = mp.power(M2,2)/mp.power(ma,2)#(M2**2)/ma**2
    return mp.polyroots([a,b,c],extraprec=100)[1]#rootx2_estable(a,b,c)[1]

#######################3
#### Roots of y11, y12s
########################
#@jit()
def y11(mi,M0,M1):
    a = mp.power(mi,2) #mi**2
    b = - (mp.power(mi,2) + mp.power(M0,2) - mp.power(M1,2))#- (mi**2 + M0**2 - M1**2)
    c = mp.power(M0,2) #M0**2
    return mp.polyroots([a,b,c],extraprec=120)[0]#roots([a,b,c])[0]

#@jit()
def y12(mi,M0,M1):
    a = mp.power(mi,2) #mi**2
    b = - (mp.power(mi,2) + mp.power(M0,2) - mp.power(M1,2))#- (mi**2 + M0**2 - M1**2)
    c = mp.power(M0,2) #M0**2
    return mp.polyroots([a,b,c],extraprec=120)[1]#rootx2_estable(a,b,c)[1]

#######################3
#### Roots of y11, y12
########################
#@jit()
def y21(mj,M0,M2):
    a = mp.power(mj,2)# mj**2
    b = - (mp.power(mj,2) + mp.power(M0,2) - mp.power(M2,2))#- (mj**2 + M0**2 - M2**2)
    c = mp.power(M0,2) #M0**2
    return mp.polyroots([a,b,c],extraprec=120)[0]#roots([a,b,c])[0]
#@jit()
def y22(mj,M0,M2):
    a = mp.power(mj,2)# mj**2
    b = - (mp.power(mj,2) + mp.power(M0,2) - mp.power(M2,2))#- (mj**2 + M0**2 - M2**2)
    c = mp.power(M0,2) #M0**2
    return mp.polyroots([a,b,c],extraprec=120)[1]#rootx2_estable(a,b,c)[1]
