#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed March 15  15:50:00 2021

@author: Moises Zeleny
"""
from numpy import sqrt, complex256, roots
from numpy import abs as npabs


def rootsx2_inv(a,b,c):
    '''
    Solutions of quadratic equation witb an alternative form of the solutions:
    x_{1,2} = -(2*c)/(b \mp sqrt(b**2 - 4*a*c))
    
    Parameters:
        a,b,c: float
    
    Returns x1 and x2 solutions of quadratic equation.
    '''
    r = b**2 - 4*a*c
    x1 = -(2*c)/(b - sqrt(r,dtype=complex256))
    x2 = -(2*c)/(b + sqrt(r,dtype=complex256))
    return x1,x2

def rootsx2(a,b,c):
    '''
    Solutions of quadratic equation :
    x_{1,2} = (-b \mp sqrt(b**2 - 4*a*c))/(2*a)
    
    Parameters:
        a,b,c: float
    
    Returns x1 and x2 solutions of quadratic equation.
    '''
    r = b**2 - 4*a*c
    x1 = (-b - sqrt(r,dtype=complex256))/(2*a)
    x2 = (-b + sqrt(r,dtype=complex256))/(2*a)
    return x1,x2

def rootx2_estable(a,b,c):
    '''
    Solutions of quadratic equation stable numerically:
    x_{1} = (-b - sqrt(b**2 - 4*a*c))/(2*a)
    x_{2} = -(2*c)/(b + sqrt(r,dtype=complex256))
    
    Parameters:
        a,b,c: float
    
    Returns x1 and x2 solutions of quadratic equation.
    '''
    x1 = (-b - sqrt(b**2 - 4*a*c,dtype=complex256))/(2*a)
    x2 = -(2*c)/(b + sqrt(b**2 - 4*a*c,dtype=complex256))
    return x1,x2

#######################3
#### Roots of x1, x2
########################
def x1(ma,M1,M2):
    a = 1.0
    b = - ((ma**2 -M1**2 + M2**2)/ma**2)
    c = (M2**2)/ma**2
    f = lambda x: a*x**2 + b*x + c
    root1 = roots([a,b,c])[0]
    return root1

def x2(ma,M1,M2):
    a = 1.0
    b = - ((ma**2 -M1**2 + M2**2)/ma**2)
    c = (M2**2)/ma**2
    return roots([a,b,c])[1]

#######################3
#### Roots of y11, y12s
########################
def y11(ma,mi,M0,M1):
    a = ma**2
    b = - (mi**2 + M1**2-M0**2)
    c = M1**2
    return roots([a,b,c])[0]

def y12(ma,mi,M0,M1):
    a = ma**2
    b = - (mi**2 + M1**2-M0**2)
    c = M1**2
    return roots([a,b,c])[1]

#######################3
#### Roots of y11, y12
########################
def y21(ma,mj,M0,M2):
    a = ma**2
    b = - (mj**2 + M2**2-M0**2)
    c = M2**2
    return roots([a,b,c])[0]

def y22(ma,mj,M0,M2):
    a = ma**2
    b = - (mj**2 + M2**2-M0**2)
    c = M2**2
    return roots([a,b,c])[1]