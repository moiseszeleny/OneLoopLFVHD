#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17  13:49:53 2021

@author: Moises Zeleny
"""
from mpmath import log, polylog, pi, mpf, log1p
import mpmath as mp
#mp.dps = 80; mp.pretty = True
#from numba import jit, njit, prange

##################################################################################################
# Funciones de Passarino Veltman pertinentes para LFVHD
##################################################################################################


from .roots_mpmath2 import x1,x2, rootsx2_inv

#@jit
def Li2(x0):
    return polylog(2,x0)

#@jit
def R0(x0,xi):
    return Li2(x0/(x0-xi)) - Li2((x0-1.0)/(x0-xi))

#R0_aprox = lambda x0,xi: Li2_aprox(x0/(x0-xi)) - Li2_aprox((x0-1)/(x0-xi))
#@jit
def x0(ma,M0,M2):
    return (M2**2-M0**2)/ma**2
#@jit
def x3(M0,M1): 
    return (-M0**2)/(M1**2-M0**2)#where(M1!=M0,(-M0**2)/(M1**2-M0**2),nan_to_num((-M0**2)/(M1**2-M0**2)))


from sympy import symbols, lambdify, integrate
y,t = symbols('y,t')
from sympy import log as logsp
f0 = integrate(logsp(1-(t/y)),(t,0,1))
f1 = 2*integrate(t*logsp(1-(t/y)),(t,0,1))

f0np = lambda y:y*log(-1.0*y) - y*log1p(-y) + log1p(-1.0/y)- 1.0

f1np = lambda y: mp.power(y,2)*log(-y) - mp.power(y,2)*log1p(-y) - y + log1p(-1/y) - mpf('0.5') #lambdify([y],f1,'mpmath')


from .roots_mpmath2 import y11 as y11np
from .roots_mpmath2 import y12 as y12np
from .roots_mpmath2 import y21 as y21np
from .roots_mpmath2 import y22 as y22np

#@jit
def f01sum(mi,M0,M1):
    out = 0.0
    for yi1j in (y11np,y12np): 
        out += f0np(yi1j(mi,M0,M1))
    return out
#f01sum

#@jit
def f02sum(mj,M0,M2):
    out = 0
    for yi2j in (y21np,y22np):
        out += f0np(yi2j(mj,M0,M2))
    return out
#f02sum
#@jit
def f11sum(mi,M0,M1):
    out = 0
    for yi1j in (y11np, y12np):
        out += f1np(yi1j(mi,M0,M1))
    return out
#f11sum
#@jit
def f12sum(mj,M0,M2):
    out = 0
    for yi2j in (y21np, y22np):
        out += f1np(yi2j(mj,M0,M2))
    return out
#f12sum
#@jit
def b1_0np(mi,M0,M1):
    return -log(M0**2)-f01sum(mi,M0,M1)

#@jit
def b2_0np(mj,M0,M2):
    return -log(M0**2)-f02sum(mj,M0,M2)

#@jit
def b1_1np(mi,M0,M1):
    return mpf('0.5')*(-log(M0**2) - f11sum(mi,M0,M1)) 

#@jit
def b2_1np(mj,M0,M2): 
    return - mpf('0.5')*(-log(M0**2) - f12sum(mj,M0,M2))

#########################################################################
# Numpy definitions of PaVe functions
#a1 = 1j/(16*mp.pi**2)
#a2 = -1j/mp.pi**2

#@jit
def A0(ma,M):
    return M**2*(1+log((ma**2)/(M**2)))

#@jit
def B1_0(mi,M0,M1):
    return b1_0np(mi,M0,M1)
    #return 1-log(M1**2/ma**2) + (M0**2)/(M0**2-M1**2)*log(M1**2/M0**2)

#@jit
def B2_0(mj,M0,M2):
    return b2_0np(mj,M0,M2)
    #return 1-log(M2**2/ma**2) + (M0**2)/(M0**2-M2**2)*log(M2**2/M0**2)

#@jit
def B1_1(mi,M0,M1):
    return b1_1np(mi,M0,M1)

#@jit
def B2_1(mj,M0,M2):
    return b2_1np(mj,M0,M2)


#@jit
def B12_0(ma,M1,M2):
    xi1,xi2 = x1(ma,M1,M2),x2(ma,M1,M2)
    #return (log((ma**2)/(M1**2))/2.0 + xi1*log((xi1- 1.0)/xi1) + xi2*log((xi2- 1.0)/xi2))
    return 2 -log(M1**2) + xi1*log((xi1- 1.0)/xi1) + xi2*log((xi2- 1.0)/xi2)
    #return log((ma**2 )/(M1**2))/2 + sum(x*log(1-1/x) for x in [xi1,xi2])

#@jit
def B12_1(ma,M1,M2):
    return ((1/(2*ma**2))*(M1**2*(1+log(ma**2/M1**2)) - M2**2*(1+log(ma**2/M2**2))) + B12_0(ma,M1,M2)/(2*ma**2)*(M2**2-M1**2 + ma**2))

#@jit
def B12_2(ma,M1,M2):
    return ((1/(2*ma**2))*(M1**2*(1+log(ma**2/M1**2)) - M2**2*(1+log(ma**2/M2**2))) + B12_0(ma,M1,M2)/(2*ma**2)*(M2**2-M1**2 - ma**2))

#@jit
def C0(ma,M0,M1,M2):
    y0 = x0(ma,M0,M2)
    y1 = x1(ma,M1,M2)
    y2 = x2(ma,M1,M2)
    y3 = x3(M0,M1)
    return ((R0(y0,y1) + R0(y0,y2) - R0(y0,y3))/ma**2)#*(-1j*16)


#@jit
def C1(ma,mi,M0,M1,M2):
    return ((1/ma**2)*(B1_0(mi,M0,M1) - B12_0(ma,M1,M2) + (M2**2-M0**2)*C0(ma,M0,M1,M2)))

#@jit
def C2(ma,mj,M0,M1,M2):
    return ( (-1/ma**2)*(B2_0(mj,M0,M2) - B12_0(ma,M1,M2) + (M1**2-M0**2)*C0(ma,M0,M1,M2)))

