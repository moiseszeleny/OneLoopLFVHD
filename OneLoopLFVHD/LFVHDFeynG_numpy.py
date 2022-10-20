#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16  01:27:53 2021

@author: Moises Zeleny
"""
from numpy import log,log1p,conjugate,pi, where,array, vectorize
from numpy import complex128, complex256,float64, float128
from numpy import nan_to_num
from numpy import sum as sumnp
from numpy import abs as npabs

δ = 0
from scipy.special import spence
def sci_polylog(s,z):
    return spence(1-z)#
##################################################################################################
# Funciones de Passarino Veltman pertinentes para LFVHD
##################################################################################################

# Definiendo divergencias
Δe = 0#symbols(r'\Delta_\epsilon')
from sympy import symbols, lambdify, integrate, solve
# definición de xk
_x,_m1,_m2, _ma = symbols('x,m1,m2,ma',positive=True)
_funcion = _x**2 - ((_ma**2 -_m1**2 + _m2**2)/_ma**2)*_x + (_m2**2)/_ma**2
_sols = solve(_funcion,_x)
_solsnp = lambdify([_ma,_m1,_m2],_sols,'numpy')


from .roots import x1,x2, rootsx2_inv

# C0 en terminos de R0
Li2 = lambda x0: sci_polylog(2,x0)
R0 = lambda x0,xi: Li2(x0/(x0-xi)) - Li2((x0-1.0)/(x0-xi))

#R0_aprox = lambda x0,xi: Li2_aprox(x0/(x0-xi)) - Li2_aprox((x0-1)/(x0-xi))
x0 = lambda ma,M0,M2: (M2**2-M0**2)/ma**2
def x3(M0,M1): 
    return (-M0**2)/(M1**2-M0**2)#where(M1!=M0,(-M0**2)/(M1**2-M0**2),nan_to_num((-M0**2)/(M1**2-M0**2)))

# Definiciones para las funciones b: https://www.sciencedirect.com/science/article/pii/S0550321317301785
y,t = symbols('y,t')
from sympy import log as logsp
f0 = integrate(logsp(1-(t/y)),(t,0,1))
f1 = 2*integrate(t*logsp(1-(t/y)),(t,0,1))
################################
def safe_log(x):
    return where(x!=0,log(x),0)

def safe_log1p(x):
    return where(x!=-1,log1p(x),0)
##############################3
#convirtiendo f0 y f1 en funciones simbolicas
from numpy import log1p
#f0np = lambda y:y*safe_log(-1.0*y) - y*safe_log(1.0-y) + safe_log1p(-1.0/y)- 1.0 #lambdify([y],f0,'numpy')
#f1np = lambdify([y],f1,'numpy')

# Aproximaciones de f_n provenienetes de Seesaw new discussions
# def f0np(y):
#     down = (1-y)*log((y-1)/y,dtype=complex256)-1 
#     X = 0
#     for l in range(1,21):
#         X += y**(-l)/(l+1)

#     upper = log((y-1)/y,dtype=complex256) + X
#     return where(npabs(y)<10, down,upper)

# def f1np(y):
#     down = (1-y**2)*log((y-1)/y)-(y + 0.5) 
#     X = 0
#     for l in range(2,22):
#         X += y**(1-l)/(l+1)

#     upper = log((y-1)/y,dtype=complex256) + X
#     return where(npabs(y)<10, down,upper)

def f0np(y):
    '''
    f0 function

    Parameters
    ----------
        y: float, mpf
    '''
    # if y == 1:
    #     out = -1
    # else:
    #     out = y*log(-1.0*y) - y*log1p(-y) + log1p(-1.0/y) - 1.0
    # return out
    return y*log(-1.0*y) - y*log1p(-y) + log1p(-1.0/y) - 1.0


def f1np(y):
    '''
    f1 function

    Parameters
    ----------
        y: float, mpf
    '''
    # if y == 1:
    #     out = -mpf('3')/2
    # else:
    #     out = mp.power(y, 2)*log(-y) -\
    #         mp.power(y, 2)*log1p(-y) -\
    #         y + log1p(-1/y) - mpf('0.5')  # lambdify([y],f1,'mpmath')
    # return out
    return y**2*log(-y) -\
        y**2*log1p(-y) -\
        y + log1p(-1/y) - 0.5

#soluciones de la ecuación
M0,M1,M2 = symbols('M_0,M_1,M_2',positive=True)
_mi,_mj = symbols('mi,mj',positive=True)

ec1 = y**2*_ma**2 - y*(_mi**2 + M1**2-M0**2) + M1**2
ec2 = y**2*_ma**2 - y*(_mj**2 + M2**2-M0**2) + M2**2
y11,y12 = solve(ec1,y)
y21,y22 = solve(ec2,y)

from .roots import y11 as y11np
from .roots import y12 as y12np
from .roots import y21 as y21np
from .roots import y22 as y22np


#y2j = vectorize(y2j)

f01sum = lambda mi,M0,M1:f0np(y11np(mi,M0,M1)) + f0np(y12np(mi,M0,M1))#sumnp(f0np(yi1j(mi,M0,M1)) for yi1j in [y11np,y12np ])
#f01sum

f02sum = lambda mj,M0,M2:f0np(y21np(mj,M0,M2)) + f0np(y22np(mj,M0,M2))#sumnp(f0np(yi2j(mj,M0,M2)) for yi2j in [y21np,y22np ])
#f02sum

f11sum = lambda mi,M0,M1:f1np(y11np(mi,M0,M1)) + f1np(y12np(mi,M0,M1))#sumnp(f1np(yi1j(mi,M0,M1)) for yi1j in [y11np, y12np ])
#f11sum

f12sum = lambda mj,M0,M2:f1np(y21np(mj,M0,M2)) + f1np(y22np(mj,M0,M2))#sumnp(f1np(yi2j(mj,M0,M2)) for yi2j in [y21np, y22np])
#f12sum

b1_0np = lambda mi,M0,M1:-log(M1**2)-f01sum(mi,M0,M1)
b2_0np = lambda mj,M0,M2: -log(M2**2)-f02sum(mj,M0,M2)
b1_1np = lambda mi,M0,M1 :-((1/2)*(-log(M1**2))-f01sum(mi,M0,M1)+ (1/2)*f11sum(mi,M0,M1)) 
b2_1np = lambda mj,M0,M2: (1/2)*(-log(M2**2))-f02sum(mj,M0,M2) + (1/2)*f12sum(mj,M0,M2)

#########################################################################
# Numpy definitions of PaVe functions
a1 = 1j/(16*pi**2)
a2 = -1j/pi**2

A0 = lambda ma,M:  M**2*(1+log((ma**2)/(M**2)))
B1_0 = lambda mi,M0,M1: b1_0np(mi,M0,M1)
B2_0 = lambda mj,M0,M2: b2_0np(mj,M0,M2)
B1_1 = lambda mi,M0,M1: b1_1np(mi,M0,M1)
B2_1 = lambda mj,M0,M2: b2_1np(mj,M0,M2)

#from numpy import log1p # log1p(x) = log(1+x) #####IMPORTANTE
def B12_0(ma,M1,M2):
    xi1,xi2 = x1(ma,M1,M2),x2(ma,M1,M2)###gk(i,ma,M1,M2)
    return (log((ma**2)/(M1**2),dtype=complex256)/2.0 + xi1*log((xi1- 1.0)/xi1,dtype=complex256) + xi2*log((xi2- 1.0)/xi2,dtype=complex256))

def B12_1(ma,M1,M2):
    return ((1/(2*ma**2))*(M1**2*(1+log(ma**2/M1**2)) - M2**2*(1+log(ma**2/M2**2))) + B12_0(ma,M1,M2)/(2*ma**2)*(M2**2-M1**2 + ma**2))

def B12_2(ma,M1,M2):
    return ((1/(2*ma**2))*(M1**2*(1+log(ma**2/M1**2)) - M2**2*(1+log(ma**2/M2**2))) + B12_0(ma,M1,M2)/(2*ma**2)*(M2**2-M1**2 - ma**2))

def C0(ma,M0,M1,M2):
    y0 = x0(ma,M0,M2)
    y1 = x1(ma,M1,M2)
    y2 = x2(ma,M1,M2)
    y3 = x3(M0,M1)
    return ((R0(y0,y1) + R0(y0,y2) - R0(y0,y3))/ma**2)

C1 = lambda ma,mi,M0,M1,M2: ((1/ma**2)*(B1_0(mi,M0,M1) - B12_0(ma,M1,M2) + (M2**2-M0**2)*C0(ma,M0,M1,M2)))
C2 = lambda ma,mj,M0,M1,M2:( (-1/ma**2)*(B2_0(mj,M0,M2) - B12_0(ma,M1,M2) + (M1**2-M0**2)*C0(ma,M0,M1,M2)))







