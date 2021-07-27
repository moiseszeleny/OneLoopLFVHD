#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16  01:27:53 2021

@author: Moises Zeleny
"""
from numpy import sqrt,log,conjugate,pi, where,array, vectorize,complex128, complex256,float64, float128
from numpy import nan_to_num
from numpy import abs as npabs

δ = 0
from scipy.special import spence
def sci_polylog(s,z):
    return spence(1-z,dtype=complex128)#
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


#def x1(ma,M1,M2):
#    B = -M1**2 + M2**2 + ma**2 
#    C = M1**4 - 2*M1**2*M2**2 - 2*M1**2*ma**2 + M2**2 - 2*M2**2*ma**2 + ma**4
#    return (B -sqrt(C,dtype=complex256))/(2*ma**2)
#def x2(ma,M1,M2):
#    B = -M1**2 + M2**2 + ma**2 
#    C = M1**4 - 2*M1**2*M2**2 - 2*M1**2*ma**2 + M2**2 - 2*M2**2*ma**2 + ma**4
#    return (B + sqrt(C,dtype=complex256))/(2*ma**2)

from .roots import x1,x2, rootsx2_inv

#def xk(i,ma,M1,M2):
#    if i in [1,2]:
#        if i==1:
#            out = x1(ma,M1,M2)
#        else:
#            out = x2(ma,M1,M2)
#    else:
#        raise ValueError('i must be equals to 1 or 2.')

#def xk(i,ma,M1,M2): 
#    if i in [1,2]:
#        if M1 == M2 and npabs(M1)<1e-6:
#            if i==1:
#                out= (M1/ma)**2 + (M1/ma)**4
#            if i==2:
#                out = 1 - (M1/ma)**2 - (M1/ma)**4
#        else:
#            out = _solsnp(ma,M1,M2)[i-1]
#    else: 
#        raise ValueError('i must be equals to 1 or 2.')
#    return out

#xk = vectorize(xk)
## aproximation of g =1-1/xk for k =1,2

#def gk(i,ma,M1,M2): 
#    if i in [1,2]:
#        if M1 == M2 and npabs(M1)<1e-6:
#            if i==1:
#                out= - (ma/M1)**2 + 2 + (M1/ma)**2 + 2*(M1/ma)**4 
#            if i==2:
#                out = - (M1/ma)**2 - 2*(M1/ma)**4
#        else:
#            x = xk(i,ma,M1,M2)
#            out = (x- 1)/x
#    else: 
#        raise ValueError('i must be equals to 1 or 2.')
#    return out
        
#gk = vectorize(gk)

# C0 en terminos de R0
Li2 = lambda x0: sci_polylog(2,x0)
R0 = lambda x0,xi: Li2(x0/(x0-xi)) - Li2((x0-1.0)/(x0-xi))

#R0_aprox = lambda x0,xi: Li2_aprox(x0/(x0-xi)) - Li2_aprox((x0-1)/(x0-xi))
x0 = lambda ma,M0,M2: (M2**2-M0**2)/ma**2
def x3(ma,M0,M1): 
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
def f0np(y):
    down = (1-y)*log((y-1)/y,dtype=complex256)-1 
    X = 0
    for l in range(1,21):
        X += y**(-l)/(l+1)

    upper = log((y-1)/y,dtype=complex256) + X
    return where(npabs(y)<10, down,upper)

def f1np(y):
    down = (1-y**2)*log((y-1)/y)-(y + 0.5) 
    X = 0
    for l in range(2,22):
        X += y**(1-l)/(l+1)

    upper = log((y-1)/y,dtype=complex256) + X
    return where(npabs(y)<10, down,upper)

#soluciones de la ecuación
M0,M1,M2 = symbols('M_0,M_1,M_2',positive=True)
_mi,_mj = symbols('mi,mj',positive=True)

ec1 = y**2*_ma**2 - y*(_mi**2 + M1**2-M0**2) + M1**2
ec2 = y**2*_ma**2 - y*(_mj**2 + M2**2-M0**2) + M2**2
y11,y12 = solve(ec1,y)
y21,y22 = solve(ec2,y)

#Convirtiendo en funciones simbolicas las soluciones yij
#y11np = lambda ma,mi,M0,M1:(1.0/(2*ma**2))*(-M0**2 + M1**2 + mi**2- sqrt(-(-M0**2 + M1**2 + 2*M1*ma + mi**2)*(M0**2 -M1**2 + 2*M1*ma - mi**2),dtype=complex256))#lambdify([_ma,_mi,M0,M1],y11,'numpy')
#y12np = lambda ma,mi,M0,M1:(1.0/(2*ma**2))*(-M0**2 + M1**2 + mi**2 + sqrt(-(-M0**2 + M1**2 + 2*M1*ma + mi**2)*(M0**2 -M1**2 + 2*M1*ma - mi**2),dtype=complex256))#lambdify([_ma,_mi,M0,M1],y12,'numpy')
#y21np = lambda ma,mj,M0,M2:y11np(ma,mj,M0,M2) #lambdify([_ma,_mj,M0,M2],y21,'numpy')
#y22np = lambda ma,mj,M0,M2:y12np(ma,mj,M0,M2)#lambdify([_ma,_mj,M0,M2],y22,'numpy')

from .roots import y11 as y11np
from .roots import y12 as y12np
from .roots import y21 as y21np
from .roots import y22 as y22np



#def y1j(j,ma,mi,M0,M1):
#    if j ==1:
#        out = y11np(ma,mi,M0,M1)
#    elif j==2:
#        if npabs(M1)<1e-6:
#            out = M1**2*(-M0**2 - 2*ma**2 + mi**2 + sqrt(M0**4 - 2*M0**2*mi**2 + mi**4))/(2*ma**2*sqrt(M0**4 - 2*M0**2*mi**2 + mi**4))
        #elif npabs(M0)>1e6 and M1==80.379:
        #    out = 
#        else:
#            out = y12np(ma,mi,M0,M1)
#    else:
#        raise ValueError('i must be equals to 1 or 2.')
#    return out

#y1j = vectorize(y1j)

#def y2j(j,ma,mj,M0,M2):
#    if j ==1:
#        out = y21np(ma,mj,M0,M2)
#    elif j==2:
#        if npabs(M2)<1e-6:
#            out = M2**2*(-M0**2 - 2*ma**2 + mj**2 + sqrt(M0**4 - 2*M0**2*mj**2 + mj**4))/(2*ma**2*sqrt(M0**4 - 2*M0**2*mj**2 + mj**4))
#        else:
#            out = y22np(ma,mj,M0,M2)
#    else:
#        raise ValueError('i must be equals to 1 or 2.')
#    return out

#y2j = vectorize(y2j)

f01sum = lambda ma,mi,M0,M1:sum(f0np(yi1j(ma,mi,M0,M1)) for yi1j in [y11np,y12np ])
#f01sum

f02sum = lambda ma,mj,M0,M2:sum(f0np(yi2j(ma,mj,M0,M2)) for yi2j in [y21np,y22np ])
#f02sum

f11sum = lambda ma,mi,M0,M1:sum(f1np(yi1j(ma,mi,M0,M1)) for yi1j in [y11np, y12np ])
#f11sum

f12sum = lambda ma,mj,M0,M2:sum(f1np(yi2j(ma,mj,M0,M2)) for yi2j in [y21np, y22np])
#f12sum

b1_0np = lambda ma,mi,M0,M1:-log(M1**2)-f01sum(ma,mi,M0,M1)
b2_0np = lambda ma,mj,M0,M2: -log(M2**2)-f02sum(ma,mj,M0,M2)
b1_1np = lambda ma,mi,M0,M1 :-((1/2)*(-log(M1**2))-f01sum(ma,mi,M0,M1)+ (1/2)*f11sum(ma,mi,M0,M1)) 
b2_1np = lambda ma,mj,M0,M2: (1/2)*(-log(M2**2))-f02sum(ma,mj,M0,M2) + (1/2)*f12sum(ma,mj,M0,M2)

#########################################################################
# Numpy definitions of PaVe functions
a1 = 1j/(16*pi**2)
a2 = -1j/pi**2

A0 = lambda ma,M:  M**2*(1+log((ma**2)/(M**2)))
B1_0 = lambda ma,mi,M0,M1: b1_0np(ma,mi,M0,M1)
B2_0 = lambda ma,mj,M0,M2: b2_0np(ma,mj,M0,M2)
B1_1 = lambda ma,mi,M0,M1: b1_1np(ma,mi,M0,M1)
B2_1 = lambda ma,mj,M0,M2: b2_1np(ma,mj,M0,M2)

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
    y3 = x3(ma,M0,M1)
    return ((R0(y0,y1) + R0(y0,y2) - R0(y0,y3))/ma**2)

C1 = lambda ma,mi,M0,M1,M2: ((1/ma**2)*(B1_0(ma,mi,M0,M1) - B12_0(ma,M1,M2) + (M2**2-M0**2)*C0(ma,M0,M1,M2)))
C2 = lambda ma,mj,M0,M1,M2:( (-1/ma**2)*(B2_0(ma,mj,M0,M2) - B12_0(ma,M1,M2) + (M1**2-M0**2)*C0(ma,M0,M1,M2)))







