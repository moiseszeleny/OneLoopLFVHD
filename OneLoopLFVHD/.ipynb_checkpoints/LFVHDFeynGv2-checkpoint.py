#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 01:07:53 2020

@author: Moises Zeleny
"""
from sympy import Function,symbols,solve,polylog,I,simplify,pi
from sympy import sqrt,collect,Add,log,conjugate,re
#from sympy import init_printing
#init_printing()
##################################################################################################
# Masas del higgs tipo SM, m1 y m2 masas de los leptones finales y delta parámetro infinitesimal para las integrales de Passarivo Veltman
##################################################################################################
m0 = input('Initial particle mass ma: ')
m1 = input('Final particle mass mi: ')
m2 = input('Final particle mass mj: ')

ma = symbols(fr'{m0}',positive=True)
mi = symbols(fr'{m1}',positive=True)
mj = symbols(fr'{m2}',positive=True)

δ = 0

from scipy.special import spence
def sci_polylog(s,z):
    return spence(1-z)
##################################################################################################
# Funciones de Passarino Veltman pertinentes para LFVHD
##################################################################################################

# Funciones de PaVe escalares
A0 = Function('A_0')
B0 = Function('B_0')
B1_0 = Function('{{B^{(1)}_{0}}}')
B2_0 = Function('{{B^{(2)}_{0}}}')
B1_1 = Function('{{B^{(1)}_{1}}}')
B2_1 = Function('{{B^{(2)}_{1}}}')
B12_0 = Function('{{B^{(12)}_{0}}}')
B12_1 = Function('{{B^{(12)}_{1}}}')
B12_2 = Function('{{B^{(12)}_{2}}}')
C0 = Function('C_0')
C1 = Function('C_1')
C2 = Function('C_2')


# Partes finitas de las funciones de PaVe en la aproximación p_i^2 = 0
a0 = Function('a_0')
b1_0 = Function('{{b^{(1)}_0}}')
b2_0 = Function('{{b^{(2)}_0}}')
b1_1 = Function('{{b^{(1)}_1}}')
b2_1 = Function('{{b^{(2)}_1}}')
b12_0 = Function('{{b^{(12)}_0}}')
b12_1 = Function('{{b^{(12)}_1}}')
b12_2 = Function('{{b^{(12)}_2}}')

# Función simbolica para las divergencias
#Div=Function('Div')

# Definiendo divergencias
Δe = symbols(r'\Delta_\epsilon')
class Div(Function):
    '''Subclass of sympy Function which give the associated divergence of the PaVe functions predefined.
    
    Atributes
    ---------
    This has the same atributtes as Function of sympy
    
    Methods
    -------
    eval(F)
        F: PaVe function
        Return the divergence of the PaVe functions predefined
    
    Example
    -------
    >>> from sympy import symbols
    >>> m = symbols('m',rel=True)
    >>> Div(A0(m))
    m**2*Δe    
    '''
    @classmethod
    def eval(cls, F):
        if F.func==A0:
            M = F.args[0]
            return M**2*Δe
        elif F.func==B1_0 or F.func==B2_0 or F.func==B12_0:
            return Δe
        elif F.func==B1_1 or F.func==B12_1:
            #M0,M1 = F.args
            return Δe/2
        elif F.func==B2_1 or F.func==B12_2:
            #M0,M2 = F.args
            return -(Δe/2)

# Cambios PaVe en términos de funciones divergentes y finitas
class PaVetoDivFin(Function):
    '''Subclass of sympy Function to rewrite PaVe functions in terms of the finite and divergent part of the corresponding PaVe
    
    Atributes
    ---------
    This has the same atributtes as Function of sympy
    
    Methods
    -------
    eval(F)
        F: PaVe Function
        Return the PaVe function in term of the finite and divergent parts
    
    Example
    -------
    >>> from sympy import symbols
    >>> m = symbols('m',rel=True)
    >>> PaVetoDivFin(A0(m))
    a0(m) + m**2*Δe    
    '''
    @classmethod
    def eval(cls, F):
        if F.func==A0:
            args = F.args
            return Div(A0(*args)) + a0(*args)
        elif F.func==B1_0:
            args = F.args
            return Div(B1_0(*args)) + b1_0(*args)
        elif F.func==B2_0:
            args = F.args
            return Div(B2_0(*args)) + b2_0(*args)
        elif F.func==B12_0:
            args = F.args
            return Div(B12_0(*args)) + b12_0(*args)
        elif F.func==B1_1:
            args = F.args
            return Div(B1_1(*args)) + b1_1(*args)
        elif F.func==B2_1:
            args = F.args
            return Div(B2_1(*args)) + b2_1(*args)
        elif F.func==C1:
            #args = F.args
            M0,M1,M2 = F.args
            return (1/ma**2)*(b1_0(M0,M1) - b12_0(M1,M2) + (M2**2-M0**2)*C0(M0,M1,M2))
        elif F.func==C2:
            #args = F.args
            M0,M1,M2 = F.args
            return (-1/ma**2)*(b2_0(M0,M2) - b12_0(M1,M2) + (M1**2-M0**2)*C0(M0,M1,M2))
        else:
            raise ValueError(f'{F.func} is not defined.')
# definición de xk
_x,_m1,_m2 = symbols('x,m1,m2')
_funcion = _x**2 - (ma**2 -_m1**2 + _m2**2)/ma**2*_x + (_m2**2-I*δ)/ma**2
_sols = solve(_funcion,_x)
#_sols
def xk(i,M1,M2):
    if i==1:
        return _sols[0].subs({_m1:M1,_m2:M2})
    elif i==2:
        return _sols[1].subs({_m1:M1,_m2:M2})
    else:
        return 'i = a 1 o 2'

# C0 en terminos de R0
Li2 = lambda x0: polylog(2,x0)
R0 = lambda x0,xi: Li2(x0/(x0-xi)) - Li2((x0-1)/(x0-xi))

#R0_aprox = lambda x0,xi: Li2_aprox(x0/(x0-xi)) - Li2_aprox((x0-1)/(x0-xi))
x0 = lambda M0,M2: (M2**2-M0**2)/ma**2
x3 = lambda M0,M1: (-M0**2+I*δ)/(M1**2-M0**2)
# definiciones para las partes finitas de las PaVe
class PaVe_aprox(Function):
    '''Subclass of sympy Function to show explicitly the  definition of finite 
    part of PaVe functions.
    Reference
    ---------
    This definition are given inthe approximation given by
    https://arxiv.org/abs/1512.03266v2
    
    Atributes
    ---------
    This has the same atributtes as Function of sympy
    
    Methods
    -------
    eval(F)
        F: Finite part of PaVe Function
        Return explicitly definition of finite part of PaVe functions
    
    Example
    -------
    >>> from sympy import symbols
    >>> m = symbols('m',rel=True)
    >>> PaVe_aprox(a0(m))
    m**2*(1+log((ma**2-I*δ)/(m**2-I*δ)))
    '''
    @classmethod
    def eval(cls, F):
        if F.func==a0:
            M = F.args[0]
            return M**2*(1+log((ma**2-I*δ)/(M**2-I*δ)))
        elif F.func==b1_0:
            M0,M1 = F.args
            return 1-log(M1**2/ma**2) + (M0**2)/(M0**2-M1**2)*log(M1**2/M0**2)
        elif F.func==b2_0:
            M0,M2 = F.args
            return 1-log(M2**2/ma**2) + (M0**2)/(M0**2-M2**2)*log(M2**2/M0**2)
        elif F.func==b1_1:
            M0,M1 = F.args
            return -log(M1**2/ma**2)/2 + (M0**4)/(2*(M0**2-M1**2)**2)*log(M0**2/M1**2) + ((M0**2-M1**2)*(3*M0**2-M1**2))/(4*(M0**2-M1**2)**2)
        elif F.func==b2_1:
            M0,M2 = F.args
            return log(M2**2/ma**2)/2 + (M0**4)/(2*(M0**2-M2**2)**2)*log(M0**2/M2**2) - ((M0**2-M2**2)*(3*M0**2-M2**2))/(4*(M0**2-M2**2)**2)
        elif F.func==b12_0:
            M1,M2 = F.args
            x1,x2 = xk(1,M1,M2),xk(2,M1,M2)
            return log((ma**2 - I*δ)/(M1**2 - I*δ))/2 + sum(x*log(1-1/x) for x in [x1,x2])
        elif F.func==b12_1:
            M1,M2 = F.args
            return (1/(2*ma**2))*(M1**2*(1+log(ma**2/M1**2)) - M2**2*(1+log(ma**2/M2**2))) + b12_0(M1,M2)/(2*ma**2)*(M2**2-M1**2 + ma**2)
        elif F.func==b12_2:
            M1,M2 = F.args
            return (1/(2*ma**2))*(M1**2*(1+log(ma**2/M1**2)) - M2**2*(1+log(ma**2/M2**2))) + b12_0(M1,M2)/(2*ma**2)*(M2**2-M1**2 - ma**2)
        elif F.func==C0:
            M0,M1,M2 = F.args
            y0 = x0(M0,M2)
            y1 = xk(1,M1,M2)
            y2 = xk(2,M1,M2)
            y3 = x0(M0,M1)
            if M1==M2:
                if M1==0:
                    return (pi**2 + 3*log((-2*M0**2)/ma**2)**2 + 6*polylog(2,1 + (2*M0**2)/ma**2))/(3.*ma**2)
                else:
                    return (-log((-4*M0**2 + 4*M1**2 - ma**2 + sqrt(-8*M1**2*ma**2 + ma**4))/(-ma**2 + sqrt(-8*M1**2*ma**2 + ma**4)))**2 + 
                    -    log((-4*M0**2 + 4*M1**2 - ma**2 + sqrt(-8*M1**2*ma**2 + ma**4))/(ma**2 + sqrt(-8*M1**2*ma**2 + ma**4)))**2 - 
                    -    2*polylog(2,(2*(M0**2 - M1**2)**2)/(2*(M0**2 - M1**2)**2 + M0**2*ma**2)) + 
                    -    2*polylog(2,((M0 - M1)*(M0 + M1)*(2*M0**2 - 2*M1**2 + ma**2))/(2*(M0**2 - M1**2)**2 + M0**2*ma**2)) - 
                    -    2*polylog(2,(4*(M0 - M1)*(M0 + M1))/(-ma**2 + sqrt(-8*M1**2*ma**2 + ma**4))) + 
                    -    2*polylog(2,(2*(2*M0**2 - 2*M1**2 + ma**2))/(ma**2 + sqrt(-8*M1**2*ma**2 + ma**4))) + 
                    -    2*polylog(2,(4*(M0 - M1)*(M0 + M1))/(4*M0**2 - 4*M1**2 + ma**2 + sqrt(-8*M1**2*ma**2 + ma**4))) - 
                    -    2*polylog(2,(2*(2*M0**2 - 2*M1**2 + ma**2))/(4*M0**2 - 4*M1**2 + ma**2 + sqrt(-8*M1**2*ma**2 + ma**4))))/ma**2
            else:
                if M1==0:
                    return (log((-2*M0**2)/(-2*M2**2 + ma**2))**2 + 2*polylog(2,1 - M2**2/M0**2) + 
                    2*polylog(2,1 + (2*M0**2)/(-2*M2**2 + ma**2)))/ma**2
                elif M2==0:
                    return (log((-2*M0**2)/(-2*M1**2 + ma**2))**2 + 2*polylog(2,1 - M1**2/M0**2) + 
                    2*polylog(2,1 + (2*M0**2)/(-2*M1**2 + ma**2)))/ma**2
                else:
                    return (1/ma**2)*(R0(y0,y1) + R0(y0,y2) - R0(y0,y3))
                
        else:
            raise ValueError(f'{F.func} is not defined.')

##################Funciones para sustituciones de las funciones de PaVe
#Para triangulos
FuncPaVe = lambda M0,M1,M2:[A0(M0),A0(M1),A0(M2),B1_0(M0,M1),B2_0(M0,M2),B1_1(M0,M1),B2_1(M0,M2),B12_0(M1,M2),C1(M0,M1,M2),C2(M0,M1,M2)]
funcPaVe = lambda M0,M1,M2:[a0(M0),a0(M1),a0(M2),b1_0(M0,M1),b2_0(M0,M2),b1_1(M0,M1),b2_1(M0,M2),b12_0(M1,M2),C0(M0,M1,M2)]
cambiosDivFin = lambda M0,M1,M2:{PV:PaVetoDivFin(PV) for PV in FuncPaVe(M0,M1,M2)}
cambios_aprox = lambda M0,M1,M2:{PV:PaVe_aprox(PV) for PV in funcPaVe(M0,M1,M2)}

# Para burbujas
#FuncPaVeBA = lambda M0,M1,M2:[A0(M0),A0(M1),A0(M2),B1_0(M0,M1),B2_0(M0,M2),B1_1(M0,M1),B2_1(M0,M2),B12_0(M1,M2)]
#funcPaVeBA = lambda M0,M1,M2:[a0(M0),a0(M1),a0(M2),b1_0(M0,M1),b2_0(M0,M2),b1_1(M0,M1),b2_1(M0,M2),b12_0(M1,M2)]
#cambiosDivFinBA = lambda M0,M1:{PV:PaVetoDivFinBA(PV) for PV in FuncPaVeBA(M0,M1,M2)}
#cambios_aproxBA = lambda M0,M1:{PV:PaVe_aproxBA(PV) for PV in funcPaVeBA(M0,M1,M2)}


##########################################################################################################3
# Funciones H para las diferentes contribuciones al proceso h->l1,l2
##########################################################################################################3
# Funciones para las contribuciones del tipo FSV
#M0,M1,M2 = symbols('M0,M1,M2',real=True)
HFSV = [0]*9
HFSV[1] = lambda M0,M1,M2: B12_0(M1,M2) + M0**2*C0(M0,M1,M2) + (mj**2- 2*ma**2)*C2(M0,M1,M2) + 2*mi**2*(-C1(M0,M1,M2) + C2(M0,M1,M2))
HFSV[2] = lambda M0,M1,M2: mi*mj*(C1(M0,M1,M2) - 2*C2(M0,M1,M2))
HFSV[3] = lambda M0,M1,M2: mi*M0*(C1(M0,M1,M2) - 2*C0(M0,M1,M2))
HFSV[4] = lambda M0,M1,M2: mj*M0*(C0(M0,M1,M2) + C2(M0,M1,M2))
##
# Funciones para las contribuciones del tipo FVS
##
HFVS = [0]*9
#HFVS
HFVS[1] = lambda M0,M1,M2: mi*M0*(C0(M0,M1,M2) + C1(M0,M1,M2))
HFVS[2] =  lambda M0,M1,M2:-mj*M0*(2*C0(M0,M1,M2) + C2(M0,M1,M2))
HFVS[3] = lambda M0,M1,M2: B12_0(M1,M2) + M0**2*C0(M0,M1,M2) + (2*ma**2 -mi**2-2*mj**2)*C1(M0,M1,M2) + 2*mj**2*C2(M0,M1,M2)
HFVS[4] = lambda M0,M1,M2: -mi*mj*(-2*C1(M0,M1,M2) + C2(M0,M1,M2))
##
###############################################3
# Funciones para las contribuciones del tipo FVV
##################################################
XRL = lambda M0,M1,M2: M0**2*(B1_1(M0,M1) - B1_0(M0,M1)) - (A0(M2) + M0**2*B2_0(M0,M2) + mj**2*B2_1(M0,M2))  +\
                       (M1**2 + M2**2 - ma**2)*(B12_1(M1,M2) + M0**2*C1(M0,M1,M2) - B12_0(M1,M2) - M0**2*C0(M0,M1,M2) - mj**2*C2(M0,M1,M2))
#XRL(M0,M1,M2)

XLR = lambda M0,M1,M2: 2*A0(M2) + M0**2*(B2_1(M0,M2) + B2_0(M0,M2)) + A0(M1) + M0**2*B1_0(M0,M1) - mi**2*B1_1(M0,M1)  +\
                       (M1**2 + M2**2 - ma**2)*(B12_2(M1,M2) + M0**2*C2(M0,M1,M2) + B12_0(M1,M2) + M0**2*C0(M0,M1,M2) - mi**2*C1(M0,M1,M2))
##############################################
D = symbols('D')
HFVV = [0]*9
#HFVV
HFVV[1] = lambda M0,M1,M2:-mi*(D-2)*C1(M0,M1,M2)
#HFVV[1](M0,M1,M2)
HFVV[2] =  lambda M0,M1,M2: (D-2)*mj*C2(M0,M1,M2)
#HFVV[2](M0,M1,M2)

HFVV[3] = lambda M0,M1,M2: M0*D*C0(M0,M1,M2)
#HFVV[4](M0,M1,M2)
################################################
# Funciones para las contribuciones del tipo SFF
###############################################
#HSFF = [0]*5
#HSFF[1] = lambda M0,M1,M2: mi*M2*(C1(M0,M1,M2)-C0(M0,M1,M2))
#HSFF[2] =  lambda M0,M1,M2: M2*(M1*C0(M0,M1,M2) - mj*C2(M0,M1,M2))
#HSFF[3] =  lambda M0,M1,M2: B12_0(M1,M2) + M0**2*C0(M0,M1,M2) - mi**2*C1(M0,M1,M2) + mj**2*C2(M0,M1,M2) + (ma**2 -mi**2 -mj**2)*(C1(M0,M1,M2) - C2(M0,M1,M2) - C0(M0,M1,M2) ) - mj*M1*(C0(M0,M1,M2)  + C2(M0,M1,M2))           
#HSFF[4] =  lambda M0,M1,M2: mi*M1*C1(M0,M1,M2) + mi*mj*(C1(M0,M1,M2) - C2(M0,M1,M2) - C0(M0,M1,M2) )

HSFF = [0]*9
HSFF[1] = lambda M0,M1,M2: B12_0(M1,M2) + M0**2*C0(M0,M1,M2) -mi**2*C1(M0,M1,M2) + mj**2*C2(M0,M1,M2)
HSFF[2] = lambda M0,M1,M2: mi*mj*(C0(M0,M1,M2) + C2(M0,M1,M2) -C1(M0,M1,M2))
HSFF[3] = lambda M0,M1,M2: mj*M2*C2(M0,M1,M2)
HSFF[4] = lambda M0,M1,M2: mi*M2*(C0(M0,M1,M2)-C1(M0,M1,M2))
HSFF[5] = lambda M0,M1,M2: mj*M1*(C0(M0,M1,M2) + C2(M0,M1,M2))
HSFF[6] = lambda M0,M1,M2: -mi*M1*C1(M0,M1,M2)
HSFF[7] = lambda M0,M1,M2: M1*M2*C0(M0,M1,M2)

#################################################
# Funciones para las contribuciones del tipo VFF
#################################################
#HVFF = [0]*5

#SLR1 = lambda M0,M1,M2: B1_0(M0,M1) + (M2**2-mj**2)*C0(M0,M1,M2) - mi**2*C1(M0,M1,M2) - mj**2*C2(M0,M1,M2) - mj*M1*(C0(M0,M1,M2) + C2(M0,M1,M2)) - (ma**2 -mi**2 -mj**2)*(C0(M0,M1,M2)  + C2(M0,M1,M2)) 

#SRL1 = lambda M0,M1,M2: mi*M1*C1(M0,M1,M2) - mi*mj*(C0(M0,M1,M2) - C1(M0,M1,M2) + C2(M0,M1,M2))

#TLR1 = lambda M0,M1,M2: mi*M2*(C1(M0,M1,M2) - C0(M0,M1,M2) )
    
#TRL1 = lambda M0,M1,M2: M2*(M1*C0(M0,M1,M2) - mj*C2(M0,M1,M2))


#HVFF[1] = lambda M0,M1,M2: -D*SLR1(M0,M1,M2)
#HVFF[2] =  lambda M0,M1,M2: -D*SRL1(M0,M1,M2)
#HVFF[3] = lambda M0,M1,M2: -D*TLR1(M0,M1,M2)
#HVFF[4] =  lambda M0,M1,M2: -D*TRL1(M0,M1,M2)

HVFF = [0]*9
HVFF[1] = lambda M0,M1,M2: -(2-D)*mi*M2*(C0(M0,M1,M2)-C1(M0,M1,M2))
HVFF[2] = lambda M0,M1,M2:-(2-D)*mj*M2*C2(M0,M1,M2)
HVFF[3] = lambda M0,M1,M2:(D-4)*mi*mj*(C1(M0,M1,M2)-C0(M0,M1,M2)-C2(M0,M1,M2))
HVFF[4] = lambda M0,M1,M2:-(D*(B12_0(M0,M1) + M0**2*C0(M0,M1,M2) +mi**2*C2(M0,M1,M2) - mi**2*C1(M0,M1,M2)) + 4*((ma**2 -mi**2 - mj**2)/2)*(C1(M0,M1,M2)-C0(M0,M1,M2)-C2(M0,M1,M2)))
HVFF[5] = lambda M0,M1,M2:(2-D)*mi*M1*C1(M0,M1,M2)
HVFF[6] = lambda M0,M1,M2:-(2-D)*mj*M1*(C0(M0,M1,M2) + C2(M0,M1,M2))
HVFF[7] = lambda M0,M1,M2:-D*M1*M2*C0(M0,M1,M2)

#########################################
#####################################################################################################3
#Definiendo las clases de los vertices genericos para este tipo de procesos
#####################################################################################################3
class VertexHSS():
    '''Class of vertex of three scalars
    
    Atributes
    ---------
        c:sympy symbols
            c represents the constant coupling among three scalars
    
    Methods
    -------
    show()
    returns three scalars coupling
    '''
    def __init__(self,c):
        '''
        Parameters
        ----------
            c:sympy symbols
                c represents the constant coupling among three scalars
        '''
        self.c = c
        
    def __str__(self):
        return f'VertexHSS({self.c!r})'
    
    def __repr__(self):
        return self.__str__()
    def show(self):
        '''Returns three scalars coupling'''
        return self.c

class VertexHFF(VertexHSS):
    '''Class of vertex of neutral scalar and two fermions.
    
    Atributes
    ---------
        c:sympy symbols
        c represents the constant coupling among neutral scalar and two fermions
    
    Methods
    -------
    show()
        returns neutral scalar and two fermions coupling
    '''
    #def __init__(self,c):
    #    self.c = c
    def __str__(self):
        return f'VertexHFF({self.c!r})'
    
    def show(self):
        '''
        Returns Higgs boson and two fermions coupling
        '''
        return self.c
    

class VertexHSV(VertexHSS):
    '''Class of vertex of two scalars and one vector boson
    
    Atributes
    ---------
        c:sympy symbols
        c represents the constant coupling among two scalars and one vector boson
    
    Methods
    -------
    show()
        returns two scalars and one vector boson coupling
    '''
    #def __init__(self,c):
    #    self.c = c
    def __str__(self):
        return f'VertexHSV({self.c!r})'
    
    def show(self):
        '''Returns two scalars and one vector boson coupling'''
        pmu1,pmu2 = symbols(r'{{p^{\mu}_1}},{{p^{\mu}_2}}')
        return self.c*(pmu1 - pmu2)

class VertexHVV(VertexHSS):
    '''Class of vertex of one scalar and two vector bosons.
    
    Atributes
    ---------
        c:sympy symbols
        c represents the constant coupling among one scalar and two vector bosons
    
    Methods
    -------
    show()
        returns one scalar and two vector bosons coupling
    '''
    def __str__(self):
        return f'VertexHVV({self.c!r})'
    
    def show(self):
        '''Returns one scalar and two vector bosons coupling'''
        gmunu = symbols(r'{{g^{\mu \nu}}}',real=True)
        return self.c*gmunu

class VertexSFF():
    '''Class vertex of one scalar and two fermions.
    
    Atributes
    ---------
        cR:sympy symbols
            cR is the coefficint of PR in the coupling of one scalar and two fermions
        cL:sympy symbols
            cL is the coefficint of PL in the coupling of one scalar and two fermions
    
    Methods
    -------
    show()
        returns one scalar and two fermions
    
    '''
    def __init__(self,cR,cL):
        '''
        Parameters
        ---------
        cR:sympy symbols
            cR is the coefficint of PR in the coupling of one scalar and two fermions
        cL:sympy symbols
            cL is the coefficint of PL in the coupling of one scalar and two fermions
        '''
        self.cR = cR
        self.cL = cL
        
    def __str__(self):
        return f'VertexSFF({self.cR!r},{self.cL!r})'
    
    def __repr__(self):
        return self.__str__()
    
    def show(self):
        '''Returns one scalar and two fermions'''
        PR,PL = symbols('P_R,P_L',commutative=False)
        cR = self.cR
        cL = self.cL
        return cR*PR + cL*PL

class VertexHF0F0(VertexSFF):
    def __str__(self):
        return f'VertexHF0F0({self.cR!r},{self.cL!r})'

class VertexVFF(VertexSFF):
    '''Class vertex of one vector boson and two fermions.
    
    Atributes
    ---------
        cR:sympy symbols
            cR is the coefficint of PR in the coupling of one vector boson 
            and two fermions
        cL:sympy symbols
            cL is the coefficint of PL in the coupling of one vector boson
            and two fermions
    
    Methods__repr__
    -------
    show()
        returns one vector boson and two fermions
    
    '''
    #def __init__(self,cR,cL):
    #    self.cR = cR
    #    self.cL = cL
    
    def __str__(self):
        return f'VertexVFF({self.cR!r},{self.cL!r})'
    def show(self):
        '''Returns one vector boson and two fermions'''
        PR,PL,gamma_mu = symbols(r'P_R,P_L,\gamma_\mu',commutative=False)
        cR = self.cR
        cL = self.cL
        return gamma_mu*(cR*PR + cL*PL)


#####################################################################################################3
#Definiendo las clases para los diferentes diagramas tipo triangulo
#####################################################################################################3

class Triangle():
    '''Represents a trinangle Feynman diagram
    
    Atributes
    ---------
        v1,v2,v3:some of the classes VertexSSS,VertexVSS,VertexSVV,
        VertexSFF,VertexSVV or VertexhFF.
            
    Methods
    -------
    show()
        returns 
    
    '''
    def __init__(self,v1,v2,v3,masas):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.masas = masas
        self.vertices = [v1,v2,v3]
        self.Cs = [C0(*masas),C1(*masas),C2(*masas)]
        
    def __str__(self):
        return f'Triangle({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def __repr__(self):
        return self.__str__()

    def AL(self):
        return symbols('M_L')
    def AR(self):
        return symbols('M_R')
    
    def AL_GIM(self):
        M0,M1,M2 = self.masas
        R = self.AL().subs(cambiosDivFin(M0,M1,M2)).expand()
        return collect(Add(*[r for r in R.args if r.has(M0.simplify())]),self.Cs,simplify).simplify()
    
    def AR_GIM(self):
        M0,M1,M2 = self.masas
        R = self.AR().subs(cambiosDivFin(M0,M1,M2)).expand()
        return collect(Add(*[r for r in R.args if r.has(M0.simplify())]),self.Cs,simplify).simplify()

    #def ML_GIM(self,funcion=simplify):
    #    M0,M1,M2 = self.masas
    #    polyM0 = collect(self.ML().subs(cambiosDivFin(M0,M1,M2)).expand(),[M0],evaluate=False)
    #    pot = list(polyM0.keys())
    #    if 1 in pot:
    #        pot.remove(1)
    #    else:
    #        pass
    #    if len(pot)>=1:
    #        return collect(Add(*[pt*funcion(polyM0[pt]) for pt in pot]),self.Cs,simplify)
    #    else:
    #        print('El factor ML de este diagrama no contribuye por el mecanismo de GIM')
    #def MR_GIM(self,funcion=simplify):
    #    M0,M1,M2 = self.masas
    #    polyM0 = collect(self.MR().subs(cambiosDivFin(M0,M1,M2)).expand(),[M0],evaluate=False)
    #    pot = list(polyM0.keys())
    #    if 1 in pot:
    #        pot.remove(1)
    #    else:
    #        pass
    #    if len(pot)>=1:
    #        return collect(Add(*[pt*funcion(polyM0[pt]) for pt in pot]),self.Cs,simplify)
    #    else:
    #        print('El factor MR de este diagrama no contribuye por el mecanismo de GIM')





#    def MLGIMDivFin(self):
#        M0,M1,M2 = self.masas
#        return self.ML_GIM().subs(cambiosDivFin(M0,M1,M2))
#    def MRGIMDivFin(self):
#        M0,M1,M2 = self.masas
#        return self.MR_GIM().subs(cambiosDivFin(M0,M1,M2))

    def Div_AL(self,M):
        M0,M1,M2 = self.masas
        AL = collect(self.AL().subs(cambiosDivFin(*self.masas)).expand(),[M],evaluate=False)
        #display(list(AL.keys()))
        Lista = []
        for m in AL.keys():
            dicΔe = collect(AL[m],[Δe],evaluate=False)
            claves = dicΔe.keys()
            if Δe in claves:
                Lista.append((dicΔe[Δe]*m).simplify()*Δe)
            else:
                Lista.append(m)
        return Lista

    def Div_MR(self,M):
        M0,M1,M2 = self.masas
        MR = collect(self.MR().subs(cambiosDivFin(*self.masas)).expand(),[M],evaluate=False)
        #display(list(MR.keys()))
        Lista = []
        for m in MR.keys():
            dicΔe = collect(MR[m],[Δe],evaluate=False)
            claves = dicΔe.keys()
            if Δe in claves:
                Lista.append((dicΔe[Δe]*m).simplify()*Δe)
            else:
                Lista.append(m)
        return Lista

    def width_hl1l2(self):
        ML,MR = self.ML(), self.MR()
        return 1/(8 *pi* ma)*sqrt((1-((mi**2+mj**2)/ma)**2)*(1-((mi**2-mj**2)/ma)**2))*((ma**2 - mi**2 - mj**2)*(abs(ML)**2 + abs(MR)**2)-4*mi*mj*re(ML*conjugate(MR)))


FactorRD = I/(16*pi**2)
class TriangleFSS(Triangle):
    def __str__(self):
        return f'TriangleFSS({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*v1.c*(mi*v2.cL*v3.cR*C1(M0,M1,M2) - mj*v2.cR*v3.cL*C2(M0,M1,M2) + M0*v2.cL*v3.cL*C0(M0,M1,M2))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*v1.c*(mi*v2.cR*v3.cL*C1(M0,M1,M2) - mj*v2.cL*v3.cR*C2(M0,M1,M2) + M0*v2.cR*v3.cR*C0(M0,M1,M2))

class TriangleFSV(Triangle):
    def __str__(self):
        return f'TriangleFSV({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return-FactorRD*v1.c*(v2.cL*v3.cL*(HFSV[1](M0,M1,M2)) + v2.cR*v3.cR*(HFSV[2](M0,M1,M2)) + v2.cL*v3.cR*(HFSV[3](M0,M1,M2)) + v2.cR*v3.cL*(HFSV[4](M0,M1,M2)))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*v1.c*(v2.cR*v3.cR*(HFSV[1](M0,M1,M2)) + v2.cL*v3.cL*(HFSV[2](M0,M1,M2)) + v2.cR*v3.cL*(HFSV[3](M0,M1,M2)) + v2.cL*v3.cR*(HFSV[4](M0,M1,M2)))

class TriangleFVS(Triangle):
    def __str__(self):
        return f'TriangleFVS({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return -FactorRD*v1.c*(v2.cL*v3.cL*(HFVS[1](M0,M1,M2)) + v2.cR*v3.cR*(HFVS[2](M0,M1,M2)) + v2.cL*v3.cR*(HFVS[3](M0,M1,M2)) + v2.cR*v3.cL*(HFVS[4](M0,M1,M2)))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return -FactorRD*v1.c*(v2.cR*v3.cR*(HFVS[1](M0,M1,M2)) + v2.cL*v3.cL*(HFVS[2](M0,M1,M2)) + v2.cR*v3.cL*(HFVS[3](M0,M1,M2)) + v2.cL*v3.cR*(HFVS[4](M0,M1,M2)))

class TriangleFVV(Triangle):
    def __str__(self):
        return f'TriangleFVV({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*v1.c*(v2.cL*v3.cL*(HFVV[1](M0,M1,M2)) + v2.cR*v3.cR*(HFVV[2](M0,M1,M2)) + v2.cL*v3.cR*(HFVV[3](M0,M1,M2)))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*v1.c*(v2.cR*v3.cR*(HFVV[1](M0,M1,M2)) + v2.cL*v3.cL*(HFVV[2](M0,M1,M2)) + v2.cR*v3.cL*(HFVV[3](M0,M1,M2)) )

class TriangleSFF(Triangle):
    def __str__(self):
        return f'TriangleSFF({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1, M2 = self.masas
        return FactorRD*(v1.cR*v2.cL*v3.cL*HSFF[1](M0,M1,M2) + v1.cL*v2.cR*v3.cR*HSFF[2](M0,M1,M2) + 
      v1.cR*v2.cR*v3.cL*HSFF[3](M0,M1,M2) + v1.cL*v2.cL*v3.cR*HSFF[4](M0,M1,M2) +
     v1.cL*v2.cR*v3.cL*HSFF[5](M0,M1,M2) + v1.cR*v2.cL*v3.cR*HSFF[6](M0,M1,M2) +
     v1.cL*v2.cL*v3.cL*HSFF[7](M0,M1,M2))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*(v1.cL*v2.cR*v3.cR*HSFF[1](M0,M1,M2) + v1.cR*v2.cL*v3.cL*HSFF[2](M0,M1,M2) + 
      v1.cL*v2.cL*v3.cR*HSFF[3](M0,M1,M2) + v1.cR*v2.cR*v3.cL*HSFF[4](M0,M1,M2) +
     v1.cR*v2.cL*v3.cR*HSFF[5](M0,M1,M2) + v1.cL*v2.cR*v3.cL*HSFF[6](M0,M1,M2) +
     v1.cR*v2.cR*v3.cR*HSFF[7](M0,M1,M2))

class TriangleVFF(Triangle):
    def __str__(self):
        return f'TriangleVFF({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*(v1.cR*v2.cL*v3.cL*HVFF[1](M0,M1,M2) + v1.cL*v2.cR*v3.cR*HVFF[2](M0,M1,M2) + 
      v1.cR*v2.cR*v3.cL*HVFF[3](M0,M1,M2) + v1.cL*v2.cL*v3.cR*HVFF[4](M0,M1,M2) +
     v1.cL*v2.cL*v3.cL*HVFF[5](M0,M1,M2) + v1.cR*v2.cR*v3.cR*HVFF[6](M0,M1,M2) +
     v1.cR*v2.cL*v3.cR*HVFF[7](M0,M1,M2))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1,M2 = self.masas
        return FactorRD*(v1.cL*v2.cR*v3.cR*HVFF[1](M0,M1,M2) + v1.cR*v2.cL*v3.cL*HVFF[2](M0,M1,M2) + 
      v1.cL*v2.cL*v3.cR*HVFF[3](M0,M1,M2) + v1.cR*v2.cR*v3.cL*HVFF[4](M0,M1,M2) +
     v1.cR*v2.cR*v3.cR*HVFF[5](M0,M1,M2) + v1.cL*v2.cL*v3.cL*HVFF[6](M0,M1,M2) +
     v1.cL*v2.cR*v3.cL*HVFF[7](M0,M1,M2))

#####################################################################################################3
#Definiendo las funciones H para los diferentes diagramas tipo burbuja
#####################################################################################################3
### Funciones para las correcciones de burbuja FV
HFV = [0]*5
HFV[1] = lambda M0,M1: ((mi*mj)/(mi**2 - mj**2))*(D-2)*B1_1(M0,M1)
HFV[2] = lambda M0,M1: ((mi**2)/(mi**2 - mj**2))*(D-2)*B1_1(M0,M1)
HFV[3] = lambda M0,M1: -((mj*M0)/(mi**2 - mj**2))*D*B1_0(M0,M1)
HFV[4] = lambda M0,M1: -((mi*M0)/(mi**2 - mj**2))*D*B1_0(M0,M1)

### Funciones para las correcciones de burbuja VF
HVF = [0]*5
FuncVF1 = lambda M0,M2: (D-2)*B2_1(M0,M2) + (1/M2**2)*(3*A0(M2) + 2*M0**2*B2_0(M0,M2) + (M0**2 + mj**2)*B2_1(M0,M2))
FuncVF2 = lambda M0,M2: (D-1)*B2_0(M0,M2) -A0(M0)/M2**2
HVF[1] = lambda M0,M2: -((mj**2)/(mj**2 - mi**2))*(D-2)*B2_1(M0,M2)
HVF[2] =  lambda M0,M2: -((mi*mj)/(mj**2 - mi**2))*(D-2)*B2_1(M0,M2)
HVF[3] = lambda M0,M2: -((mi*M0)/(mj**2 - mi**2))*D*B2_0(M0,M2)
HVF[4] = lambda M0,M2: -((mj*M0)/(mj**2 - mi**2))*D*B2_0(M0,M2)

### Funciones para las correcciones de burbuja FS
HFS = [0]*5
HFS[1] = lambda M0,M1: ((mj*M0)/(mi**2 - mj**2))*B1_0(M0,M1)
HFS[2] =  lambda M0,M1: ((mi*M0)/(mi**2 - mj**2))*B1_0(M0,M1)
HFS[3] = lambda M0,M1: ((mi*mj)/(mi**2 - mj**2))*B1_1(M0,M1)
HFS[4] = lambda M0,M1: ((mi**2)/(mi**2 - mj**2))*B1_1(M0,M1)

### Funciones para las correcciones de burbuja SF
HSF = [0]*5
HSF[1] = lambda M0,M2: ((mi*M0)/(mj**2 - mi**2))*B2_0(M0,M2)
HSF[2] =  lambda M0,M2: ((mj*M0)/(mj**2 - mi**2))*B2_0(M0,M2)
HSF[3] = lambda M0,M2: ((-mj**2)/(mj**2 - mi**2))*B2_1(M0,M2)
HSF[4] = lambda M0,M2: ((-mi*mj)/(mj**2 - mi**2))*B2_1(M0,M2)

#####################################################################################################3
#Definiendo las clases para los diferentes diagramas tipo burbuja
#####################################################################################################3
class Bubble():    
    def __init__(self,v1,v2,v3,masas):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.masas = masas
        self.vertices = [v1,v2,v3]
    
    def __str__(self):
        return f'Bubble({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def __repr__(self):
        return self.__str__()

    def AL(self):
        return symbols('A_L')
    def AR(self):
        return symbols('A_R')

#    def MLGIMDivFin(self):
#        M0,M1,M2 = self.masas
#        return self.ML_GIM().subs(cambiosDivFin(M0,M1,M2))
#    def MRGIMDivFin(self):
#        M0,M1,M2 = self.masas
#        return self.MR_GIM().subs(cambiosDivFin(M0,M1,M2))
    def width_hl1l2(self):
        ML,MR = self.AL(), self.AR()
        return 1/(8 *pi* ma)*sqrt((1-((mi**2+mj**2)/ma)**2)*(1-((mi**2-mj**2)/ma)**2))*((ma**2 - mi**2 - mj**2)*(abs(ML)**2 + abs(MR)**2)-4*mi*mj*re(ML*conjugate(MR)))

class BubbleFV(Bubble):
    def __str__(self):
        return f'BubbleFV({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1 = self.masas
        return FactorRD*v1.c*(v2.cL*v3.cL*(HFV[1](M0,M1)) + v2.cR*v3.cR*(HFV[2](M0,M1)) + v2.cL*v3.cR*(HFV[3](M0,M1)) + v2.cR*v3.cL*(HFV[4](M0,M1)))
    def AR(self):
        v1,v2, v3 = self.vertices
        M0,M1 = self.masas
        return FactorRD*v1.c*(v2.cR*v3.cR*(HFV[1](M0,M1)) + v2.cL*v3.cL*(HFV[2](M0,M1)) + v2.cR*v3.cL*(HFV[3](M0,M1)) + v2.cL*v3.cR*(HFV[4](M0,M1)))

class BubbleVF(Bubble):
    def __str__(self):
        return f'BubbleVF({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M2 = self.masas
        return FactorRD*v1.c*(v2.cL*v3.cL*(HVF[1](M0,M2)) + v2.cR*v3.cR*(HVF[2](M0,M2)) + v2.cL*v3.cR*(HVF[3](M0,M2)) + v2.cR*v3.cL*(HVF[4](M0,M2)))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M2 = self.masas
        return FactorRD*v1.c*(v2.cR*v3.cR*(HVF[1](M0,M2)) + v2.cL*v3.cL*(HVF[2](M0,M2)) + v2.cR*v3.cL*(HVF[3](M0,M2)) + v2.cL*v3.cR*(HVF[4](M0,M2)))

class BubbleFS(Bubble):
    def __str__(self):
        return f'BubbleFS({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M1 = self.masas
        return FactorRD*v1.c*(v2.cL*v3.cL*(HFS[1](M0,M1)) + v2.cR*v3.cR*(HFS[2](M0,M1)) + v2.cL*v3.cR*(HFS[3](M0,M1)) + v2.cR*v3.cL*(HFS[4](M0,M1)))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M1 = self.masas
        return FactorRD*v1.c*(v2.cR*v3.cR*(HFS[1](M0,M1)) + v2.cL*v3.cL*(HFS[2](M0,M1)) + v2.cR*v3.cL*(HFS[3](M0,M1)) + v2.cL*v3.cR*(HFS[4](M0,M1)))

class BubbleSF(Bubble):
    def __str__(self):
        return f'BubbleSF({self.v1!r}, {self.v2!r}, {self.v3!r},{self.masas!r})'
    
    def AL(self):
        v1,v2,v3 = self.vertices
        M0,M2 = self.masas
        return FactorRD*v1.c*(v2.cL*v3.cL*(HSF[1](M0,M2)) + v2.cR*v3.cR*(HSF[2](M0,M2)) + v2.cL*v3.cR*(HSF[3](M0,M2)) + v2.cR*v3.cL*(HSF[4](M0,M2)))
    def AR(self):
        v1,v2,v3 = self.vertices
        M0,M2 = self.masas
        return FactorRD*v1.c*(v2.cR*v3.cR*(HSF[1](M0,M2)) + v2.cL*v3.cL*(HSF[2](M0,M2)) + v2.cR*v3.cL*(HSF[3](M0,M2)) + v2.cL*v3.cR*(HSF[4](M0,M2)))

##########################################
### Width and BR h -> li lj
#########################################
import numpy as np
def Γhlilj(ML,MR,ma=125.18,mi=1.777,mj=0.1507):
    return 1/(8 *np.pi* ma)*np.sqrt((1-((mi**2+mj**2)/ma)**2)*(1-((mi**2-mj**2)/ma)**2))*((ma**2 - mi**2 - mj**2)*(np.abs(ML)**2 + np.abs(MR)**2)-4*mi*mj*np.real(ML*np.conj(MR)))


def BRhlilj(ML,MR,ma=125.18,mi=1.777,mj=0.1507):
    return Γhlilj(ML ,MR,ma,mi,mj)/(Γhlilj(ML ,MR,ma,mi,mj)+ 4.07e-3)

if __name__=='__main__':
    print('All right LFVHDFeynGv2')






