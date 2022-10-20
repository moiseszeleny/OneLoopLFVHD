#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17  13:49:53 2021

@author: Moises Zeleny
"""
from mpmath import log, polylog, mpf, log1p, pi
from mpmath import quad as integrate_mp
import mpmath as mp
from sympy import symbols, integrate
from sympy import log as logsp
from .roots_mpmath2 import x1, x2
from .roots_mpmath2 import y11 as y11np
from .roots_mpmath2 import y12 as y12np
from .roots_mpmath2 import y21 as y21np
from .roots_mpmath2 import y22 as y22np

# mp.dps = 80; mp.pretty = True
# from numba import jit, njit, prange

##############################################
# Funciones de Passarino Veltman pertinentes para LFVHD
##############################################


# @jit
def Li2(x0):
    '''
    Dilogarithm function

    Parameters
    ----------

    x0: float, mpf
    '''
    return polylog(2, x0)


# @jit
def R0(x0, xi):
    '''
    R0 function related with the C0 PV function

    Parameters
    ----------

    x0: float, mpf
    x0 root

    xi: float, mpf
    xi root, where i = 1,2,3
    '''
    return Li2(x0/(x0 - xi)) - Li2((x0 - 1.0)/(x0 - xi))

# R0_aprox = lambda x0,xi: Li2_aprox(x0/(x0-xi)) - Li2_aprox((x0-1)/(x0-xi))


# @jit
def x0(ma, M0, M2):
    '''
    x0 root

    Parameters
    ----------
    ma: float, mpf
    Mass of the Higgs H_a

    M0: float, mpf
    Mass of P0 particle inside the loop

    M2: float, mpf
    Mass of P2 particle inside the loop
    '''
    # return (M2**2 - M0**2)/ma**2
    return ((M2 - M0)*(M2 + M0))/ma**2


# @jit
delta = mpf('0.0')  # Delta value ####
def x3(M0, M1):
    '''
    x3 root

    Parameters
    ----------

    M0: float, mpf
    Mass of P0 particle inside the loop

    M1: float, mpf
    Mass of P1 particle inside the loop
    '''
    # return (-M0**2)/(M1**2 - M0**2)
    return (-M0**2 + 1j*Delta)/((M1 - M0)*(M1 + M0))


y, t = symbols('y,t')
# Sympy definition of f0
def f0sp(y):
    '''
    f0 function with integrate definition sympy

    Parameters
    ----------
        y: float, mpf
    '''
    return integrate(logsp(1 - (t/y)), (t, 0, 1))

# Sympy definition of f1
def f1sp(y):
    '''
    f1 function with integrate definition sympy

    Parameters
    ----------
        y: float, mpf
    '''
    return 2*integrate(t*logsp(1 - (t/y)), (t, 0, 1))

# mpmath definition of f0
def f0np(y):
    '''
    f0 function with integrate definition mpmath

    Parameters
    ----------
        y: float, mpf
    '''
    return integrate_mp(lambda t: log(1 - (t/y)), [0, 1])

# mpmath definition of f1
def f1np(y):
    '''
    f1 function with integrate definition mpmath

    Parameters
    ----------
        y: float, mpf
    '''
    return 2*integrate_mp(lambda x: x*log(1 - (x/y)), [0, 1])

# def f0np(y):
#     '''
#     f0 function

#     Parameters
#     ----------
#         y: float, mpf
#     '''
#     if y == 1:
#         out = -1
#     else:
#         out = y*log(-1.0*y) - y*log1p(-y) + log1p(-1.0/y) - 1.0
#     return out
#     # return y*log(-1.0*y) - y*log1p(-y) + log1p(-1.0/y) - 1.0


# def f1np(y):
#     '''
#     f1 function

#     Parameters
#     ----------
#         y: float, mpf
#     '''
#     if y == 1:
#         out = -mpf('3')/2
#     else:
#         out = mp.power(y, 2)*log(-y) -\
#             mp.power(y, 2)*log1p(-y) -\
#             y + log1p(-1/y) - mpf('0.5')  # lambdify([y],f1,'mpmath')
#     return out
#     # return mp.power(y, 2)*log(-y) -\
#     #     mp.power(y, 2)*log1p(-y) -\
#     #     y + log1p(-1/y) - mpf('0.5')


# #####################################################

# @jit

def f01sum(mi, M0, M1):
    '''
    Sum of f0(y11) + f0(y12)

    Parameters
    ----------
        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop
    '''
    out = 0.0
    for yi1j in (y11np, y12np):
        out += f0np(yi1j(mi, M0, M1))
    return out
# f01sum


# @jit
def f02sum(mj, M0, M2):
    '''
    Sum of f0(y21) + f0(y22)

    Parameters
    ----------
        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    out = 0
    for yi2j in (y21np, y22np):
        out += f0np(yi2j(mj, M0, M2))
    return out
# f02sum


# @jit
def f11sum(mi, M0, M1):
    '''
    Sum of f1(y11) + f1(y12)

    Parameters
    ----------
        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop
    '''
    out = 0
    for yi1j in (y11np, y12np):
        out += f1np(yi1j(mi, M0, M1))
    return out
# f11sum


# @jit
def f12sum(mj, M0, M2):
    '''
    Sum of f1(y21) + f1(y22)

    Parameters
    ----------
        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    out = 0
    for yi2j in (y21np, y22np):
        out += f1np(yi2j(mj, M0, M2))
    return out
#f12sum


#@jit
Delta = mpf('0.0')
def b1_0np(mi, M0, M1):
    '''
    b1_0 finite term of B1_0 PV function

    Parameters
    ----------
        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop
    '''
    return Delta - log(M0**2) - f01sum(mi, M0, M1)


# @jit
def b2_0np(mj, M0, M2):
    '''
    b2_0 finite term of B2_0 PV function

    Parameters
    ----------
        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return Delta - log(M0**2) - f02sum(mj, M0, M2)


# @jit
def b1_1np(mi, M0, M1):
    '''
    b1_1 finite term of B1_1 PV function

    Parameters
    ----------
        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop
    '''
    return mpf('0.5')*(Delta - log(M0**2) - f11sum(mi, M0, M1))


# @jit
def b2_1np(mj, M0, M2):
    '''
    b2_1 finite term of B2_1 PV function

    Parameters
    ----------
        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return - mpf('0.5')*(Delta - log(M0**2) - f12sum(mj, M0, M2))

#########################################################################
# Numpy definitions of PaVe functions
#a1 = 1j/(16*mp.pi**2)
#a2 = -1j/mp.pi**2

# @jit
def A0(ma, M):
    '''
    A0 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        M: float, mpf
        Mass of P particle inside the loop
    '''
    return M**2*(1 + log((ma**2)/(M**2)))


# @jit
def B1_0(mi, M0, M1):
    '''
    B1_0 PV function

    Parameters
    ----------
        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop
    '''
    return b1_0np(mi, M0, M1)
    # return 1-log(M1**2/ma**2) + (M0**2)/(M0**2-M1**2)*log(M1**2/M0**2)


# @jit
def B2_0(mj, M0, M2):
    '''
    B2_0 PV function

    Parameters
    ----------
        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return b2_0np(mj, M0, M2)
    # return 1-log(M2**2/ma**2) + (M0**2)/(M0**2-M2**2)*log(M2**2/M0**2)


# @jit
def B1_1(mi, M0, M1):
    '''
    B1_1 PV function

    Parameters
    ----------
        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop
    '''
    return b1_1np(mi, M0, M1)


# @jit
def B2_1(mj, M0, M2):
    '''
    B2_1 PV function

    Parameters
    ----------
        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return b2_1np(mj, M0, M2)


# @jit
def B12_0(ma, M1, M2):
    '''
    B12_0 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        M1: float, mpf
        Mass of P1 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    xi1, xi2 = x1(ma, M1, M2), x2(ma, M1, M2)
    # return (
    # log((ma**2)/(M1**2))/2.0 +
    # xi1*log((xi1- 1.0)/xi1) +
    # xi2*log((xi2- 1.0)/xi2))
    return 2 - log(M1**2) + xi1*log((xi1 - 1.0)/xi1) + xi2*log((xi2 - 1.0)/xi2)
    # return log((ma**2 )/(M1**2))/2 + sum(x*log(1-1/x) for x in [xi1,xi2])


# @jit
def B12_1(ma, M1, M2):
    '''
    B12_1 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        M1: float, mpf
        Mass of P1 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return (
        (1/(2*ma**2))*(
            M1**2*(1 + log(ma**2/M1**2)) - M2**2*(1 + log(ma**2/M2**2))
            ) +
        B12_0(ma, M1, M2)/(2*ma**2)*(M2**2-M1**2 + ma**2))


# @jit
def B12_2(ma, M1, M2):
    '''
    B12_2 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        M1: float, mpf
        Mass of P1 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return (
        (1/(2*ma**2))*(
            M1**2*(1 + log(ma**2/M1**2)) - M2**2*(1 + log(ma**2/M2**2))
            ) +
        B12_0(ma, M1, M2)/(2*ma**2)*(M2**2 - M1**2 - ma**2)
        )


# @jit
def C0(ma, M0, M1, M2):
    '''
    C0 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    y0 = x0(ma, M0, M2)
    y1 = x1(ma, M1, M2)
    y2 = x2(ma, M1, M2)
    y3 = x3(M0, M1)
    return ((R0(y0, y1) + R0(y0, y2) - R0(y0, y3))/ma**2)  # *(-1j*16)


# @jit
def C1(ma, mi, M0, M1, M2):
    '''
    C1 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        mi: float, mpf
        Mass of lepton li

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return (
        (1/ma**2)*(
            B1_0(mi, M0, M1) -
            B12_0(ma, M1, M2) +
            (M2**2 - M0**2)*C0(ma, M0, M1, M2)
            )
        )


# @jit
def C2(ma, mj, M0, M1, M2):
    '''
    C2 PV function

    Parameters
    ----------
        ma: float, mpf
        Mass of Higgs H_a

        mj: float, mpf
        Mass of lepton lj

        M0: float, mpf
        Mass of P0 particle inside the loop

        M1: float, mpf
        Mass of P1 particle inside the loop

        M2: float, mpf
        Mass of P2 particle inside the loop
    '''
    return (
        (-1/ma**2)*(
            B2_0(mj, M0, M2) -
            B12_0(ma, M1, M2) +
            (M1**2 - M0**2)*C0(ma, M0, M1, M2)
            )
        )
