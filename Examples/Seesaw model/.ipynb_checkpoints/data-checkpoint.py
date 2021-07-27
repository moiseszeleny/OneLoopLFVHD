#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22  12:20:53 2021

@author: Moises Zeleny

Numerical data of standar model particles and constants. ALso som useful tools to numeric implementations
"""
# charged leptón masses
mtau = 1.77686 #GeV
mmu = 0.10566 #GeV 
me = 0.000511 #GeV
ml = {1:me,2:mmu,3:mtau}

# intermediate pave functions with names allowed by lambdify
B10 = Function('B10')
B20 = Function('B20')
B11 = Function('B11')
B21 = Function('B21')
B120 = Function('B12_0')
def replaceBs(FF):
    '''
    Function to replace the pave function implemented LFVHDFeynGv3 
    (This is necessary because the original pave functions does not work with lambdify.)
    
    Return
    -------
        FF: Any simbolic expression with B PaVe functions.
        Return the original FF expression with the B functions replaced.
    '''
    from .LFVHDFeynGv3 import B1_0, B2_0, B1_1, B2_1, B12_0
    FF = FF.replace(B1_0,B10)
    FF = FF.replace(B2_0,B20)
    FF = FF.replace(B1_1,B11)
    FF = FF.replace(B2_1,B21)
    FF = FF.replace(B12_0,B120)
    return FF



def pave_functions(mh,a,b):
    '''
    Function to give the pertinent numeric implementations of pave functions.
    (This is necessary to give to lambdify the pertinent definitions of PaVe functions.
    
    Return
    -------
        mh: numerical value of decaying scalar mass.
        Return a dictionary with the changes of PaVe functions to numeric implementations.
    '''
    ma = mh
    mi = ml[a]
    mj = ml[b]
    from .LFVHDFeynG_numpy import B1_0, B2_0, B1_1, B2_1, B12_0, C0, C1, C2
    return {
        'B10': lambda M0,M1:B1_0(ma,mi,M0,M1),
        'B20': lambda M0,M2:B2_0(ma,mj,M0,M2),
        'B11': lambda M0,M1:B1_1(ma,mi,M0,M1),
        'B21': lambda M0,M2:B2_1(ma,mj,M0,M2),
        'B12_0': lambda M1,M2:B12_0(ma,M1,M2),
        'C_0': lambda M0,M1,M2:C0(ma,M0,M1,M2),
        'C_1': lambda M0,M1,M2:C1(ma,mi,M0,M1,M2),
        'C_2': lambda M0,M1,M2:C2(ma,mj,M0,M1,M2)
    }