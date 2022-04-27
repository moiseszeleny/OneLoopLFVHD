#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur May  13 02:18:53 2021

@author: Moises Zeleny
"""

from sympy import symbols, init_printing, conjugate,I,IndexedBase,sqrt,collect,simplify
from sympy import lambdify, Add

import OneLoopLFVHD as lfvhd
mh,mi,mj = lfvhd.ma,lfvhd.mi,lfvhd.mj

from mpmath import mp
mp.dps = 35; mp.pretty = True

# Defining variables
g,v = symbols(r'g,v',real=True)
λ1 = mh**2/v**2
U = IndexedBase(r'U')
l,b,i,j = symbols('l,b,i,j',integer=True)
mnul,mW = symbols(r'm_{{\nu_l}},m_W',real=True)

## Vertexes
hGdGu = lfvhd.VertexHSS(-I*λ1*v)
Guljνl = lfvhd.VertexSFF(I*(sqrt(2)/v)*mj*U[l,j],0)
Gdliνl = lfvhd.VertexSFF(0,I*(sqrt(2)/v)*mi*conjugate(U[l,i]))
# Guljνl = lfvhd.VertexSFF((I*g)/(sqrt(2)*mW)*mj*U[l,j],(-I*g)/(sqrt(2)*mW)*mnul*U[l,j])
# Gdliνl = lfvhd.VertexSFF((-I*g)/(sqrt(2)*mW)*mnul*conjugate(U[l,i]),
#                          (I*g)/(sqrt(2)*mW)*mi*conjugate(U[l,i]))

hGdWu = lfvhd.VertexHSpVm(-I*g/2)
Wuljνl = lfvhd.VertexVFF(0,-I*(g/sqrt(2))*U[l,j])
hWdGu = lfvhd.VertexHVpSm(-I*g/2)
Wdliνl = lfvhd.VertexVFF(0,-I*(g/sqrt(2))*conjugate(U[l,i]))
hWdWu = lfvhd.VertexHVV(I*g**2/2*v)
hljlj = lfvhd.VertexHFF((I*g*mj)/(2*sqrt(2)*mW))
hlili = lfvhd.VertexHFF((I*g*mi)/(2*sqrt(2)*mW))

## Diagrams
νlGG = lfvhd.TriangleFSS(hGdGu,Guljνl,Gdliνl,[mnul,mW,mW])
νlGW = lfvhd.TriangleFSV(hGdWu,Wuljνl,Gdliνl,[mnul,mW,mW])
νlWG = lfvhd.TriangleFVS(hWdGu,Guljνl,Wdliνl,[mnul,mW,mW])
νlWW = lfvhd.TriangleFVV(hWdWu,Wuljνl,Wdliνl,[mnul,mW,mW])
nlW = lfvhd.BubbleFV(hljlj,Wuljνl,Wdliνl,[mnul,mW])
Wnl = lfvhd.BubbleVF(hlili,Wuljνl,Wdliνl ,[mnul,mW])
nlG = lfvhd.BubbleFS(hljlj,Guljνl,Gdliνl,[mnul,mW])
Gnl = lfvhd.BubbleSF(hlili,Guljνl,Gdliνl,[mnul,mW])

Diagrams = [νlGG, νlGW, νlWG, νlWW, nlW, Wnl, nlG, Gnl]
#Diagrams = [νlGG,nlG, Gnl]

#print(Add(*[D.AL() for D in Diagrams]).expand().collect([mnul],evaluate=False).keys())
ALtot = Add(*[D.AL() for D in Diagrams])#.expand().collect([mnul],evaluate=False)[mnul**2]*mnul**2
ARtot = Add(*[D.AR() for D in Diagrams])#.expand().collect([mnul],evaluate=False)[mnul**2]*mnul**2

### Lamdifyng
from OneLoopLFVHD.data import ml
mh = symbols('m_h',real=True)
valores ={mW:mp.mpf('80.379'),mh:mp.mpf('125.10'),
          g:(2*mp.mpf('80.379'))/mp.mpf('246'),v:mp.mpf('246')}

cambios_hij = lambda ii, jj:{lfvhd.ma:valores[mh],lfvhd.mi:ml[ii],lfvhd.mj:ml[jj]}

Ulj, Ucli = symbols('U_{lj}, {{U_{li}^*}}')
UOne = {U[l,j]:Ulj,conjugate(U[l,i]):Ucli}


from OneLoopLFVHD.data import  pave_functions, replaceBs
zero = lambda a,b: 0
#.replace(lfvhd.B12_0,zero)
ALtot_mp = lambda i,j: lambdify([mnul,Ulj, Ucli],
    replaceBs(ALtot.replace(lfvhd.B12_0,zero)).subs(cambios_hij(i,j)).subs(lfvhd.D,4).subs(valores).subs(UOne),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

ARtot_mp = lambda i,j: lambdify([mnul,Ulj, Ucli],
    replaceBs(ARtot.replace(lfvhd.B12_0,zero)).subs(cambios_hij(i,j)).subs(lfvhd.D,4).subs(valores).subs(UOne),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

## Oscillations data
from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization, NuOscObservables
Nudata = NuOscObservables
Upmns = mp.matrix([
[ 0.821302075974486,  0.550502406897554, 0.149699699398496],
[-0.463050759961518,  0.489988544456971, 0.738576482160108],
[ 0.333236993293153, -0.675912957636513, 0.657339166640784]])

#### Sum over neutrino generations
def Aijtot(m1,ii,jj,quirality='L'):
    #m1 = mp.mpf('1e-12')  #GeV 
    #current values to Square mass differences
    d21 = mp.mpf(str(Nudata.squareDm21.central))*mp.mpf('1e-18')# factor to convert eV^2 to GeV^2
    d31 = mp.mpf(str(Nudata.squareDm31.central))*mp.mpf('1e-18')

    m2 = mp.sqrt(m1**2 + d21)
    m3 = mp.sqrt(m1**2 + d31)
    U = Upmns
    
    AL = ALtot_mp(ii,jj)
    ALsuml =  (AL(m1,U[0,jj-1],mp.conj(U[0,ii-1])) + 
            AL(m2,U[1,jj-1],mp.conj(U[1,ii-1])) + 
            AL(m3,U[2,jj-1],mp.conj(U[2,ii-1])))
    AR = ARtot_mp(ii,jj)
    ARsuml =  (AR(m1,U[0,jj-1],mp.conj(U[0,ii-1])) + 
            AR(m2,U[1,jj-1],mp.conj(U[1,ii-1])) + 
            AR(m3,U[2,jj-1],mp.conj(U[2,ii-1])))
    if quirality=='L':
        out = ALsuml
    elif quirality=='R':
        out = ARsuml
    else:
        raise ValueError('quirality must be L or R')
    return out

def AL23tot(m1):
    return Aijtot(m1,2,3,'L')
def AR23tot(m1):
    return Aijtot(m1,2,3,'R')

def AL32tot(m1):
    return Aijtot(m1,3,2,'L')
def AR32tot(m1):
    return Aijtot(m1,3,2,'R')

def AL13tot(m1):
    return Aijtot(m1,1,3,'L')
def AR13tot(m1):
    return Aijtot(m1,1,3,'R')

def AL31tot(m1):
    return Aijtot(m1,3,1,'L')
def AR31tot(m1):
    return Aijtot(m1,3,1,'R')

def AL12tot(m1):
    return Aijtot(m1,1,2,'L')
def AR12tot(m1):
    return Aijtot(m1,1,2,'R')

def AL21tot(m1):
    return Aijtot(m1,2,1,'L')
def AR21tot(m1):
    return Aijtot(m1,2,1,'R')

