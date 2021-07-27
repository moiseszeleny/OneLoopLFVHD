#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur May  13 02:18:53 2021

@author: Moises Zeleny
"""

from sympy import symbols, init_printing, conjugate,I,IndexedBase,sqrt,collect,simplify
from sympy import lambdify

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

### Lamdifyng
from OneLoopLFVHD.data import ml
mh = symbols('m_h',real=True)
valores ={mW:mp.mpf('80.379'),mh:mp.mpf('125.10'),
          g:(2*mp.mpf('80.379'))/mp.mpf('246'),v:mp.mpf('246')}

cambios_hij = lambda ii, jj:{lfvhd.ma:valores[mh],lfvhd.mi:ml[ii],lfvhd.mj:ml[jj]}

Ulj, Ucli = symbols('U_{lj}, {{U_{li}^*}}')
UOne = {U[l,j]:Ulj,conjugate(U[l,i]):Ucli}

from OneLoopLFVHD.data import replaceBs, pave_functions
Dia1mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlGG.AL().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia1mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlGG.AR().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia2mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlGW.AL().subs(lfvhd.B12_0(mW,mW),0).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia2mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlGW.AR().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia3mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlWG.AL().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia3mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlWG.AR().subs(lfvhd.B12_0(mW,mW),0).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia4mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlWW.AL().subs(lfvhd.D,4).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia4mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    νlWW.AR().subs(lfvhd.D,4).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia5mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    nlW.AL().subs(lfvhd.D,4).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia5mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    nlW.AR().subs(lfvhd.D,4).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia6mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    Wnl.AL().subs(lfvhd.D,4).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia6mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    Wnl.AR().subs(lfvhd.D,4).subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia7mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    nlG.AL().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia7mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    nlG.AR().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia8mpL = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    Gnl.AL().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

Dia8mpR = lambda i,j: lambdify([mnul,Ulj, Ucli],replaceBs(
    Gnl.AR().subs(cambios_hij(i,j)).subs(valores).subs(UOne)),
                  modules=[pave_functions(valores[mh],i,j,lib='mpmath'),'mpmath'])

## Oscillations data
from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization, NuOscObservables
Nudata = NuOscObservables
Upmns = mp.matrix([
[ 0.821302075974486,  0.550502406897554, 0.149699699398496],
[-0.555381876513578,  0.489988544456971, 0.738576482160108],
[ 0.333236993293153, -0.675912957636513, 0.657339166640784]])

#### Sum over neutrino generations
def Dia_tot(m1,ii,jj,Dia_dict,quirality='L'):
    #m1 = mp.mpf('1e-12')  #GeV 
    #current values to Square mass differences
    d21 = mp.mpf(str(Nudata.squareDm21.central))*mp.mpf('1e-18')# factor to convert eV^2 to GeV^2
    d31 = mp.mpf(str(Nudata.squareDm31.central))*mp.mpf('1e-18')

    m2 = mp.sqrt(m1**2 + d21)
    m3 = mp.sqrt(m1**2 + d31)
    U = Upmns
    
    if quirality=='L':
        F = Dia_dict[0](ii,jj)
    elif quirality=='R':
        F = Dia_dict[1](ii,jj)
    else:
        raise ValueError('quirality must be L or R')
    return (F(m1,U[0,jj-1],mp.conj(U[0,ii-1])) + 
            F(m2,U[1,jj-1],mp.conj(U[1,ii-1])) + 
            F(m3,U[2,jj-1],mp.conj(U[2,ii-1])))

def Aijtot(m1,ii,jj,quirality='L'):
    d1 = Dia_tot(m1,ii,jj,[Dia1mpL,Dia1mpR],quirality)
    d2 = Dia_tot(m1,ii,jj,[Dia2mpL,Dia2mpR],quirality)
    d3 = Dia_tot(m1,ii,jj,[Dia3mpL,Dia3mpR],quirality)
    d4 = Dia_tot(m1,ii,jj,[Dia4mpL,Dia4mpR],quirality)
    d5 = Dia_tot(m1,ii,jj,[Dia5mpL,Dia5mpR],quirality)
    d6 = Dia_tot(m1,ii,jj,[Dia6mpL,Dia6mpR],quirality)
    d7 = Dia_tot(m1,ii,jj,[Dia7mpL,Dia7mpR],quirality)
    d8 = Dia_tot(m1,ii,jj,[Dia8mpL,Dia8mpR],quirality)    
    return d1 + d2 +d3 +d4 +d4+ d6+ d7+ d8

def AL23tot(m1):
    return Aijtot(m1,2,3,'L')
def AR23tot(m1):
    return Aijtot(m1,2,3,'R')

def AL13tot(m1):
    return Aijtot(m1,1,3,'L')
def AR13tot(m1):
    return Aijtot(m1,1,3,'R')

def AL12tot(m1):
    return Aijtot(m1,1,2,'L')
def AR12tot(m1):
    return Aijtot(m1,1,2,'R')

AL23tot = mp.memoize(mp.maxcalls(AL23tot, 1))
AR23tot = mp.memoize(mp.maxcalls(AR23tot, 1))

AL13tot = mp.memoize(mp.maxcalls(AL13tot, 1))
AR13tot = mp.memoize(mp.maxcalls(AR13tot, 1))

AL12tot = mp.memoize(mp.maxcalls(AL12tot, 1))
AR12tot = mp.memoize(mp.maxcalls(AR12tot, 1))