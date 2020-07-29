#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 01:07:53 2020

@author: Moises Zeleny
"""

import OneLoopLFVHD as lfvhd
from sympy import IndexedBase,symbols,conjugate,I

##########################
### Triangulo Nu eta eta 
#########################

# Defining symbols
μ2,v = symbols(r'\mu_2,v',real=True)
Yν = IndexedBase(r'{{Y^\nu}}')
l,i,j = symbols('l,i,j',integer=True)

# Defining vertexes
mNul,mη = symbols(r'm_{{N_l}},m_{{\eta}}',real=True)
λ3 = (2/v**2)*(mη**2-μ2**2)

mNul,mη = symbols(r'm_{{N_l}},m_{{\eta}}',real=True)
masasNηη = [mNul,mη,mη]

hηuηd = lfvhd.VertexHSS(-I*λ3*v)
ηdljNl = lfvhd.VertexSFF(0,I*conjugate(Yν[l,j]))
ηuliNl = lfvhd.VertexSFF(I*Yν[l,i],0)

Nlηuηd = lfvhd.TriangleFSS(hηuηd,ηdljNl,ηuliNl,masasNηη)
#################################
### Burbuja Nu eta
#################################

#Hdμνl,Huτνl
g2,mW = symbols('g_2,m_W',positive=True)
c = lambda m:(I*g2*m)/(2*mW)

hl2l2 = lfvhd.VertexHFF((I*g2*lfvhd.mj)/(2*mW))
Nlη = lfvhd.BubbleFS(hl2l2,ηdljNl,ηuliNl,[mNul,mη])


#################################
### Burbuja eta Nu
#################################

hl1l1 = lfvhd.VertexHFF((I*g2*lfvhd.mi)/(2*mW))
ηNl = lfvhd.BubbleSF(hl1l1,ηdljNl,ηuliNl,[mNul,mη])
################################################
#### Sumando burbujas
###############################################

BubML = Nlη.ML() + ηNl.ML()
BubML = BubML.subs(lfvhd.cambiosDivFin(mNul,mη,mη)).subs(
    lfvhd.cambios_aprox(mNul,mη,mη)).simplify()


BubMR =Nlη.MR() + ηNl.MR()
BubMR = BubMR.subs(lfvhd.cambiosDivFin(mNul,mη,mη)).subs(
    lfvhd.cambios_aprox(mNul,mη,mη)).simplify()

#########################
### Triangles
#########################
constantes = {mW:80.379,v:246,g2:2*(80.379/246)}
#ML_GIM()
TriangNL = (Nlηuηd.ML().subs(lfvhd.cambiosDivFin(*Nlηuηd.masas)).subs(lfvhd.cambios_aprox(*Nlηuηd.masas)))


TriangNR = Nlηuηd.MR().subs(lfvhd.cambiosDivFin(*Nlηuηd.masas)).subs(lfvhd.cambios_aprox(*Nlηuηd.masas))




##########################3
### Sumando todas las contribuciones
##########################

MLhlilj  = (TriangNL + BubML)#.subs(constantes)
MRhlilj = (TriangNR + BubMR)#.subs(constantes)






























