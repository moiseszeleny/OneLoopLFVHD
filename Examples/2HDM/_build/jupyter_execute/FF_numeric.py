#!/usr/bin/env python
# coding: utf-8

# # LFV Higgs decays in SeeSaw model ( Thao et al results) 

# ## In this notebook we use the mpmath implementations of PaVe functions

# In[1]:


from sympy import init_printing, Symbol,lambdify, symbols, Matrix
init_printing()
import OneLoopLFVHD as lfvhd


# In[2]:


from seesaw_FF import TrianglesOneFermion, TrianglesTwoFermion, Bubbles,DiagramsOneFermionW, DiagramsOneFermionG
from seesaw_FF import g, mW, Uν, Uνc, mn, m, C, Cc, a,b,i,h
from seesaw_FF import j as jj


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from mpmath import *


# In[5]:


mp.dps = 80; mp.pretty = True


# In[6]:


import numpy as np


# In[7]:


import subprocess as s


# In[8]:


from multiprocessing import Pool
from time import time


# In[9]:


def speedup_array(f,array,procs=4): 
    pool = Pool(procs,maxtasksperchild=100).map(f, array)
    result = np.array(list(pool))
    return result


# ## Numeric implementation of form factors

# **Neutrino benchmark** is given by 

# In[10]:


from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables


# In[11]:


m1 = mpf('1e-12')  #GeV 

#current values to Square mass differences
d21 = mpf(str(Nudata.squareDm21.central))*mpf('1e-18')# factor to convert eV^2 to GeV^2
d31 = mpf(str(Nudata.squareDm31.central))*mpf('1e-18')

#d21 = 7.5e-5*1e-18
#d31 = 2.457e-3*1e-18
m2 = sqrt(m1**2 + d21)
m3 = sqrt(m1**2 + d31)

m4 = lambda m6: m6/3
m5 = lambda m6: m6/2


# ### Form factor with one fermion in the loop.

# #### AL one fermion 

# In[12]:


from OneLoopLFVHD.data import ml


# In[13]:


mh,ma,mb = symbols('m_h,m_a,m_b',real=True)
valores ={mW:mpf('80.379'),mh:mpf('125.10'),g:(2*mpf('80.379'))/mpf('246')}

cambios_hab = lambda a,b:{lfvhd.ma:valores[mh],lfvhd.mi:ml[a],lfvhd.mj:ml[b]}


Ubi, Ucai,mni = symbols('U_{bi}, {{U_{ai}^*}},m_{n_i}')
UnuOne = {mn[i]:mni,Uν[b,i]:Ubi,Uνc[a,i]:Ucai}

from Unu_seesaw import diagonalizationMnu
diagonalizationMnu1 = lambda m1,m6: diagonalizationMnu(
    m1,m2,m3,m6/mpf('3.0'),m6/mpf('2.0'),m6)


# In[14]:


def GIM_One(exp):
    from sympy import Add
    args = exp.expand().args
    func = exp.expand().func
    if isinstance(func,Add):
        X = Add(*[t for t in args if t.has(mni)]).simplify()
    else:
        X = exp
    #X1 = X.collect([mni],evaluate=False)
    return X#mni**2*X1[mni**2]


# In[15]:


def sumOne(m6,Aab,a,b): 
    mnk,Unu = diagonalizationMnu1(m1,m6)
    AL = []
    for k in range(1,7):
        A = Aab(mnk[k-1],Unu[b-1,k-1],conj(Unu[a-1,k-1]))
        #print('Ai = ',A)
        AL.append(A)
    return mp.fsum(AL)


# In[16]:


from OneLoopLFVHD.data import replaceBs, pave_functions


# In[17]:


def numeric_sum_diagramsOne(a,b,quirality='L'):
    FdiagOneFer = []
    for Set in [TrianglesOneFermion,Bubbles]:#TrianglesOneFermion,Bubbles
        for dia in Set:
            if quirality=='L':
                x = dia.AL().subs(lfvhd.D,4).subs(lfvhd.B12_0(mW,mW),0).subs(cambios_hab(a,b)).subs(valores).subs(UnuOne)
            elif quirality=='R':
                x = dia.AR().subs(lfvhd.D,4).subs(lfvhd.B12_0(mW,mW),0).subs(cambios_hab(a,b)).subs(valores).subs(UnuOne)
            else:
                raise ValueError('quirality must be L or R')
            f = lambdify([mni,Ubi,Ucai],replaceBs(x),
                         modules=[pave_functions(valores[mh],a,b,lib='mpmath'),'mpmath'])
            #print(f(1,2,3))
            #fsum = lambda m6:sumOne(m6,f,a,b)
            FdiagOneFer.append(f)
    def suma(m6):
        out = []
        xs = []
        for g in FdiagOneFer:
            
            x = sumOne(m6,g,a,b)
            out.append(x)
            xs.append(x)
        return np.array(xs), mp.fsum(out)
    return suma


# In[18]:


# #a = 2, b = 3
# ALOneTot23 = numeric_sum_diagramsOne(2,3,quirality='L')
# AROneTot23 = numeric_sum_diagramsOne(2,3,quirality='R')

# #a = 3, b = 2
# ALOneTot32 = numeric_sum_diagramsOne(3,2,quirality='L')
# AROneTot32 = numeric_sum_diagramsOne(3,2,quirality='R')

# #a = 1, b = 3
# ALOneTot13 = numeric_sum_diagramsOne(1,3,quirality='L')
# AROneTot13 = numeric_sum_diagramsOne(1,3,quirality='R')

# #a = 3, b = 1
# ALOneTot31 = numeric_sum_diagramsOne(3,1,quirality='L')
# AROneTot31 = numeric_sum_diagramsOne(3,1,quirality='R')

# #a = 1, b = 2
# ALOneTot12 = numeric_sum_diagramsOne(1,2,quirality='L')
# AROneTot12 = numeric_sum_diagramsOne(1,2,quirality='R')

# #a = 2, b = 1
# ALOneTot21 = numeric_sum_diagramsOne(2,1,quirality='L')
# AROneTot21 = numeric_sum_diagramsOne(2,1,quirality='R')


# In[19]:


def ALOneTot23(m6):
    return numeric_sum_diagramsOne(2,3,quirality='L')(m6)[1]
def AROneTot23(m6):
    return numeric_sum_diagramsOne(2,3,quirality='R')(m6)[1]

def ALOneTot32(m6):
    return numeric_sum_diagramsOne(3,2,quirality='L')(m6)[1]
def AROneTot32(m6):
    return numeric_sum_diagramsOne(3,2,quirality='R')(m6)[1]

def ALOneTot13(m6):
    return numeric_sum_diagramsOne(1,3,quirality='L')(m6)[1]
def AROneTot13(m6):
    return numeric_sum_diagramsOne(1,3,quirality='R')(m6)[1]

def ALOneTot31(m6):
    return numeric_sum_diagramsOne(3,1,quirality='L')(m6)[1]
def AROneTot31(m6):
    return numeric_sum_diagramsOne(3,1,quirality='R')(m6)[1]

def ALOneTot12(m6):
    return numeric_sum_diagramsOne(1,2,quirality='L')(m6)[1]
def AROneTot12(m6):
    return numeric_sum_diagramsOne(1,2,quirality='R')(m6)[1]

def ALOneTot21(m6):
    return numeric_sum_diagramsOne(2,1,quirality='L')(m6)[1]
def AROneTot21(m6):
    return numeric_sum_diagramsOne(2,1,quirality='R')(m6)[1]


# In[20]:


ALOneTot23(1)


# In[21]:


n = 200
expmp = linspace(-1,15,n)
m6np = np.array([mpf('10.0')**k for k in expmp])#np.logspace(-1,15,n)


# In[22]:


get_ipython().run_cell_magic('time', '', 'YLOne23 = speedup_array(ALOneTot23,m6np)\n#YLOne32 = speedup_array(ALOneTot32,m6np)\n\nYLOne13 = speedup_array(ALOneTot13,m6np)\n#YLOne31 = speedup_array(ALOneTot31,m6np)\n\nYLOne12 = speedup_array(ALOneTot12,m6np)\n#YLOne21 = speedup_array(ALOneTot21,m6np)')


# In[26]:


plt.figure(figsize=(15,8))
plt.loglog(np.real(m6np),factor*abs(YLOne23)**2,'-.',label='$A_L^{(1)}(2,3)$')
#plt.loglog(np.real(m6np),factor*abs(YLOne32)**2,'--',label='$A_L^{(1)}(3,2)$')

plt.loglog(np.real(m6np),factor*abs(YLOne13)**2,'-.',label='$A_L^{(1)}(1,3)$')
#plt.loglog(np.real(m6np),factor*abs(YLOne31)**2,'--',label='$A_L^{(1)}(3,1)$')

plt.loglog(np.real(m6np),factor*abs(YLOne12)**2,'-.',label='$A_L^{(1)}(1,2)$')
#plt.loglog(np.real(m6np),factor*abs(YLOne21)**2,'--',label='$A_L^{(1)}(2,1)$')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=15)


# ### Form factor with two fermion in the loop.

# In[27]:


mnj = symbols('m_{n_j}',positive=True)
Cijs, Cijcs, Ubj = symbols('C_{ij}, {{C_{ij}^*}},U_{bj}')
UnuTwo = {mn[i]:mni,mn[jj]:mnj,C[i,jj]:Cijs, Cc[i,jj]:Cijcs, Uν[b,jj]:Ubj, Uνc[a,i]:Ucai}
UnuTwo


# In[28]:


fsL = lambda k,a,b:TrianglesTwoFermion[k].AL().subs(lfvhd.D,4).subs(cambios_hab(a,b)).subs(valores).subs(UnuTwo)
fsR = lambda k,a,b:TrianglesTwoFermion[k].AR().subs(lfvhd.D,4).subs(cambios_hab(a,b)).subs(valores).subs(UnuTwo)


# In[29]:


fL = lambda k,a,b:lambdify([mni,mnj,Ubj,Ucai,Cijs,Cijcs],replaceBs(fsL(k,a,b)),
                     modules=[pave_functions(valores[mh],a,b,lib='mpmath'),'mpmath'] )
fR = lambda k,a,b:lambdify([mni,mnj,Ubj,Ucai,Cijs,Cijcs],replaceBs(fsR(k,a,b)),
                     modules=[pave_functions(valores[mh],a,b,lib='mpmath'),'mpmath'] )


# In[30]:


fL(0,2,3)(1,2,3,4,5,6)


# In[31]:


def sumatwo(mm6,k,a,b,quirality='L'):
    xs = []
    if quirality=='L':
        g = fL(k,a,b)
    elif quirality=='R':
        g = fR(k,a,b)
    else:
        raise ValueError('quirality must be L or R')
        
    mnk,Unu = diagonalizationMnu1(m1,mm6)
    Cij = lambda i,j: mp.fsum([Unu[c,i]*conj(Unu[c,j]) for c in range(3)])
    for p in range(1,7):
        for q in range(1,7):
            x = g(mnk[p-1],mnk[q-1],Unu[b-1,q-1],conj(Unu[a-1,p-1]),Cij(p-1,q-1),conj(Cij(p-1,q-1)))
            xs.append(x)
            #print(f'i = {p} and j = {q}')
            #print(f'|f| = {x}')
    return mp.fsum(xs)


# In[35]:


def totaltwo(m6,a,b,quirality='L'):
    return sumatwo(m6,0,a,b,quirality) + sumatwo(m6,1,a,b,quirality)


# In[36]:


ALTwoTot23 = lambda m6: totaltwo(m6,2,3,'L')
ARTwoTot23 = lambda m6: totaltwo(m6,2,3,'R')

ALTwoTot32 = lambda m6: totaltwo(m6,3,2,'L')
ARTwoTot32 = lambda m6: totaltwo(m6,3,2,'R')

ALTwoTot13 = lambda m6: totaltwo(m6,1,3,'L')
ARTwoTot13 = lambda m6: totaltwo(m6,1,3,'R')

ALTwoTot31 = lambda m6: totaltwo(m6,3,1,'L')
ARTwoTot31 = lambda m6: totaltwo(m6,3,1,'R')

ALTwoTot12 = lambda m6: totaltwo(m6,1,2,'L')
ARTwoTot12 = lambda m6: totaltwo(m6,1,2,'R')

ALTwoTot21 = lambda m6: totaltwo(m6,2,1,'L')
ARTwoTot21 = lambda m6: totaltwo(m6,2,1,'R')


# In[37]:


abs(ALTwoTot23(m6np[-1])),abs(sumatwo(m6np[-1],0,2,3,'L')+ sumatwo(m6np[-1],1,2,3,'L'))


# ## Total Form Factors

# In[38]:


#a = 2, b = 3
def ALtot23(m6):
    return  ALOneTot23(m6) + ALTwoTot23(m6)
def ARtot23(m6):
    return  AROneTot23(m6) + ARTwoTot23(m6)

#a = 3, b = 2
def ALtot32(m6):
    return  ALOneTot32(m6) + ALTwoTot32(m6)
def ARtot32(m6):
    return  AROneTot32(m6) + ARTwoTot32(m6)

#a = 1, b = 3
def ALtot13(m6):
    return  ALOneTot13(m6) + ALTwoTot13(m6)
def ARtot13(m6):
    return  AROneTot13(m6) + ARTwoTot13(m6)

#a = 3, b = 1
def ALtot31(m6):
    return  ALOneTot31(m6) + ALTwoTot31(m6)
def ARtot31(m6):
    return  AROneTot31(m6) + ARTwoTot31(m6)

#a = 1, b = 2
def ALtot12(m6):
    return  ALOneTot12(m6) + ALTwoTot12(m6)
def ARtot12(m6):
    return  AROneTot12(m6) + ARTwoTot12(m6)

#a = 2, b = 1
def ALtot21(m6):
    return  ALOneTot21(m6) + ALTwoTot21(m6)
def ARtot21(m6):
    return  AROneTot21(m6) + ARTwoTot21(m6)


# ## Width decay of $h \to e_a e_b$

# In[39]:


from OneLoopLFVHD import Γhlilj


# In[40]:


def Γhl2l3(m6):
    return Γhlilj(ALtot23(m6),ARtot23(m6),valores[mh],ml[2],ml[3])
def Γhl3l2(m6):
    return Γhlilj(ALtot32(m6),ARtot32(m6),valores[mh],ml[3],ml[2])

def Γhl1l3(m6):
    return Γhlilj(ALtot13(m6),ARtot13(m6),valores[mh],ml[1],ml[3])
def Γhl3l1(m6):
    return Γhlilj(ALtot31(m6),ARtot31(m6),valores[mh],ml[3],ml[1])

def Γhl1l2(m6):
    return Γhlilj(ALtot12(m6),ARtot12(m6),valores[mh],ml[1],ml[2])
def Γhl2l1(m6):
    return Γhlilj(ALtot21(m6),ARtot21(m6),valores[mh],ml[2],ml[1])


# In[41]:


n = 800
expmp = linspace(-1,15,n)
m6np = np.array([mpf('10.0')**k for k in expmp])#np.logspace(-1,15,n)


# In[42]:


get_ipython().run_cell_magic('time', '', 'YW23 = speedup_array(Γhl2l3,m6np)\n#YW32 = speedup_array(Γhl3l2,m6np)\n\nYW13 = speedup_array(Γhl1l3,m6np)\n#YW31 = speedup_array(Γhl3l1,m6np)\n\nYW12 = speedup_array(Γhl1l2,m6np)\n\n#YW21 = speedup_array(Γhl2l1,m6np)')


# In[43]:


Wtot = YW23 + YW13 + YW12 + 0.0032# + YW32 + YW31 + YW21


# In[44]:


plt.figure(figsize=(15,8))
plt.loglog(np.real(m6np),(YW23 #+ YW32
                         )/Wtot,label=r'Br($h \to \mu \tau$)')
plt.loglog(np.real(m6np),(YW13 #+ YW31
                         )/Wtot,label=r'Br($h \to e \tau$)')
plt.loglog(np.real(m6np),(YW12 #+ YW21
                         )/Wtot,label=r'Br($h \to e \mu$)')

#xx = ((YW23 + YW32)/Wtot)[-1]
plt.hlines(1e-10,0.1,1e15,linestyles='-.',label=r'$1.7\times 10^{-2}$')
plt.hlines(5e-43,0.1,1e15,linestyles='--',color='b',label=r'$1\times 10^{-32}$')
plt.vlines(125.1,5e-43,1e-10,linestyles='--',color='r',label=r'$m_W$')
plt.xlim(1e-1,1e15)
plt.yticks([1e-39,1e-29,1e-19,1e-9,1])
plt.xticks([1,1e4,1e8,1e12,1e16])

plt.legend(fontsize=15)


# In[37]:


import pandas as pd


# In[38]:


df = pd.DataFrame({'m6':m6np,
                   'Whl2l3':YW23,
                   #'Whl3l2':YW32,
                   'Whl1l3':YW13,
                   #'Whl3l1':YW31,
                   'Whl1l2':YW12})
                   #'Whl2l1':YW21})


# In[39]:


df.to_csv('LFVHD-3.txt',sep='\t')


# In[40]:


plt.semilogy(np.array(list(map(mpf,df['Whl2l3']))))


# In[ ]:




