
# %%
from mpmath import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
mp.dps = 80; mp.pretty = True
def Wh_nuN(mh,Mj,Cij):
    g = (2*80.379)/(246)
    mW = 80.379
    if Mj<mh:
        out = (Cij**2/(8*mp.pi*mh**3))*(mh**2 - Mj**2)**2
    else:
        out = 0.0
    return out

# %%
LFVHD_data = pd.read_csv("Version 4/Widths.txt",sep='\t')#_no_mni

# %%
convert_to_array = lambda df,col:np.array(list(map(mpmathify,df[col])))

# %%
m6 = convert_to_array(LFVHD_data,'m6')
Whl1l2 = convert_to_array(LFVHD_data,'Whl1l2')
#Whl2l1 = convert_to_array(LFVHD_data,'Whl2l1')
Whl1l3 = convert_to_array(LFVHD_data,'Whl1l3')
#Whl3l1 = convert_to_array(LFVHD_data,'Whl3l1')
Whl2l3 = convert_to_array(LFVHD_data,'Whl2l3')
#Whl3l2 = convert_to_array(LFVHD_data,'Whl3l2')

# %%
from Unu_seesaw import YN, Osc_data, mheavy, mlight,mn1,mn6
from sympy import lambdify

# %%
Ynu = lambdify([mn1,mn6],YN.subs(Osc_data).subs(mheavy).subs(mlight),'mpmath')

# %%
Ynuij = lambda m6,i,j: Ynu(mpf(1e-12),m6)[i,j]

# %%
mn = lambda m6: (m6/3, m6/2, m6)
def WhnuN_sum(m6):
    out = 0
    for i in range(3):
        for j in range(3):
            out += Wh_nuN(125.1,mn(m6)[j],Ynuij(m6,i,j))
    return out

# %% [markdown]
# ## Total width

# %%
WidthSM = 0.0032
Whlilj_tot = Whl1l2  + Whl1l3  + Whl2l3  #+ Whl2l1 + Whl3l1 + Whl3l2

# %%
WHnuN_tot = np.array([WhnuN_sum(m) for m in m6])

# %%
Wtot = (
    Whlilj_tot + 
    abs(WHnuN_tot) + 
    WidthSM
)
#Wtot

# %% [markdown]
# ## Branching ratios

# %%
from OneLoopLFVHD.data import ml

# %%
with plt.style.context('seaborn-deep'):
    plt.figure(figsize=(12,8))
    plt.loglog(m6,abs(Whl2l3/Wtot),label=r'Br$(h \to \mu \tau)$',linewidth=2.5)
    plt.loglog(m6,abs(Whl1l3/Wtot),label=r'Br$(h \to e \tau)$',linewidth=2.5)
    plt.loglog(m6,abs(Whl1l2/Wtot),label=r'Br$(h \to e \mu)$',linewidth=2.5)

    #Horizontal lines
    #plt.hlines(1e-10,0.1,1e15,color='0.1',linewidth=1.5,linestyles='-.',label=r'$10^{-10}$')
    #plt.hlines(1e-30,0.1,1e15,color='0.1',linewidth=1.5,linestyles=':',label=r'$10^{-30}$')
    #plt.hlines(1e-43,0.1,1e15,color='0.1',linewidth=1.5,linestyles='--',label=r'$1\times 10^{-43}$')

    
    #plt.xticks([1,125.1,1e4,1e8,1e12,1e15],
    #           ['1','$m_h$','$10^4$','$10^8$','$10^{12}$','$10^{15}$'],fontsize=20)
    #plt.yticks([1e-49,1e-39,1e-29,1e-19,1e-9],fontsize=20)
    #plt.xlim(1e-1,1e15)

    plt.legend(fontsize=18,ncol=2,frameon=False,bbox_to_anchor=(0.5,0.6))
    #plt.ylabel(r'BR($h \to e_a e_b$)',fontsize=18)
    plt.xlabel('$m_{n_6}$ [GeV]',fontsize=20)
    plt.ylabel(r'$\mathcal{BR}(h \to l_a l_B)$',fontsize=20)   
    plt.savefig('Widths.png',dpi=200)#_no_mni
    plt.grid(True)
    plt.show()
