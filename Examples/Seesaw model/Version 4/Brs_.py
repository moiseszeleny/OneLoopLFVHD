
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
LFVHD_data = pd.read_csv("Seesaw model/Version 4/Widths.txt",sep='\t')#_no_mni
LFVHD_data_1 = pd.read_csv("Seesaw model/Version 4/Widths_no_mni.txt",sep='\t')#_no_mni
LFVHD_data_mni_one = pd.read_csv("/home/moiseszm/Escritorio/ProyectoLFVHD/LFVHD/Examples/Widths_mni_one.txt",sep='\t')#_no_mni
LFVHD_data_mni_two = pd.read_csv("/home/moiseszm/Escritorio/ProyectoLFVHD/LFVHD/Examples/Widths_mni_two.txt",sep='\t')#_no_mni


# %%
convert_to_array = lambda df,col:np.array(list(map(mpmathify,df[col])))

# %%
m6 = convert_to_array(LFVHD_data,'m6')
Whl1l2 = convert_to_array(LFVHD_data,'Whl1l2')
Whl1l3 = convert_to_array(LFVHD_data,'Whl1l3')
Whl2l3 = convert_to_array(LFVHD_data,'Whl2l3')

Whl1l2_1 = convert_to_array(LFVHD_data_1,'Whl1l2')
Whl1l3_1 = convert_to_array(LFVHD_data_1,'Whl1l3')
Whl2l3_1 = convert_to_array(LFVHD_data_1,'Whl2l3')

Whl1l2_mni_one = convert_to_array(LFVHD_data_mni_one,'Whl1l2')
Whl1l3_mni_one = convert_to_array(LFVHD_data_mni_one,'Whl1l3')
Whl2l3_mni_one = convert_to_array(LFVHD_data_mni_one,'Whl2l3')

Whl1l2_mni_two = convert_to_array(LFVHD_data_mni_two,'Whl1l2')
Whl1l3_mni_two = convert_to_array(LFVHD_data_mni_two,'Whl1l3')
Whl2l3_mni_two = convert_to_array(LFVHD_data_mni_two,'Whl2l3')

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
Whlilj_tot = Whl1l2  + Whl1l3  + Whl2l3 
Whlilj_tot_1 = Whl1l2_1  + Whl1l3_1  + Whl2l3_1 
Whlilj_tot_mni_one = Whl1l2_mni_one  + Whl1l3_mni_one  + Whl2l3_mni_one 
Whlilj_tot_mni_two = Whl1l2_mni_two  + Whl1l3_mni_two  + Whl2l3_mni_two 


# %%
WHnuN_tot = np.array([WhnuN_sum(m) for m in m6])

# %%
Wtot = (
    Whlilj_tot + 
    abs(WHnuN_tot) + 
    WidthSM
)

Wtot_1 = (
    Whlilj_tot_1 + 
    abs(WHnuN_tot) + 
    WidthSM
)

Wtot_mni_one = (
    Whlilj_tot_mni_one + 
    abs(WHnuN_tot) + 
    WidthSM
)

Wtot_mni_two = (
    Whlilj_tot_mni_two + 
    abs(WHnuN_tot) + 
    WidthSM
)
#Wtot

# ## Branching ratios

from OneLoopLFVHD.data import ml

#import numpy as np
#y = Whl1l2/Wtot
#dy = np.diff(y)/np.diff(m6)

# %%
BR_dict_l1l2 = {'name':r"$\mathcal{BR}(h \to e \mu)$",
                'width':{
                    '1':Whl1l2_1,
                    'mni_one':Whl1l2_mni_one,
                    'mni_two':Whl1l2_mni_two,
                    'total':Whl1l2
                    },
                'fname':'Widths_hl1l2_splitted'}

BR_dict_l1l3 = {'name':r"$\mathcal{BR}(h \to e \tau)$",
                'width':{
                    '1':Whl1l3_1,
                    'mni_one':Whl1l3_mni_one,
                    'mni_two':Whl1l3_mni_two,
                    'total':Whl1l3
                    },
                'fname':'Widths_hl1l3_splitted'}

BR_dict_l2l3 = {'name':r"$\mathcal{BR}(h \to \mu \tau)$",
                'width':{
                    '1':Whl2l3_1,
                    'mni_one':Whl2l3_mni_one,
                    'mni_two':Whl2l3_mni_two,
                    'total':Whl2l3
                    },
                'fname':'Widths_hl2l3_splitted'}
# BR_dict = BR_dict_l1l2
for BR_dict in [BR_dict_l2l3,BR_dict_l1l3,BR_dict_l1l2]:
    with plt.style.context('seaborn-deep'):
        plt.figure(figsize=(12,8))
        plt.loglog(m6,abs(BR_dict['width']['total']/Wtot),'--',
        label=fr'{BR_dict["name"]}',linewidth=2.5)

        #plt.loglog(m6[:-1],abs(dy),'--',
        #label=r'$\frac{d Br(h \to e \mu)}{d m_6}$',linewidth=2.5)

        plt.loglog(m6,abs(BR_dict['width']['1']/Wtot_1),
        label=fr'{BR_dict["name"]} ->1',linewidth=2.5)

        plt.loglog(m6,abs(BR_dict['width']['mni_one']/Wtot_mni_one),'--',
        label=fr'{BR_dict["name"]}'+' -> $m_{n_i}$ One',linewidth=2.5)

        plt.loglog(m6,abs(BR_dict['width']['mni_two']/Wtot_mni_two),'--',
        label=fr'{BR_dict["name"]}'+' -> $m_{n_i}$ Two',linewidth=2.5)

        #Horizontal lines
        #plt.hlines(1e-30,0.1,1e15,color='0.1',linewidth=1.5,linestyles=':',label=r'$10^{-30}$')
        #plt.hlines(1e-43,0.1,1e15,color='0.1',linewidth=1.5,linestyles='--',label=r'$1\times 10^{-43}$')
        #plt.vlines(mpf('125.1')**2,1e-10,1,label=r'$m_h^2$')
        #plt.vlines(16*(125.1 + 80.379 + 1.776)**2 ,1e-65,1e-3,label='$16 m_h^2 + 16 m_W^2$')
        
        
        #plt.xticks([1,125.1,1e4,1e8,1e12,1e15],
        #           ['1','$m_h$','$10^4$','$10^8$','$10^{12}$','$10^{15}$'],fontsize=20)
        #plt.yticks([1e-49,1e-39,1e-29,1e-19,1e-9],fontsize=20)
        #plt.xlim(1e-1,1e15)

        plt.legend(fontsize=18,ncol=1,frameon=False,bbox_to_anchor=(0.5,0.6))
        #plt.ylabel(r'BR($h \to e_a e_b$)',fontsize=18)
        plt.xlabel('$m_{n_6}$ [GeV]',fontsize=20)
        #plt.ylabel(r'$\mathcal{BR}(h \to l_a l_B)$',fontsize=20)   
        #
        plt.xlim(1e-8,1e15)
        plt.grid(True)
        plt.savefig(f'Seesaw model/Version 4/{BR_dict["fname"]}.png',dpi=200)#_no_mni
        #plt.show()
plt.show()
