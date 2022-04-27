import matplotlib.pyplot as plt

from mpmath import *
mp.dps = 80; mp.pretty = True

import numpy as np

from multiprocessing import Pool
def speedup_array(f,array,procs=4): 
    pool = Pool(procs,maxtasksperchild=100).map(f, array)
    result = np.array(list(pool))
    return result

n = 5
expmp = mp.linspace(-1,15,n)
m6np = np.array([mpf('10.0')**k for k in expmp])#np.logspace(-1,15,n)

# print(m6np)
######################################################
# Diagrams with One fermion 
######################################################
from FF_splitted import ALOneTot23_1, ALOneTot23_mni2, AROneTot23_1, AROneTot23_mni2
from FF_splitted import ALOneTot13_1, ALOneTot13_mni2, AROneTot13_1, AROneTot13_mni2
from FF_splitted import ALOneTot12_1, ALOneTot12_mni2, AROneTot12_1, AROneTot12_mni2

##############################################################
##############################################################
YLOne23_1 = speedup_array(ALOneTot23_1,m6np)
YLOne23_mni2 = speedup_array(ALOneTot23_mni2,m6np)

YROne23_1 = speedup_array(AROneTot23_1,m6np)
YROne23_mni2 = speedup_array(AROneTot23_mni2,m6np)
##############################################################
##############################################################
YLOne13_1 = speedup_array(ALOneTot13_1,m6np)
YLOne13_mni2 = speedup_array(ALOneTot13_mni2,m6np)

YROne13_1 = speedup_array(AROneTot13_1,m6np)
YROne13_mni2 = speedup_array(AROneTot13_mni2,m6np)

##############################################################
##############################################################
YLOne12_1 = speedup_array(ALOneTot12_1,m6np)
YLOne12_mni2 = speedup_array(ALOneTot12_mni2,m6np)

YROne12_1 = speedup_array(AROneTot12_1,m6np)
YROne12_mni2 = speedup_array(AROneTot12_mni2,m6np)
##############################################################
##############################################################
YLOne23 = YLOne23_1 + YLOne23_mni2
YROne23 = YROne23_1 + YROne23_mni2

YLOne13 = YLOne13_1 + YLOne13_mni2
YROne13 = YROne13_1 + YROne13_mni2

YLOne12 = YLOne12_1 + YLOne12_mni2
YROne12 = YROne12_1 + YROne12_mni2

######################################################
# Diagrams with Two fermion 
######################################################
from FF_splitted import ALTwoTot23, ARTwoTot23
from FF_splitted import ALTwoTot13, ARTwoTot13
from FF_splitted import ALTwoTot12, ARTwoTot12

YLTwo23 = speedup_array(ALTwoTot23,m6np)
YRTwo23 = speedup_array(ARTwoTot23,m6np)

YLTwo13 = speedup_array(ALTwoTot13,m6np)
YRTwo13 = speedup_array(ARTwoTot13,m6np)

YLTwo12 = speedup_array(ALTwoTot12,m6np)
YRTwo12 = speedup_array(ARTwoTot12,m6np)


######################################################
# total Form Factors
######################################################
# #a = 2, b = 3
ALtot23_1 = YLOne23_1
ARtot23_1 = YROne23_1

ALtot23_mni = YLOne23_mni2 + YLTwo23
ARtot23_mni = YROne23_mni2 + YRTwo23

ALtot23 = YLOne23_1 +  YLOne23_mni2 + YLTwo23
ARtot23 = YROne23_1 +  YROne23_mni2 + YRTwo23

# #a = 1, b = 3
ALtot13_1 = YLOne13_1
ARtot13_1 = YROne13_1

ALtot13_mni = YLOne13_mni2 + YLTwo13
ARtot13_mni = YROne13_mni2 + YRTwo13

ALtot13 = YLOne13_1 +  YLOne13_mni2 + YLTwo13
ARtot13 = YROne13_1 +  YROne13_mni2 + YRTwo13

# #a = 1, b = 3
ALtot12_1 = YLOne12_1
ARtot12_1 = YROne12_1

ALtot12_mni = YLOne12_mni2 + YLTwo12
ARtot12_mni = YROne12_mni2 + YRTwo12

ALtot12 = YLOne12_1 +  YLOne12_mni2 + YLTwo12
ARtot12 = YROne12_1 +  YROne12_mni2 + YRTwo12

######################################################
# Width decays of h to la lb
######################################################
from OneLoopLFVHD import Γhlilj
Γhl2l3_1 = Γhlilj(ALtot23_1,ARtot23_1,valores[mh],ml[2],ml[3])
Γhl2l3_mni = Γhlilj(ALtot23_mni,ARtot23_mni,valores[mh],ml[2],ml[3])
Γhl2l3 = Γhlilj(ALtot23,ARtot23,valores[mh],ml[2],ml[3])
########################################################3
#######################################################
Γhl1l3_1 = Γhlilj(ALtot13_1,ARtot13_1,valores[mh],ml[1],ml[3])
Γhl1l3_mni = Γhlilj(ALtot13_mni,ARtot13_mni,valores[mh],ml[1],ml[3])
Γhl1l3 = Γhlilj(ALtot13,ARtot13,valores[mh],ml[1],ml[3])
########################################################3
#######################################################
Γhl1l2_1 = Γhlilj(ALtot12_1,ARtot12_1,valores[mh],ml[1],ml[2])
Γhl1l2_mni = Γhlilj(ALtot12_mni,ARtot12_mni,valores[mh],ml[1],ml[2])
Γhl1l2 = Γhlilj(ALtot12,ARtot12,valores[mh],ml[1],ml[2])

#####################
####################
WidthSM = 0.0032 #GeV
Wtot_1 = Γhl2l3_1 + Γhl1l3_1 + Γhl1l2_1 + WidthSM# + YW32 + YW31 + YW21
Wtot_mni = Γhl2l3_mni + Γhl1l3_mni + Γhl1l2_mni + WidthSM# + YW32 + YW31 + YW21
Wtot = Γhl2l3 + Γhl1l3 + Γhl1l2 + WidthSM# + YW32 + YW31 + YW21

plt.figure(figsize=(15,8))
plt.loglog(np.real(m6np),abs(Γhl2l3_1 #+ YW32
                         /Wtot_1),label=r'Br($h \to \mu \tau$) with no $m_{n_i}$')

plt.loglog(np.real(m6np),abs(Γhl2l3_mni #+ YW32
                         /Wtot_mni),label=r'Br($h \to \mu \tau$) with only $m_{n_i}$',
           alpha=0.5)

plt.loglog(np.real(m6np),abs(Γhl2l3 #+ YW32
                         /Wtot),label=r'Br($h \to \mu \tau$)')

#xx = ((YW23 + YW32)/Wtot)[-1]
plt.hlines(1e-10,0.1,1e15,linestyles='-.',label=r'$1.7\times 10^{-2}$')
plt.hlines(5e-43,0.1,1e15,linestyles='--',color='b',label=r'$1\times 10^{-32}$')
plt.vlines(125.1,5e-43,1e-10,linestyles='--',color='r',label=r'$m_h$')
plt.vlines(80.379,5e-43,1e-10,linestyles='--',color='g',label=r'$m_W$')
plt.xlim(1e-1,1e15)
plt.yticks([1e-39,1e-29,1e-19,1e-9,1])
plt.xticks([1,1e4,1e8,1e12,1e16])
plt.legend(fontsize=15)
plt.show()