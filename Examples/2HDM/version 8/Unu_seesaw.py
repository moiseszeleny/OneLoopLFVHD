# -*- coding: utf-8 -*-
from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization
from sympy import symbols, conjugate

θ12,θ13,θ23 = symbols(r'\theta_{12},\theta_{13},\theta_{23}',real=True)
Upmns = UpmnsStandardParametrization(θ12,θ13,θ23)

from sympy import Matrix,sqrt,I,eye,S,lambdify
mn4,mn5,mn6 = symbols('m_{n_4},m_{n_5},m_{n_6}',positive=True)
mn1,mn2,mn3 = symbols('m_{n_1},m_{n_2},m_{n3}',positive=True)
mndsqrt = Matrix([[sqrt(mn1),0,0],[0,sqrt(mn2),0],[0,0,sqrt(mn3)]])
#mndsqrt


from sympy.physics.quantum.dagger import Dagger
V = eye(3,3)
I3 = eye(3,3)
MNdsqrt = Matrix([[sqrt(mn4),0,0],[0,sqrt(mn5),0],[0,0,sqrt(mn6)]])
MNdsqrtinv = Matrix([[1/sqrt(mn4),0,0],[0,1/sqrt(mn5),0],[0,0,1/sqrt(mn6)]])
MNdinv = Matrix([[1/mn4,0,0],[0,1/mn5,0],[0,0,1/mn6]])



MD = I*conjugate(Upmns)*MNdsqrt*mndsqrt
MDT = MD.T


# ######################################################33
# ### Caso THAO ######################################33

Rexp = conjugate(MD*MNdinv)#-I*Upmns*mndsqrt*MNdsqrtinv


v = 246 # GeV
YN = (sqrt(2)/v)*MD



mheavy = {mn4:mn6/3,mn5:mn6/2}


from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables
d21 = Nudata.squareDm21.central*1e-18
d31 = Nudata.squareDm31.central*1e-18


mlight ={mn2:sqrt(mn1**2 + d21),mn3:sqrt(mn1**2 + d31)}

Osc_data = Nudata().substitutions(θ12,θ13,θ23)


############################## MPMATH  ############################################3
from mpmath import mp
#######3


mndsqrt_mp = lambda mn1,mn2,mn3: mp.matrix([[mp.sqrt(mn1),0,0],[0,mp.sqrt(mn2),0],[0,0,mp.sqrt(mn3)]])

MNdsqrt_mp = lambda mn4,mn5,mn6: mp.matrix([[mp.sqrt(mn4),0,0],[0,mp.sqrt(mn5),0],[0,0,mp.sqrt(mn6)]])
MNdsqrtinv_mp = lambda mn4,mn5,mn6: mp.matrix([[1/mp.sqrt(mn4),0,0],[0,1/mp.sqrt(mn5),0],[0,0,1/mp.sqrt(mn6)]])
MNdinv_mp = lambda mn4,mn5,mn6: mp.matrix([[1/mn4,0,0],[0,1/mn5,0],[0,0,1/mn6]])
#####################################
angle = lambda sin2: mp.asin(mp.sqrt(sin2))

th12 = angle(Nudata.sin2theta12.central)
th13 = angle(Nudata.sin2theta13.central)
th23 = angle(Nudata.sin2theta23.central)

Upmns_sp = UpmnsStandardParametrization(
    theta12=th12,theta13=th13,theta23=th23,
    delta=0.0,alpha1=0.0,alpha2=0.0
)
Upmns_mp = mp.matrix([[Upmns_sp[r,s] for r in range(3)] for s in range(3)]).T
#print(Upmns_mp)

# Upmns_mp = mp.matrix([
# [ 0.821302075974486,  0.550502406897554, 0.149699699398496],
# [-0.463050759961518,  0.489988544456971, 0.738576482160108], # Elemento 21 corregido
# [ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real


################################################################3
MD_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: 1j*Upmns_mp*MNdsqrt_mp(mn4,mn5,mn6)*mndsqrt_mp(mn1,mn2,mn3)


####### Diagonalizando con mpmath
MD_NO = lambda m1,m2,m3,m4,m5,m6: MD_mp(m1,m2,m3,m4,m5,m6)
def Mnu(m1,m2,m3,m4,m5,m6): 
    M = MD_NO(m1,m2,m3,m4,m5,m6)
    return mp.matrix([
    [0.0,0.0,0.0,M[0,0],M[0,1],M[0,2]],
    [0.0,0.0,0.0,M[1,0],M[1,1],M[1,2]],
    [0.0,0.0,0.0,M[2,0],M[2,1],M[2,2]],
    [M[0,0],M[1,0],M[2,0],m4,0.0,0.0],
    [M[0,1],M[1,1],M[2,1],0.0,m5,0.0],
    [M[0,2],M[1,2],M[2,2],0.0,0.0,m6]
])

def diagonalizationMnu(m1,m2,m3,m4,m5,m6):
    Mi,UL,UR = mp.eig(Mnu(m1,m2,m3,m4,m5,m6),left = True, right = True)
    Mi,UL,UR = mp.eig_sort(Mi,UL,UR)
    return Mi,UL,UR


############################## NUMPY  ############################################3
import numpy as np
#######3

def mndsqrt_np(mn1,mn2,mn3):
    return np.array(
        [
            [np.sqrt(mn1),0,0],
            [0,np.sqrt(mn2),0],
            [0,0,np.sqrt(mn3)]
        ]
    )

def MNdsqrt_np(mn4,mn5,mn6):
    return np.array(
        [
            [np.sqrt(mn4),0,0],
            [0,np.sqrt(mn5),0],
            [0,0,np.sqrt(mn6)]
        ]
    )

def MNdsqrtinv_np(mn4,mn5,mn6):
    return np.array(
        [
            [1/np.sqrt(mn4),0,0],
            [0,1/np.sqrt(mn5),0],
            [0,0,1/np.sqrt(mn6)]
        ]
    )

def MNdinv_np(mn4,mn5,mn6):
    return np.array(
        [mn4,mn5,mn6
            [1/mn4,0,0],
            [0,1/mn5,0],
            [0,0,1/mn6]
        ]
    )
#####################################
def angle_np(sin2):
    return np.arcsin(np.sqrt(sin2))

th12 = angle_np(Nudata.sin2theta12.central)
th13 = angle_np(Nudata.sin2theta13.central)
th23 = angle_np(Nudata.sin2theta23.central)

Upmns_sp = UpmnsStandardParametrization(
    theta12=th12,theta13=th13,theta23=th23,
    delta=0.0,alpha1=0.0,alpha2=0.0
)
Upmns_np = np.array(Upmns_sp)#.astype(np.float64)#np.array([[Upmns_sp[r,s].n() for r in range(3)] for s in range(3)]).T
#print(Upmns_np)

# Upmns_mp = mp.matrix([
# [ 0.821302075974486,  0.550502406897554, 0.149699699398496],
# [-0.463050759961518,  0.489988544456971, 0.738576482160108], # Elemento 21 corregido
# [ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real


################################################################3
def MD_np(mn1,mn2,mn3,mn4,mn5,mn6):
    return 1j*Upmns_np*MNdsqrt_np(mn4,mn5,mn6)*mndsqrt_np(mn1,mn2,mn3)


####### Diagonalizando con mpmath
MD_NO_np = lambda m1,m2,m3,m4,m5,m6: MD_np(m1,m2,m3,m4,m5,m6)
def Mnu_np(m1,m2,m3,m4,m5,m6): 
    M = MD_NO_np(m1,m2,m3,m4,m5,m6)
    return np.array([
    [0.0,0.0,0.0,M[0,0],M[0,1],M[0,2]],
    [0.0,0.0,0.0,M[1,0],M[1,1],M[1,2]],
    [0.0,0.0,0.0,M[2,0],M[2,1],M[2,2]],
    [M[0,0],M[1,0],M[2,0],m4,0.0,0.0],
    [M[0,1],M[1,1],M[2,1],0.0,m5,0.0],
    [M[0,2],M[1,2],M[2,2],0.0,0.0,m6]
])
from numpy import linalg as LA

#M = Mnu_np(1,2,3,4,5,6)
#print(LA.eig(M))
def diagonalizationMnu_np(m1,m2,m3,m4,m5,m6):
    M = Mnu_np(m1,m2,m3,m4,m5,m6)
    Mi,U = LA.eig(M)
    return Mi,U

#print(diagonalizationMnu_np(1,2,3,4,5,6))