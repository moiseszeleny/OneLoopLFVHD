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


#######################################################33
#### Caso THAO ######################################33

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
############3

#####################################
angle = lambda sin2: mp.asin(mp.sqrt(sin2))

th12 = angle(Nudata.sin2theta12.central)
th13 = angle(Nudata.sin2theta13.central)
th23 = angle(Nudata.sin2theta23.central)

Upmns_mp = UpmnsStandardParametrization(
    theta12=th12,theta13=th13,theta23=th23,
    delta=0.0,alpha1=0.0,alpha2=0.0
)
Upmns_mp = mp.matrix([[Upmns_mp[r,s] for r in range(3)] for s in range(3)]).T


# Upmns_mp = mp.matrix([
# [ 0.821302075974486,  0.550502406897554, 0.149699699398496],
# [-0.463050759961518,  0.489988544456971, 0.738576482160108], # Elemento 21 corregido
# [ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real


#############################################################3
MD_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: 1j*Upmns_mp*MNdsqrt_mp(mn4,mn5,mn6)*mndsqrt_mp(mn1,mn2,mn3)


####### Diagonalizando con mpmath
MD_NO = lambda m1,m2,m3,m4,m5,m6: MD_mp(m1,m2,m3,m4,m5,m6)

def Ynu(m1,m2,m3,m4,m5,m6):
    v = mp.mpf('246')
    return MD_NO(m1,m2,m3,m4,m5,m6)*(mp.sqrt('2')/v)
    

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

def diagonalizationMnu_svd(m1,m2,m3,m4,m5,m6):
    UL,Mi,UR = mp.svd_c(Mnu(m1,m2,m3,m4,m5,m6))
    return Mi,UL,UR

############################## NUMPY  ############################################3
import numpy as np
#######3

mndsqrt_np = lambda mn1,mn2,mn3: np.array([[np.sqrt(mn1),0,0],[0,np.sqrt(mn2),0],[0,0,np.sqrt(mn3)]])

MNdsqrt_np = lambda mn4,mn5,mn6: np.array([[np.sqrt(mn4),0,0],[0,np.sqrt(mn5),0],[0,0,np.sqrt(mn6)]])
MNdsqrtinv_np = lambda mn4,mn5,mn6: np.array([[1/np.sqrt(mn4),0,0],[0,1/np.sqrt(mn5),0],[0,0,1/np.sqrt(mn6)]])
MNdinv_np = lambda mn4,mn5,mn6: np.array([[1/mn4,0,0],[0,1/mn5,0],[0,0,1/mn6]])
#####################################
angle_np = lambda sin2: np.arcsin(np.sqrt(sin2))

th12 = angle_np(Nudata.sin2theta12.central)
th13 = angle_np(Nudata.sin2theta13.central)
th23 = angle_np(Nudata.sin2theta23.central)

Upmns_sp = UpmnsStandardParametrization(
    theta12=th12,theta13=th13,theta23=th23,
    delta=0.0,alpha1=0.0,alpha2=0.0
)
Upmns_np = np.array(Upmns_sp).astype(np.float64)#np.array([[Upmns_sp[r,s].n() for r in range(3)] for s in range(3)]).T
#print(Upmns_np)

# Upmns_mp = mp.matrix([
# [ 0.821302075974486,  0.550502406897554, 0.149699699398496],
# [-0.463050759961518,  0.489988544456971, 0.738576482160108], # Elemento 21 corregido
# [ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real


################################################################3
MD_np = lambda mn1,mn2,mn3,mn4,mn5,mn6: 1j*Upmns_np*MNdsqrt_np(mn4,mn5,mn6)*mndsqrt_np(mn1,mn2,mn3)


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
    M = Mnu_np(1,2,3,4,5,6)
    Mi,U = LA.eig(M)
    return Mi,U
    
from scipy.linalg import eig as eig_sp
def diagonalizationMnu_sp(m1,m2,m3,m4,m5,m6):
    M = Mnu_np(1,2,3,4,5,6)
    Mi,UL,UR = eig_sp(M,left=True, right=True)
    return Mi,UL,UR

from scipy.linalg import svd as svd_sp
def diagonalizationMnu_sp_svd(m1,m2,m3,m4,m5,m6):
    M = Mnu_np(1,2,3,4,5,6)
    UL,Mi,UR = svd_sp(M)
    return Mi,UL,UR

#print(diagonalizationMnu_np(1,2,3,4,5,6))

if __name__=='__main__':
    
    m1 = mp.mpf('1e-12')  #GeV 
    m2 = mp.sqrt(m1**2 + d21)
    m3 = mp.sqrt(m1**2 + d31)
    m4 = lambda m6: m6/3
    m5 = lambda m6: m6/2
    
    import numpy as np
    import matplotlib.pyplot as plt
    M = np.logspace(-1.0,18.0,num=100)
    Y01_2 = []
    Y02_2 = []
    Y12_2 = []
    for m in M:
        Y01_2.append(abs(Ynu(m1,m2,m3,m4(m),m5(m),m)[0,1])**2)
        Y02_2.append(abs(Ynu(m1,m2,m3,m4(m),m5(m),m)[0,2])**2)
        Y12_2.append(abs(Ynu(m1,m2,m3,m4(m),m5(m),m)[1,2])**2)
        
    Y01_2 = np.array(Y01_2)
    Y02_2 = np.array(Y02_2)
    Y12_2 = np.array(Y12_2)
    
    
    #print(Y01_2)
    cond_01 = Y01_2<1.5*4*np.pi
    cond_02 = Y02_2<1.5*4*np.pi
    cond_12 = Y12_2<1.5*4*np.pi
    
    plt.figure(figsize=(10,8))
    plt.loglog(M,Y01_2,label=r'Y01 total',alpha=0.5)
    plt.loglog(M,Y02_2,label=r'Y02 total',alpha=0.5)
    plt.loglog(M,Y12_2,label=r'Y12 total',alpha=0.5)
    
    
    #print(Y01_2[cond_01])
    plt.loglog(M[cond_01],Y01_2[cond_01],label=r'Y01 perturvatividad')
    plt.loglog(M[cond_02],Y02_2[cond_02],label=r'Y02 perturvatividad')
    plt.loglog(M[cond_12],Y12_2[cond_12],label=r'Y12 perturvatividad')
    
    plt.legend()
    plt.show()
    