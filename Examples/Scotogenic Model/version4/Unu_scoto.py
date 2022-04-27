from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization

θ12,θ13,θ23 = symbols(r'\theta_{12},\theta_{13},\theta_{23}',real=True)
Upmns = UpmnsStandardParametrization(θ12,θ13,θ23)


############################## MPMATH  ############################################3
from mpmath import mp
#######3
#######################################################33
from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables
d21 = Nudata.squareDm21.central*mp.mpf("1e-18")
d31 = Nudata.squareDm31.central*mp.mpf("1e-18")

mn1 = mp.mpf("1e-12")
mn2 = lambda mn1: mp.sqrt(mn1**2 + d21)
mn3 = lambda mn1: mp.sqrt(mn1**2 + d31)

Osc_data = Nudata().substitutions(θ12,θ13,θ23)

############3
Upmns_mp = mp.matrix([
[ 0.821302075974486,  0.550502406897554, 0.149699699398496],
[-0.555381876513578,  0.489988544456971, 0.738576482160108],
[ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real



####### Diagonalizando con mpmath
def YMw(L1,L2,L3): 
    return mp.matrix([[mp.sqrt(mn1/L1),0,0],[0,mp.sqrt(mn2/L2),0],[0,0,mp.sqrt(mn3/L3)]])

def YMf(L1,L2,L3): 
    return YMw(L1,L2,L3)*Upmns_mp

def Mnu(m1,m2,m3,m4,m5,m6): 
    M = MD_NO(m1,m2,m3,m4,m5,m6)
    return mp.matrix([
    [0.0,0.0,0.0,M[0,0],M[0,1],M[0,2]],
    [0.0,0.0,0.0,M[1,0],M[1,1],M[1,2]],
    [0.0,0.0,0.0,M[2,0],M[2,1],M[2,2]],
    [M[0,0],M[1,0],M[2,0],m4,0.0,0.0],
    [M[0,1],M[1,1],M[2,1],0.0,m5,0.0],
])
    [M[0,2],M[1,2],M[2,2],0.0,0.0,m6]

def diagonalizationMnu(m1,m2,m3,m4,m5,m6):
    Mi,UL,UR = mp.eig(Mnu(m1,m2,m3,m4,m5,m6),left = True, right = True)
    Mi,UL,UR = mp.eig_sort(Mi,UL,UR)
    return Mi,UL,UR