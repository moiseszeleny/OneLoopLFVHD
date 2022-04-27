from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization
from sympy import symbols, conjugate

θ12,θ13,θ23 = symbols(r'\theta_{12},\theta_{13},\theta_{23}',real=True)
Upmns = UpmnsStandardParametrization(θ12,θ13,θ23)

#########################################################3
########CASAS IBARRA parametrization
###########################################################
from sympy import Matrix,sqrt,I,eye,S,lambdify
mn4,mn5,mn6 = symbols('m_{n_4},m_{n_5},m_{n_6}',positive=True)
mn1,mn2,mn3 = symbols('m_{n_1},m_{n_2},m_{n3}',positive=True)
mndsqrt = Matrix([[sqrt(mn1),0,0],[0,sqrt(mn2),0],[0,0,sqrt(mn3)]])

V = eye(3,3)
I3 = eye(3,3)
MNdsqrt = Matrix([[sqrt(mn4),0,0],[0,sqrt(mn5),0],[0,0,sqrt(mn6)]])

Ynui = MNdsqrt*mndsqrt*conjugate(Upmns)


#mheavy = {mn4:mn6/3,mn5:mn6/2}

#########################################################33
# Neutrino Oscillation data
########################################################
from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables
d21 = Nudata.squareDm21.central*1e-18 #GeV^2
d31 = Nudata.squareDm31.central*1e-18 #GeV^2

#unidades en GeV^2
#d21 = 7.5e-5*1e-18 #GeV^2
#d31 = 2.457e-3*1e-18 #GeV^2
mlight ={mn2:sqrt(mn1**2 + d21),mn3:sqrt(mn1**2 + d31)}

Osc_data = Nudata().substitutions(θ12,θ13,θ23)
#from numpy import arcsin as asin
#from numpy import sqrt
#Osc_data = {θ12:asin(sqrt(0.304)),
#        θ13:asin(sqrt(0.0218)),θ23:asin(sqrt(0.452))}

#### Numeric definitions
############################## MPMATH  ############################################3
from mpmath import mp
#######3

### CASAS Ibarra parametrization 
mndsqrt_mp = lambda mn1,mn2,mn3: mp.matrix([[mp.sqrt(mn1),0,0],[0,mp.sqrt(mn2),0],[0,0,mp.sqrt(mn3)]])

MNdsqrt_mp = lambda mn4,mn5,mn6: mp.matrix([[mp.sqrt(mn4),0,0],[0,mp.sqrt(mn5),0],[0,0,mp.sqrt(mn6)]])

############3
Upmns_mp = mp.matrix([
[ 0.821302075974486,  0.550502406897554, 0.149699699398496],
[-0.555381876513578,  0.489988544456971, 0.738576482160108],
[ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real
MD_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: 1j*Upmns_mp*MNdsqrt_mp(mn4,mn5,mn6)*mndsqrt_mp(mn1,mn2,mn3)

# V_mp = mp.diag([1.0,1.0,1.0])
# I3_mp =  mp.diag([1.0,1.0,1.0])

d21_mp = mp.mpf(str(Nudata.squareDm21.central))*mp.mpf('1e-18')
d31_mp = mp.mpf(str(Nudata.squareDm31.central))*mp.mpf('1e-18')
mheavy_mp = lambda mn6: [mn6/3,mn6/2]
mlight_mp = lambda mn1: [sqrt(mn1**2 + d21_mp),sqrt(mn1**2 + d31_mp)]

####### Diagonalizando con mpmath
MD_NO = lambda m1,m2,m3,m4,m5,m6: MD_mp(m1,m2,m3,m4,m5,m6)
def Mnu(m1,m2,m3,m4,m5,m6): 
    #beta = mp.atan(tb)
    #print('doblete =',doblet)
#     v = 246 #GeV
#     if doblet==1:
#         vi = v*mp.cos(beta)
#     elif doblet==2:
#         vi = v*mp.sin(beta)
#     else:
#         raise ValueError('Doblet must be 1 or 2 depending of which Higgs doublet couples with neutrinos.')
        
    M = MD_NO(m1,m2,m3,m4,m5,m6)
    #print('MD = \n',M)
    return mp.matrix([
    [0.0,0.0,0.0,M[0,0],M[0,1],M[0,2]],
    [0.0,0.0,0.0,M[1,0],M[1,1],M[1,2]],
    [0.0,0.0,0.0,M[2,0],M[2,1],M[2,2]],
    [M[0,0],M[1,0],M[2,0],m4,0.0,0.0],
    [M[0,1],M[1,1],M[2,1],0.0,m5,0.0],
    [M[0,2],M[1,2],M[2,2],0.0,0.0,m6]
    ])

def diagonalizationMnu(m1,m2,m3,m4,m5,m6):
    Mi,U = mp.eighe(Mnu(m1,m2,m3,m4,m5,m6))
#     Mi,U = mp.eighe(Mnu(m1,m2,m3,m4,m5,m6,beta,doblet))
    return Mi,U