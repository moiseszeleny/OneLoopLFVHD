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

Aexp = I3 - S(1)/2*Rexp*Dagger(Rexp)
Dexp = I3 - S(1)/2*Dagger(Rexp)*Rexp
#Aexp

U11 = Aexp*Upmns
U12 = Rexp*V
U21 = -Dagger(Rexp)*Upmns
U22 = Dexp*V

####################################################

#######################################################33
#### Caso Arganda ######################################33

# Eexp = MD*MNdinv#-I*Upmns*mndsqrt*MNdsqrtinv


# v = 246 # GeV
# YN = (sqrt(2)/v)*MD

# Aexp = I3 - S(1)/2*conjugate(Eexp)*Eexp.T
# Dexp = I3 - S(1)/2*Eexp.T*conjugate(Eexp)

# Omega = Matrix(
#     [[],
#      [],
#      [],
#      [],
#      [],
#      []]
# )
# #Aexp

# U11 = Aexp*Upmns
# U12 = conjugate(Eexp)*Dexp
# U21 = -Eexp.T*Aexp*Upmns
# U22 = Dexp

####################################################
mheavy = {mn4:mn6/3,mn5:mn6/2}


from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables
d21 = Nudata.squareDm21.central*1e-18
d31 = Nudata.squareDm31.central*1e-18

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
################### NUMPY  ##############################################3
U11np = lambdify([mn1,mn6],U11.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')
U12np = lambdify([mn1,mn6],U12.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')
U21np = lambdify([mn1,mn6],U21.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')
U22np = lambdify([mn1,mn6],U22.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')

def Ununp(m1,m6,i,j):
    if i<=3 and j<=3:
        U = U11np(m1,m6)[i-1,j-1]
    elif i<=3 and j>3:
        U = U12np(m1,m6)[i-1,j-4]
    elif i>3 and j<=3:
        U = U21np(m1,m6)[i-4,j-1]
    elif i>3 and j>3:
        U = U22np(m1,m6)[i-4,j-4]
    else:
        raise ValueError('The index i and j must be in range [1,6]')
    return U

def Cijnp(m1,m6,i,j):
    from numpy import conjugate
    suma = 0
    for c in range(1,4):
        suma += Ununp(m1,m6,c,i)*conjugate(Ununp(m1,m6,c,j))
    return suma

############################## MPMATH  ############################################3
from mpmath import mp
#######3


mndsqrt_mp = lambda mn1,mn2,mn3: mp.matrix([[mp.sqrt(mn1),0,0],[0,mp.sqrt(mn2),0],[0,0,mp.sqrt(mn3)]])

MNdsqrt_mp = lambda mn4,mn5,mn6: mp.matrix([[mp.sqrt(mn4),0,0],[0,mp.sqrt(mn5),0],[0,0,mp.sqrt(mn6)]])
MNdsqrtinv_mp = lambda mn4,mn5,mn6: mp.matrix([[1/mp.sqrt(mn4),0,0],[0,1/mp.sqrt(mn5),0],[0,0,1/mp.sqrt(mn6)]])
MNdinv_mp = lambda mn4,mn5,mn6: mp.matrix([[1/mn4,0,0],[0,1/mn5,0],[0,0,1/mn6]])
############3
Upmns_mp = mp.matrix([
[ 0.821302075974486,  0.550502406897554, 0.149699699398496],
[-0.555381876513578,  0.489988544456971, 0.738576482160108],
[ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real
MD_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: 1j*Upmns_mp*MNdsqrt_mp(mn4,mn5,mn6)*mndsqrt_mp(mn1,mn2,mn3)

V_mp = mp.diag([1.0,1.0,1.0])
I3_mp =  mp.diag([1.0,1.0,1.0])

d21_mp = mp.mpf(str(Nudata.squareDm21.central))*mp.mpf('1e-18')
d31_mp = mp.mpf(str(Nudata.squareDm31.central))*mp.mpf('1e-18')
mheavy_mp = lambda mn6: [mn6/3,mn6/2]
mlight_mp = lambda mn1: [sqrt(mn1**2 + d21_mp),sqrt(mn1**2 + d31_mp)]

###################################################################################
##################################################################################3
######################### Caso Arganda   ##########################################


# Eexp_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: MD_mp(mn1,mn2,mn3,mn4,mn5,mn6)*MNdinv_mp(mn4,mn5,mn6)#
# #v = 246 # GeV
# Aexp_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: (I3_mp - mp.mpf('0.5')*Eexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).conjugate()*Eexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).transpose())

# Dexp_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: (I3_mp - mp.mpf('0.5')*Eexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).transpose()*Eexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).conjugate())
# #Aexp

# U11_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: Aexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)*Upmns_mp
# U12_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: Eexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).conjugate()*Dexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)
# U21_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: -Eexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).transpose()*Aexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)*Upmns_mp
# U22_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: Dexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)

#######################################################33
#### Caso THAO ######################################33

Rexp_mp  = lambda mn1,mn2,mn3,mn4,mn5,mn6: (MD_mp(mn1,mn2,mn3,mn4,mn5,mn6)*MNdinv_mp(mn4,mn5,mn6)).conjugate()

Aexp_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: (I3_mp - mp.mpf('0.5')*Rexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)*Rexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).transpose().conjugate())
Dexp_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: (I3_mp - mp.mpf('0.5')*Rexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).transpose().conjugate()*Rexp_mp(mn1,mn2,mn3,mn4,mn5,mn6))
#Aexp

U11_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: Aexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)*Upmns_mp
U12_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: Rexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).conjugate()*V_mp
U21_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: -Rexp_mp(mn1,mn2,mn3,mn4,mn5,mn6).transpose().conjugate()*Upmns_mp
U22_mp = lambda mn1,mn2,mn3,mn4,mn5,mn6: Dexp_mp(mn1,mn2,mn3,mn4,mn5,mn6)*V_mp

####################################################


U11mp = lambda mn1,mn6: U11_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)
U12mp = lambda mn1,mn6: U12_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)
U21mp = lambda mn1,mn6: U21_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)
U22mp = lambda mn1,mn6: U22_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)

def Unump(m1,m6,i,j):
    if i<=3 and j<=3:
        U = U11mp(m1,m6)[i-1,j-1]
    elif i<=3 and j>3:
        U = U12mp(m1,m6)[i-1,j-4]
    elif i>3 and j<=3:
        U = U21mp(m1,m6)[i-4,j-1]
    elif i>3 and j>3:
        U = U22mp(m1,m6)[i-4,j-4]
    else:
        raise ValueError('The index i and j must be in range [1,6]')
    return U

def Cijmp(m1,m6,i,j):
    from mpmath import conj
    suma = 0.0
    for c in range(1,4):
        suma += Unump(m1,m6,c,i)*conj(Unump(m1,m6,c,j))
    return suma
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
    Mi,U = mp.eighe(Mnu(m1,m2,m3,m4,m5,m6))
    return Mi,U


#######
O11mp = lambda mn1,mn6: Aexp_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)
O12mp = lambda mn1,mn6: Rexp_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)
O21mp = lambda mn1,mn6: -Rexp_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6).transpose().conjugate()
O22mp = lambda mn1,mn6: Dexp_mp(mn1,mlight_mp(mn1)[0],mlight_mp(mn1)[1],mn6/mp.mpf('3.0'),mn6/mp.mpf('2.0'),mn6)

def Omega(m1,m6,i,j):
    if i<=3 and j<=3:
        U = O11mp(m1,m6)[i-1,j-1]
    elif i<=3 and j>3:
        U = O12mp(m1,m6)[i-1,j-4]
    elif i>3 and j<=3:
        U = O21mp(m1,m6)[i-4,j-1]
    elif i>3 and j>3:
        U = O22mp(m1,m6)[i-4,j-4]
    else:
        raise ValueError('The index i and j must be in range [1,6]')
    return U

def Oijmp(m1,m6,i,j):
    from mpmath import conj
    suma = 0.0
    for c in range(1,4):
        suma += Omega(m1,m6,c,i)*conj(Omega(m1,m6,c,j))
    return suma

#####################   Omega


def Unump(m1,m6,i,j):
    if i<=3 and j<=3:
        U = U11mp(m1,m6)[i-1,j-1]
    elif i<=3 and j>3:
        U = U12mp(m1,m6)[i-1,j-4]
    elif i>3 and j<=3:
        U = U21mp(m1,m6)[i-4,j-1]
    elif i>3 and j>3:
        U = U22mp(m1,m6)[i-4,j-4]
    else:
        raise ValueError('The index i and j must be in range [1,6]')
    return U

def Cijmp(m1,m6,i,j):
    from mpmath import conj
    suma = 0.0
    for c in range(1,4):
        suma += Unump(m1,m6,c,i)*conj(Unump(m1,m6,c,j))
    return suma


## Symbolic definition
U11s = lambdify([mn1,mn6],U11.subs(Osc_data).subs(mheavy).subs(mlight).n(),'sympy')
U12s = lambdify([mn1,mn6],U12.subs(Osc_data).subs(mheavy).subs(mlight).n(),'sympy')
U21s = lambdify([mn1,mn6],U21.subs(Osc_data).subs(mheavy).subs(mlight).n(),'sympy')
U22s = lambdify([mn1,mn6],U22.subs(Osc_data).subs(mheavy).subs(mlight).n(),'sympy')

def Unusym(m1,m6,i,j):
    if i<=3 and j<=3:
        U = U11s(m1,m6)[i-1,j-1]
    elif i<=3 and j>3:
        U = U12s(m1,m6)[i-1,j-4]
    elif i>3 and j<=3:
        U = U21s(m1,m6)[i-4,j-1]
    elif i>3 and j>3:
        U = U22s(m1,m6)[i-4,j-4]
    else:
        raise ValueError('The index i and j must be in range [1,6]')
    return U

def Cijsym(m1,m6,i,j):
    from numpy import conjugate
    suma = 0
    for c in range(1,4):
        suma += Unu(m1,m6,c,i)*conjugate(Unu(m1,m6,c,j))
    return suma

from numpy import array

Unu1 = array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

def Cij1(i,j):
    from numpy import conjugate
    suma = 0
    for c in range(1,4):
        suma += Unu1(c,i)*conjugate(Unu1(c,j))
    return suma