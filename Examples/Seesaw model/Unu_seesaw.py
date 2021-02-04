from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization
from sympy import symbols

θ12,θ13,θ23 = symbols(r'\theta_{12},\theta_{13},\theta_{23}',real=True)
Upmns = UpmnsStandardParametrization(θ12,θ13,θ23)

from sympy import Matrix,sqrt,I,eye,S,lambdify
mn4,mn5,mn6 = symbols('m_{n_4},m_{n_5},m_{n_6}',positive=True)
mn1,mn2,mn3 = symbols('m_{n_1},m_{n_2},m_{n3}',positive=True)
mndsqrt = Matrix([[sqrt(mn1),0,0],[0,sqrt(mn2),0],[0,0,sqrt(mn3)]])
mndsqrt


from sympy.physics.quantum.dagger import Dagger
V = eye(3,3)
I3 = eye(3,3)
MNdsqrtinv = Matrix([[1/sqrt(mn4),0,0],[0,1/sqrt(mn5),0],[0,0,1/sqrt(mn6)]])
Rexp = -I*Upmns*mndsqrt*MNdsqrtinv 
#Rexp

Aexp = I3 - S(1)/2*Rexp*Dagger(Rexp)
Dexp = I3 - S(1)/2*Dagger(Rexp)*Rexp
#Aexp

U11 = Aexp*Upmns
U12 = Rexp*V
U21 = -Dagger(Rexp)*Upmns
U22 = Dexp*V

mheavy = {mn4:mn6/3,mn5:mn6/2}


from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables
#d21 = Nudata.squareDm21.central*1e-18
#d31 = Nudata.squareDm31.central*1e-18

d21 = 7.5e-5*1e-18
d31 = 2.457e-3*1e-18
mlight ={mn2:sqrt(mn1**2 + d21),mn3:sqrt(mn1**2 + d31)}

#Osc_data = Nudata().substitutions(θ12,θ13,θ23)
from numpy import arctan as atan
from numpy import sqrt
Osc_data = {θ12:asin(sqrt(0.304)),
        θ13:asin(sqrt(0.0218)),θ23:asin(sqrt(0.452))}

U11num = lambdify([mn1,mn6],U11.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')
U12num = lambdify([mn1,mn6],U12.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')
U21num = lambdify([mn1,mn6],U21.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')
U22num = lambdify([mn1,mn6],U22.subs(Osc_data).subs(mheavy).subs(mlight).n(),'numpy')

def Unu(m1,m6,i,j):
    if i<=3 and j<=3:
        U = U11num(m1,m6)[i-1,j-1]
    elif i<=3 and j>3:
        U = U12num(m1,m6)[i-1,j-4]
    elif i>3 and j<=3:
        U = U21num(m1,m6)[i-4,j-1]
    elif i>3 and j>3:
        U = U22num(m1,m6)[i-4,j-4]
    else:
        ValueError('The index i and j must be in range [1,6]')
    return U

def Cij(m1,m6,i,j):
    from numpy import conjugate
    suma = 0
    for c in range(1,4):
        suma += Unu(m1,m6,c,i)*conjugate(Unu(m1,m6,c,j))
    return suma