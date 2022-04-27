from sympy import symbols,Function,pi,Abs,Matrix,conjugate,Add,ln,Symbol,lambdify,S,sqrt
from spacemathpy import GF,g,αem,numeric_substitutions,me,mmu,mtau,SMvev
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Sum mi (neutrino mass)
m1,m2,m3 = symbols('m_1,m_2,m_3',positive=True)
sum_nu = m1+m2+m3

#Square mass diferences
Δ21,Δ31 = symbols(r'\Delta_{21},\Delta_{31}',positive=True)
ms = {m2:sqrt(m1**2 + Δ21),m3:sqrt(m1**2 + Δ31)}

num = {Δ21:7.39e-23,#GeV^2
       Δ31:2.525e-21#GeV^2
      }

sumnp = lambdify([m1],sum_nu.subs(ms).subs(num),'numpy')

#####Upmns
from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization

θ12,θ13,θ23,d = symbols(r'\theta_{12},\theta_{13},\theta_{23},\delta',real=True)
U = UpmnsStandardParametrization(θ12,θ13,θ23,delta=d)
#### Lambda_k
def Deltak(mR,mI,mk):
    return ((mk)/(32*pi**2)*(mR**2/(mR**2 - mk**2)*ln(mR**2/mk**2) - 
                          mI**2/(mI**2 - mk**2)*ln(mI**2/mk**2)))

mR,mI,λ5,v,M1,M2,M3 = symbols('m_R,m_I,\lambda_5,v,M_1,M_2,M_3',positive=True)
mIexp2 = mR**2 - 2*λ5*v**2

#Yukawa base debil
Y11, Y22, Y33 = symbols('Y_{11},Y_{22},Y_{33}')
Ydebil = Matrix([[Y11,0,0],[0,Y22,0],[0,0,Y33]])
#Symbolic yukawa physics basis
YY = Ydebil*U
#### Symbolic Lambda_k
Λ1,Λ2,Λ3 = symbols(r'\Lambda_1,\Lambda_2,\Lambda_3',real=True)
cambiosΛ = {Λ1:Deltak(mR,mI,M1),Λ2:Deltak(mR,mI,M2),Λ3:Deltak(mR,mI,M3)}

#Λ = Matrix([[Λ1,0,0],[0,Λ2,0],[0,0,Λ3]])
#Mnu = Ydebil*Λ*Ydebil.T

### Yukawa debil basis despejada en terminos de mi y Lambda_k
Yf = lambda m,M: sqrt(m/M)
Yexp11 = Yf(m1,Λ1)
Yexp22 = Yf(m2,Λ2)
Yexp33 = Yf(m3,Λ3)
# relation beetwen Yukawa debil and physics basis 
Ybase = {Y11:Yexp11,Y22:Yexp22,Y33:Yexp33}
Y = YY.subs(Ybase)

#####Oscillation data
from OneLoopLFVHD.neutrinos import NuOscObservables
Oscdata = NuOscObservables()
anglesOsc = Oscdata.substitutions(th12=θ12,th13=θ13,th23=θ23)

#######cLFV
from spacemathpy import GF,g,αem,numeric_substitutions,me,mmu,mtau,SMvev
mη = symbols(r'm_\eta',positive=True)
ml = [me['symbol'],mmu['symbol'],mtau['symbol']]
e,mu,tau = symbols('e,mu,tau')
Bljliνν = symbols(r'{BR(l_j->l_i\nu\nu)}')


# Central values of BR(lj -> linunu)
B = Matrix([[0,1,0.1783],[0,0,0.1741],[0,0,0]])

#Definition of BR(lj -> li gamma)
F = Function('\mathcal{F}')
Fsym = lambda x:(1-6*x + 3*x**2 + 2*x**3 - 6*x**2*ln(x))/(6*(1-x)**4)
def Bljligamma(Mks,Y,i,j):
    sumak = Add(*[F(Mks[k]**2/mη**2)*Y[i-1,k]*conjugate(Y[j-1,k]) for k in range(len(Mks))])
    return ((3*αem['symbol']*B[i-1,j-1])/(64*pi*GF['symbol']**2*mη**4))*Abs(sumak)**2

# Definition of amu
def Delta_ali(i):
    sumak = Add(*[Abs(Y[i-1,k])**2*F(Mks[k]**2/mη**2) for k in range(len(Mks))])
    return -ml[i-1]**2/(16*pi**2*mη**2)*sumak



# Symbolic masses of heavy neutrinos
Mks = [M1,M2,M3]

# Symbolic definition of BR(mu -> e gamma)
constans =numeric_substitutions('All')
#Explicit definition of F
Fsym = lambda x:(1-6*x + 3*x**2 + 2*x**3 - 6*x**2*ln(x))/(6*(1-x)**4)

Bmue = Bljligamma(Mks,Y,1,2).subs(constans).subs(anglesOsc).replace(F,Fsym).n()

# Symbolic definition of BR(tau -> e gamma)
Btaue = Bljligamma(Mks,Y,1,3).subs(constans).subs(anglesOsc).replace(F,Fsym).n()

# Symbolic definition of BR(tau -> mu gamma)
Btaumu = Bljligamma(Mks,Y,2,3).subs(constans).subs(anglesOsc).replace(F,Fsym).n()

#Symbolic definition of Δaμ
Δaμ = Delta_ali(2).subs(constans).subs(anglesOsc).replace(F,Fsym).n()

# Dependencia de cada BR(lj -> li gamma)
#symA = Bmue.atoms(Symbol)
#print('dependencia de Bmue: \n')
#print(symA)
#symB = Btaue.atoms(Symbol)
#symC = Btaumu.atoms(Symbol)


# Numeric definition of BR(lj -> li gamma)
Bmuenp = lambdify([Mks[0],Mks[1],Mks[2],λ5,mR,mη,m1,d],
                  Bmue.subs(ms).subs(cambiosΛ).subs(num).subs(mI**2,mIexp2).subs(v,246),
                  'numpy')
Btauenp = lambdify([Mks[0],Mks[1],Mks[2],λ5,mR,mη,m1,d],
                   Btaue.subs(ms).subs(cambiosΛ).subs(num).subs(mI**2,mIexp2).subs(v,246),
                   'numpy')
Btaumunp = lambdify([Mks[0],Mks[1],Mks[2],λ5,mR,mη,m1],
                    Btaumu.subs(ms).subs(cambiosΛ).subs(num).subs(mI**2,mIexp2).subs(v,246)
                    ,'numpy')

#Numeric definition of Δaμ
Δaμnp = lambdify([Mks[0],Mks[1],Mks[2],λ5,mR,mη,m1],
                 Δaμ.subs(ms).subs(cambiosΛ).subs(num).subs(mI**2,mIexp2).subs(v,246)
                 ,'numpy')
# Upper bounds for BR(lj -> li gamma)
Bbound = np.array([[0,5.7e-13,3.3e-8],
                   [0,0,4.4e-8],
                   [0,0,0]])

######################################
# Definition of parameter space function 
###########################################
def clfv_indx(M1,M2,M3,l5,mR,meta,m1,d):
    Indxmue = Bmuenp(M1,M2,M3,l5,mR,meta,m1,d)<Bbound[0,1]
    Indxtaue = Btauenp(M1,M2,M3,l5,mR,meta,m1,d)<Bbound[0,2]
    Indxtaumu = Btaumunp(M1,M2,M3,l5,mR,meta,m1)<Bbound[1,2]
    Indxamu = np.abs(Δaμnp(M1,M2,M3,l5,mR,meta,m1))<9e-10
    Indxmue *= Indxtaue
    Indxmue *= Indxtaumu
    Indxmue *= Indxamu
    return Indxmue

def sumnu_indx(m1):
    return sumnp(m1)<0.12e-9

def scan(parameters):
    M1 = parameters['M1']
    M2 = parameters['M2']
    M3 = parameters['M3']
    l5 = parameters['l5']
    mR = parameters['mR']
    meta = parameters['meta']
    m1 = parameters['m1']
    d = parameters['delta']
    indxsum = sumnu_indx(m1)
    indx = clfv_indx(M1,M2,M3,l5,mR,meta,m1,d)
    for key in parameters.keys():
        parameters[key] = parameters[key][indx]
    return pd.DataFrame(parameters)

################################################################
# Numeric scan
################################################################


if __name__ == '__main__':
    n = S(input('n = '))
    M1np = np.random.uniform(1,1000,n)
    M2np = np.random.uniform(100,5000,n)
    M3np = np.random.uniform(100,5000,n)
    l5np = np.random.uniform(1e-12,1e-2,n)
    mRnp = np.random.uniform(100,5000,n)
    metanp= np.random.uniform(100,5000,n)
    m1np = np.random.uniform(1e-19,1e-9,n)
    dmax = 217+40
    dmin = 217-28
    dmaxrad = (dmax/180)*np.pi
    dminrad = (dmin/180)*np.pi
    dnp = np.random.uniform(dminrad,dmaxrad,n)

    parameters ={'M1':M1np,'M2':M2np,'M3':M3np,
                'l5':l5np,'mR':mRnp,'meta':metanp,
                'm1':m1np,'delta':dnp}
    
    space = scan(parameters)
    space.to_csv(r'lfv_space.txt', index=None, sep='\t')
    plt.loglog(space['l5'],space['m1'],'.')
    plt.xlabel(r'$\lambda_5$')
    plt.ylabel(r'$m_1$')
    plt.title(f"Puntos = {len(space['M1'])}")
    #space.plot(x='l5',y='m1',kind='loglog',title=f"Puntos = {len(space['M1'])}")
#plt.show()
