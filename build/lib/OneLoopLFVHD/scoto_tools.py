from neutrinos import MixingMatrix, UTBM_correction,parametrizationCI,UpmnsStandardParametrization
from sympy import symbols, sin, cos, sqrt,solve, pi,Eq, Matrix,log
θ,ξ,δ = symbols(r'\theta,\xi,\delta',real=True)
U = UTBM_correction(θ,ξ,0)
#print('UTBM = \n',U)

## Expresiones para las masas del neutrino
Yτ1,Yτ2,Yτ3 = symbols(r'{{Y_{\tau1}}},{{Y_{\tau2}}},{{Y_{\tau3}}}',real=True)
Λ1,Λ2,Λ3 = symbols(r'Lamda_1,Lambda_2,Lambda_3',real=True)

mnu1 = (2*Yτ1**2*Λ1)/(sin(θ)*cos(ξ)-sin(ξ))**2
mnu2 =(2*Yτ2**2*Λ2)/(cos(θ))**2
mnu3 = (2*Yτ3**2*Λ3)/(sin(θ)*sin(ξ)+cos(ξ))**2
# Valores numericos para los angulos theta y xi
t13 = cos(θ)*sin(ξ)
vals_ang = {θ:(32.89*pi/180)}
t13_v1 = t13.subs(vals_ang).n()
solxi = solve(Eq(t13_v1,sqrt(0.02241)),ξ,dict=True)
t13_v1.subs(solxi[0])
vals_ang.update(solxi[0])
mnuTBM = [mnu1.subs(vals_ang).n(),mnu2.subs(vals_ang).n(),mnu3.subs(vals_ang).n()]
UTBMnum = U.subs(vals_ang).n()
#print(UTBMnum)
def Λf(Mk,m0,l5):
    '''
    Parameters
    ----------
    Mk:int,float,symbol
        Mass of heavy neutrino k.
    m0: int,float,symbols
        Mass of the neutral scalar from second Higgs doublet in the
        approximation of l5<<1.
    l5:int,float,symbol
        Parameter lambda 5 from Higgs potential of Scotogenic model.
    
    Returns
    -------
        This function gives is associated with the one loop correction
        for neutrino masses in Scotogenic model.

    Example
    -------
    >>> Mk = 1000
    >>> m0 = 200
    >>> l5 = 1e-8
    >>> Λf(Mk,m0,l5)
    9.149552813868895e-07
    '''
    from spacemathpy import issymbolic
    if issymbolic(Mk,m0,l5):
        v =246.0 
        A = (l5*v**2)/16*pi**2
        return A*(Mk)/(m0**2-Mk**2)*(1-(Mk**2)/(m0**2-Mk**2)*log(m0**2/Mk**2))
        
    else:
         import numpy as np
         v =246.0 
         A = (l5*v**2)/16*np.pi**2
         return A*(Mk)/(m0**2-Mk**2)*(1-(Mk**2)/(m0**2-Mk**2)*np.log(m0**2/Mk**2))
#Inverse matrix 
sqrtΛ_inv = Matrix([[1/sqrt(Λ1),0,0],[0,1/sqrt(Λ2),0],[0,0,1/sqrt(Λ3)]])
# R matrix
θ1,θ2,θ3 = symbols(r'\theta_1,\theta_2,\theta_3')
c1,c2,c3 = cos(θ1), cos(θ2), cos(θ3)
s1,s2,s3 = sin(θ1), sin(θ2), sin(θ3)
R = Matrix([[c2*c3,-c1*s3-s1*s2*s3,s1*s3-c1*s2*c3],[c2*s3,c1*c3-s1*s2*s3,-s1*c3-c1*s2*s3],[s2,s1*c2,c1*c2]])
# sqrt m
mn1,mn2,mn3 = symbols(r'm_{\nu_1},m_{\nu_2},m_{\nu_3}',positive=True)
sqrtmnu = Matrix([[sqrt(mn1),0,0],[0,sqrt(mn2),0],[0,0,sqrt(mn3)]])
sqrtmnu
theta12,theta13,theta23 = symbols(r'\theta_{12},\theta_{13},\theta_{23}')
Upmns = UpmnsStandardParametrization(theta12,theta13,theta23,delta=0,alpha1=0,alpha2=0)
YCI = parametrizationCI(sqrtΛ_inv ,R,sqrtmnu,Upmns)
print('All right scoto_tools')
