#-*- coding: utf-8 -*-
import mpmath as mp
GF = 0.0
mW = mp.mpf('80.379')
mZ = mp.mpf('90.1')

def lambda_phi(x,y):
    return 1 + x**2 + y**2 -2*x - 2*y -2*x*y

def beta_xy(mphi,mx,my):
    return mp.sqrt(lambda_phi(mx**2/mphi**2,my**2/mphi**2))


def Wphi_ff(mphi,mf,xi_fphi,Nc):
    F = Nc*((GF*mphi*mf**2)/(4*mp.pi*mp.sqrt(2)))
    return F*xi_qphi**2*beta_q(mphi,mf,mf)**3

def Wphi_qq(mphi,mq,xi_qphi,Nc):
    return Wphi_ff(mphi,mq,xi_qphi,Nc)

def Wphi_ll(mphi,ml,xi_lphi):
    return Wphi_ff(mphi,ml,xi_lphi,1)


def g(x):
    if x>=1.0:
        out = (x-1)**mp.asin(mp.sqrt(1/x))
    else:
        out = mp.mpf('0.5')*(1-x)*(mp.log((1 + mp.sqrt(1-x))/(1 - mp.sqrt(1-x))) - 1j*mp.pi)
    return out
        
def f(x):
    if x>=1.0:
        out = mp.asin(mp.sqrt(1/x))**2
    else:
        out = -mp.mpf('0.25')*(mp.log((1 + mp.sqrt(1-x))/(1 - mp.sqrt(1-x))) - 1j*mp.pi)**2
    return out

def B0(mX,m):
    return -2*g(4*m**2/mX**2)

def C0_00(mphi,m):
    return -(2/mphi**2)*f(4*m**2/mphi**2)

def C0_0(mphi,m):
    return -(2/(mphi**2 - mZ**2))*(f(4*m**2/mphi**2) - f(4*m**2/mZ**2))

def J1(mphi,m):
    return (2*m**2/(mphi**2 - mZ**2))*(
        1 + 2m**2*C0_0(mphi,m) + mZ**2/(mphi**2 - mZ**2
                                        )*(B0(mphi,m) - B0(m)))

def J2(mphi,m):
    return m**2*C0_0(mphi,m)

def J_f(mphi,mf,xi_fphi,Nc,cfV):
    return xi_fphi*4*Nc*cfV*(J1(mphi,mf) - J2(mphi,mf))

def I_f(mphi,mf,xi_fphi,Nc):
    return - xi_fphi*Nc*(4*mf**2/mphi**2)*(2 - beta_xy(mphi,mf,mf)**2*mphi**2*(C0_00(mphi,mf)))

def Wphi_gaga(mphi,)

    
    
    

