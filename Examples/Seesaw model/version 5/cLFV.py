
##########################################################
##########################################################
########### mpmath
from mpmath import log, pi, conj
from mpmath import fsum, mpf
#######################3
###Seesaw###
#######################3

def Ggamma(x):
    a = 2*x**2 + 5*x - 1
    #print('a = ')
    t1 = x*a/(4*(1-x)**3)
    t2 = 3*x**3/(2*(1-x)**4)
    return -t1 - t2*log(x)

from Unu_seesaw import diagonalizationMnu
from OneLoopLFVHD.data import ml
def BR_lm_gammalk(m,k,Width_lm,m1,m2,m3,m4,m5,m6):
    mW = mpf('80.379')
    gw = 2*mpf('80.379')/mpf('246')
    aW = gw**2/(4*pi)#mpf('1.0')/mpf('137.035999084')
    sW2 = mpf('0.231')
    Mi,UL, UR = diagonalizationMnu(m1,m2,m3,m4,m5,m6)
    Gmk = fsum(
        [
            UL[k-1,i]*UR[m-1,i]*Ggamma(Mi[i]**2/mW**2) 
            for i in range(3,6)
        ]
        )
    #print(Gmk)
    return aW**3*sW2/(256*pi**2)*(ml[m]/mW)**4*ml[m]/Width_lm*abs(Gmk)**2

from Unu_seesaw import diagonalizationMnu_svd
def BR_lm_gammalk_svd(m,k,Width_lm,m1,m2,m3,m4,m5,m6):
    mW = mpf('80.379')
    gw = 2*mpf('80.379')/mpf('246')
    aW = gw**2/(4*pi)#mpf('1.0')/mpf('137.035999084')
    sW2 = mpf('0.231')
    Mi,UL, UR = diagonalizationMnu_svd(m1,m2,m3,m4,m5,m6)
    Gmk = fsum(
        [
            UL[k-1,i]*conj(UR[m-1,i])*Ggamma(Mi[i]**2/mW**2) 
            for i in range(3)
        ]
        )
    #print(Gmk)
    return aW**3*sW2/(256*pi**2)*(ml[m]/mW)**4*ml[m]/Width_lm*abs(Gmk)**2


#######################3
###ISS###
#######################3

from Unu_seesaw_ISS import diagonalizationMnu_ISS
def BR_lm_gammalk_ISS(m,k,Width_lm,m1,m2,m3,M1,M2,M3,mu1,mu2,mu3):
    mW = mpf('80.379')
    gw = 2*mpf('80.379')/mpf('246')
    aW = gw**2/(4*pi)
    sW2 = mpf('0.231')
    Mi,UL, UR = diagonalizationMnu_ISS(m1,m2,m3,M1,M2,M3,mu1,mu2,mu3)
    Gmk = fsum(
        [
            UL[k-1,i]*UR[m-1,i]*Ggamma(Mi[i]**2/mW**2) 
            for i in range(len(Mi))##############################
        ]
        )
    #print(Gmk)
    return aW**3*sW2/(256*pi**2)*(ml[m]/mW)**4*ml[m]/Width_lm*abs(Gmk)**2

from Unu_seesaw_ISS import diagonalizationMnu_ISS_svd
def BR_lm_gammalk_ISS_svd(m,k,Width_lm,m1,m2,m3,M1,M2,M3,mu1,mu2,mu3):
    mW = mpf('80.379')
    gw = 2*mpf('80.379')/mpf('246')
    aW = gw**2/(4*pi)
    sW2 = mpf('0.231')
    Mi,UL, UR = diagonalizationMnu_ISS_svd(m1,m2,m3,M1,M2,M3,mu1,mu2,mu3)
    Gmk = fsum(
        [
            UL[k-1,i]*conj(UR[m-1,i])*Ggamma(Mi[i]**2/mW**2) 
            for i in range(len(Mi)-3)
        ]
        )
    #print(Gmk)
    return aW**3*sW2/(256*pi**2)*(ml[m]/mW)**4*ml[m]/Width_lm*abs(Gmk)**2
##########################################################
##########################################################
########### numpy

from numpy import log as log_np
from numpy import pi as pi_np
from numpy import conjugate, array

from numpy import sum as sum_np
def Ggamma_np(x):
    a = 2*x**2 + 5*x - 1
    #print('a = ')
    t1 = x*a/(4*(1-x)**3)
    t2 = 3*x**3/(2*(1-x)**4)
    return -t1 - t2*log_np(x)

from Unu_seesaw import diagonalizationMnu_sp
from OneLoopLFVHD.data import ml_np
def BR_lm_gammalk_sp(m,k,Width_lm,m1,m2,m3,m4,m5,m6):
    mW = 80.379
    gw = 2*80.379/246
    aW = gw**2/(4*pi_np)#1.0/137.035999084
    sW2 = 0.231
    Mi,UL, UR = diagonalizationMnu_sp(m1,m2,m3,m4,m5,m6)
    Gmk = sum_np(array(
        [
            UL[k-1,i]*UR[m-1,i]*Ggamma_np(Mi[i]**2/mW**2) 
            for i in range(6)
        ])
        )
    #print(Gmk)
    return aW**3*sW2/(256*pi_np**2)*(ml_np[m]/mW)**4*ml_np[m]/Width_lm*abs(Gmk)**2

from Unu_seesaw import diagonalizationMnu_sp_svd
from OneLoopLFVHD.data import ml_np
def BR_lm_gammalk_sp_svd(m,k,Width_lm,m1,m2,m3,m4,m5,m6):
    mW = 80.379
    gw = 2*80.379/246
    aW = gw**2/(4*pi_np)#1.0/137.035999084
    sW2 = 0.231
    Mi,UL, UR = diagonalizationMnu_sp_svd(m1,m2,m3,m4,m5,m6)
    Gmk = sum_np(array(
        [
            UL[k-1,i]*UR[m-1,i]*Ggamma_np(Mi[i]**2/mW**2) 
            for i in range(3)
        ])
        )
    #print(Gmk)
    return aW**3*sW2/(256*pi_np**2)*(ml_np[m]/mW)**4*ml_np[m]/Width_lm*abs(Gmk)**2