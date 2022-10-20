# Definitions od the Common factoras and xi factors for diferents 2HDM
from mpmath import sin, cos, tan, cot, acos, atan, mpf

### Def angles alpha and beta
def betaf(tb):
    return atan(tb)
def alphaf(tb,x0=mpf('0.01')): #Alignment h SM-like.
    return atan(tb) - acos(x0)
### Type I
def typeI_h(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    cotb = cot(beta)
    #tanb = mp.tan(beta)
    Xi_phi = sin(bad)
    etaphi = cos(bad)
    rhophi = cos(bau)
    Dphi = cos(a3b)
    xi_lphi = ca/sb
    xi_nphi = ca/sb
    xi_lA = -cotb
    xi_nA = -cotb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi

def typeI_H(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    cotb = cot(beta)
    #tanb = mp.tan(beta)
    Xi_phi = cos(bad)
    etaphi = -sin(bad)
    rhophi = sin(bau)
    Dphi = sin(a3b)
    xi_lphi = sa/sb
    xi_nphi = sa/sb
    xi_lA = -cotb
    xi_nA = -cotb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi

### TypeII
def typeII_h(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    cotb = cot(beta)
    tanb = tan(beta)
    Xi_phi = sin(bad)
    etaphi = cos(bad)
    rhophi = cos(bau)
    Dphi = cos(a3b)
    xi_lphi = -sa/cb
    xi_nphi = ca/sb
    xi_lA = tanb
    xi_nA = -cotb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi

def typeII_H(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    cotb = cot(beta)
    tanb = tan(beta)
    Xi_phi = cos(bad)
    etaphi = -sin(bad)
    rhophi = sin(bau)
    Dphi = sin(a3b)
    xi_lphi = ca/cb
    xi_nphi = sa/sb
    xi_lA = tanb
    xi_nA = -cotb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi

### Lepton-specific
def Lepton_specific_h(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    #cotb = mp.cot(beta)
    tanb = tan(beta)
    Xi_phi = sin(bad)
    etaphi = cos(bad)
    rhophi = cos(bau)
    Dphi = cos(a3b)
    xi_lphi = -sa/cb
    xi_nphi = -sa/cb
    xi_lA = tanb
    xi_nA = tanb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi

def Lepton_specific_H(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    #cotb = mp.cot(beta)
    tanb = tan(beta)
    Xi_phi = cos(bad)
    etaphi = -sin(bad)
    rhophi = sin(bau)
    Dphi = sin(a3b)
    xi_lphi = ca/cb
    xi_nphi = ca/cb
    xi_lA = tanb
    xi_nA = tanb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi


### Flipped
def Flipped_h(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    cotb = cot(beta)
    tanb = tan(beta)
    Xi_phi = sin(bad)
    etaphi = cos(bad)
    rhophi = cos(bau)
    Dphi = cos(a3b)
    xi_lphi = ca/sb
    xi_nphi = -sa/cb
    xi_lA = -cotb
    xi_nA = tanb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi

def Flipped_H(tb,cab):
    beta = betaf(tb)
    alpha = alphaf(tb,x0=cab)
    bad = beta -alpha
    bau = alpha + beta
    a3b = alpha - 3*beta
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    cotb = cot(beta)
    tanb = tan(beta)
    Xi_phi = cos(bad)
    etaphi = -sin(bad)
    rhophi = sin(bau)
    Dphi = sin(a3b)
    xi_lphi = sa/sb
    xi_nphi = ca/cb
    xi_lA = -cotb
    xi_nA = tanb
        
    return xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi