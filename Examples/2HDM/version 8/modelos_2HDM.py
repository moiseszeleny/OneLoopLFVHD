# Definitions od the Common factoras and xi factors for diferents 2HDM
from sympy import sin, cos, tan, cot, acos, atan, symbols

class Coeff_Higgs_model():
    
    def __init__(self,Xi_phi,etaphi,rhophi,
        Dphi, xi_lphi, xi_nphi, 
        xi_lA, xi_nA):

        self.Xi_phi = Xi_phi
        self.etaphi = etaphi
        self.rhophi = rhophi
        self.Dphi = Dphi
        self.xi_lphi = xi_lphi
        self.xi_nphi = xi_nphi
        self.xi_lA = xi_lA
        self.xi_nA = xi_nA

### Def angles alpha and beta
tb = symbols(r't_{\beta}',real=True)
cab = symbols(r'c_{\alpha\beta}',real=True)

beta = atan(tb)
alpha = atan(tb) - acos(cab) 
# cab = 0.01 # Alignment h SM-like.

bad = beta - alpha
bau = alpha + beta
a3b = alpha - 3*beta

sa = sin(alpha)
ca = cos(alpha)
sb = sin(beta)
cb = cos(beta)
tanb = tan(beta)
cotb = cot(beta)

### Type I
coeff_typeI_h = Coeff_Higgs_model(
    Xi_phi=sin(bad),
    etaphi = cos(bad),
    rhophi = cos(bau),
    Dphi = cos(a3b),
    xi_lphi = ca/sb,
    xi_nphi = ca/sb,
    xi_lA = -cotb,
    xi_nA = -cotb,
)

coeff_typeI_H = Coeff_Higgs_model(
    Xi_phi = cos(bad),
    etaphi = -sin(bad),
    rhophi = sin(bau),
    Dphi = sin(a3b),
    xi_lphi = sa/sb,
    xi_nphi = sa/sb,
    xi_lA = -cotb,
    xi_nA = -cotb,
)

### TypeII
coeff_typeII_h = Coeff_Higgs_model(
    Xi_phi = sin(bad),
    etaphi = cos(bad),
    rhophi = cos(bau),
    Dphi = cos(a3b),
    xi_lphi = -sa/cb,
    xi_nphi = ca/sb,
    xi_lA = tanb,
    xi_nA = -cotb,
)

coeff_typeII_H = Coeff_Higgs_model(
    Xi_phi = cos(bad),
    etaphi = -sin(bad),
    rhophi = sin(bau),
    Dphi = sin(a3b),
    xi_lphi = ca/cb,
    xi_nphi = sa/sb,
    xi_lA = tanb,
    xi_nA = -cotb,
)

### Lepton-specific
coeff_lepton_specific_h = Coeff_Higgs_model(
    Xi_phi = sin(bad),
    etaphi = cos(bad),
    rhophi = cos(bau),
    Dphi = cos(a3b),
    xi_lphi = -sa/cb,
    xi_nphi = -sa/cb,
    xi_lA = tanb,
    xi_nA = tanb,
)

coeff_lepton_specific_H = Coeff_Higgs_model(
    Xi_phi = cos(bad),
    etaphi = -sin(bad),
    rhophi = sin(bau),
    Dphi = sin(a3b),
    xi_lphi = ca/cb,
    xi_nphi = ca/cb,
    xi_lA = tanb,
    xi_nA = tanb,
)


### Flipped
coeff_flipped_h = Coeff_Higgs_model(
    Xi_phi = sin(bad),
    etaphi = cos(bad),
    rhophi = cos(bau),
    Dphi = cos(a3b),
    xi_lphi = ca/sb,
    xi_nphi = -sa/cb,
    xi_lA = -cotb,
    xi_nA = tanb,
)

coeff_flipped_H = Coeff_Higgs_model(
    Xi_phi = cos(bad),
    etaphi = -sin(bad),
    rhophi = sin(bau),
    Dphi = sin(a3b),
    xi_lphi = sa/sb,
    xi_nphi = ca/cb,
    xi_lA = -cotb,
    xi_nA = tanb,
)

#print(atan(tb))
#print(coeff_typeI_h.Xi_phi)