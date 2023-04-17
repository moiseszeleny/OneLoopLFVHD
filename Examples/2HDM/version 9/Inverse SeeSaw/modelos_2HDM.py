# Definitions od the Common factoras and xi factors for diferents 2HDM
from sympy import sin, cos, tan, cot, acos, atan, symbols, lambdify, solve

kappaTau1=1.01
kappaTauSUP1sig=1.02+0.17
kappaTauINF1sig=1.02-0.17
kappaTauSUP2sig=1.36
kappaTauINF2sig=0.68



class Coeff_Higgs_model():
    
    def __init__(self,Xi_phi,etaphi,rhophi,
        Dphi, xi_lphi, xi_nphi, 
        xi_lA, xi_nA,name):

        self.Xi_phi = Xi_phi
        self.etaphi = etaphi
        self.rhophi = rhophi
        self.Dphi = Dphi
        self.xi_lphi = xi_lphi
        self.xi_nphi = xi_nphi
        self.xi_lA = xi_lA
        self.xi_nA = xi_nA
        self.name = name

    def cotas_tb_kappa_tau(self,cab_val):

        xi_lphi = self.xi_lphi.subs(cab, cab_val).simplify()

        cot_inf_1sig = solve(xi_lphi-kappaTauINF1sig, tb)
        cot_sup_1sig = solve(xi_lphi-kappaTauSUP1sig, tb)

        cot_inf_2sig = solve(xi_lphi-kappaTauINF2sig, tb)
        cot_sup_2sig = solve(xi_lphi-kappaTauSUP2sig, tb)

        return (cot_inf_1sig, cot_sup_1sig), (cot_inf_2sig, cot_sup_2sig)

    def kappatau_cond(self, cab_val, tb_np):

        xi_lphi = self.xi_lphi
        xi_lphif = lambdify(tb, xi_lphi.subs(cab, cab_val), 'numpy')

        xi_lphi_np = xi_lphif(tb_np)

        cond1sig = (
            (kappaTauSUP1sig > xi_lphi_np) *
            (kappaTauINF1sig < xi_lphi_np)
            )
        cond2sig = (
            ((kappaTauSUP2sig > xi_lphi_np) & (xi_lphi_np > kappaTauSUP1sig)) *
            ((kappaTauINF2sig < xi_lphi_np) & (xi_lphi_np< kappaTauINF1sig))
            )
        cond_not_allowed = (
            (kappaTauSUP2sig < xi_lphi_np) *
            (kappaTauINF2sig > xi_lphi_np)
            )
        return cond1sig, cond2sig, cond_not_allowed

    def plot_couplings(self,cab_val,save=False):
        import matplotlib.pyplot as plt
        import numpy as np

        xi_lphi = self.xi_lphi
        xi_lphif = lambdify(tb, xi_lphi.subs(cab, cab_val), 'numpy')
        xi_nphi = self.xi_nphi
        xi_nphif = lambdify(tb, xi_nphi.subs(cab, cab_val), 'numpy')

        tb_np = np.logspace(-3,3)
        tbi = 1e-3
        tbf = 1e3
        plt.semilogx(tb_np, xi_lphif(tb_np),'r',label=r'$\kappa_\tau$',alpha=0.5)
        #plt.semilogx(tb_np, xi_nphif(tb_np),'--b',label=r'$\xi^{n}$',alpha=0.5)

        # 2 sigma lines
        plt.fill_between(
            x=tb_np ,y1=kappaTauINF2sig, y2=kappaTauSUP2sig,
            alpha=0.3, color='g',
            label=r'$\kappa^\tau_{2 \sigma}$'
            )
        #plt.vlines(x=cota_sup_2sig, ymin=0.1, ymax=2, colors='g')

        # 1 sigma lines
        plt.fill_between(
            x=tb_np ,y1=kappaTauINF1sig, y2=kappaTauSUP1sig,
            alpha=0.3, color='b',
            label=r'$\kappa^\tau_{1 \sigma}$'
            )
        #plt.vlines(x=cota_sup_1sig, ymin=0.1, ymax=2, colors='b')

        plt.xlabel(r'$\tan{\beta}$', fontsize=16)
        #plt.ylabel(f'Couplings {self.name}', fontsize=16)
        plt.title(rf' $\cos(\beta - \alpha) = {cab_val}$')
        plt.xlim(tbi, tbf)
        plt.ylim(kappaTauINF2sig-0.1,2)
        plt.legend(fontsize=15, ncol=2)
        if save:
            path = 'Cotas/Kappa/'
            plt.savefig(f'couplings_{self.name}.png')
        else:
            pass
        plt.show()

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
    name = 'h-Type I' 
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
    name = 'H-Type I'
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
    name = 'h-Type II'
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
    name = 'H-Type II'
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
    name = 'h-Lepton-specific'
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
    name = 'H-Lepton-specific'
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
    name = 'h-Flipped'
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
    name = 'H-Flipped'
)

#print(atan(tb))
#print(coeff_typeI_h.Xi_phi)
if __name__ == '__main__':
    print(coeff_typeI_h.cotas_tb_kappa_tau(cab_val=0.01))

    #####################################################3
    ########################################################
    #####################################################3

    cab0 = 0.01
    coeff_typeI_h.plot_couplings(cab_val=cab0, save=True)
    # coeff_typeI_H.plot_couplings(cab_val=cab0, save=True)
    coeff_typeII_h.plot_couplings(cab_val=cab0, save=True)
    # coeff_typeII_H.plot_couplings(cab_val=cab0, save=True)
    coeff_lepton_specific_h.plot_couplings(cab_val=cab0, save=True)
    # coeff_lepton_specific_H.plot_couplings(cab_val=cab0, save=True)
    coeff_flipped_h.plot_couplings(cab_val=cab0, save=True)
    # coeff_flipped_H.plot_couplings(cab_val=cab0, save=True)

