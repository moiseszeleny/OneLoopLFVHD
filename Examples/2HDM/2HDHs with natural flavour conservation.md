# Models with natural flavour conservation

Models with two Higgs doublet models and flavour conservation are given in the next table {cite}`Branco:2011iw`


|    **Model**     |  $u_{R}^{i}$   | $d_{R}^{i}$  |  $e_{R}^{i}$   |
| :------------:   | :-------------: |:------------: | :-------------: |
| Type I           | $\Phi_{2}$ | $\Phi_{2}$ | $\Phi_{2}$ | 
| Type II          | $\Phi_{2}$ | $\Phi_{1}$ | $\Phi_{1}$ | 
|  Lepton-specific | $\Phi_{2}$ | $\Phi_{2}$ | $\Phi_{1}$ | 
|      Flipped     | $\Phi_{2}$ | $\Phi_{1}$ | $\Phi_{2}$ | 

where $u_{R}^{i}$, $d_{R}^{i}$ and $e_{R}^{i}$ denote up-type, down-type quarks and charged leptons respectibely. All of this are distinguished by the Yukawa sector, in particular the way in which each Higgs doublet ($\Phi_{1,2}$) interact with fermions, as it is shown in the beforementioned table. 

## Lagrangian

We will in this work consider that there is no CP violation in the vacuum expectation values (vevs) of the scalar doublets $\Phi_{1,2}$ as a consecuance $v_{1,2}$ are reals. Thus

$$
\Phi_{j}=\left(\begin{array}{c}
\phi_{j}^{+} \\
\left(v_{j}+\rho_{j}+i \eta_{j}\right) / \sqrt{2}
\end{array}\right)
$$

with $v_1 = v \cos{\beta}$ and $v_2 = v \sin{\beta}$. 

The neutral scalar are $h$ and $H$ with $m_h < m_H$ which are orthogonal combinations of $\rho_1$ and $\rho_2$ 

```{math}
:label: hH_mix

h =\rho_{1} \sin{\alpha} -\rho_{2} \cos{\alpha}

H =-\rho_{1} \cos{\alpha} -\rho_{2} \sin{\alpha}
```

Notice that the Standard-Model Higgs boson would be

```{math}
:label: hSM

\begin{aligned}
H^{\mathrm{SM}} &=\rho_{1} \cos \beta+\rho_{2} \sin \beta \\
&=h \sin (\alpha-\beta)-H \cos (\alpha-\beta)
\end{aligned}
```

To other side, we choose $\alpha$ and $\beta $ in the first quadrant, also $v_1$ and $v_2$ non-negative without loss of generality. 

Following the notation of Aoki et. al. {cite}`Aoki:2009ha`, we define the parameters $\xi_h^f$, $\xi_H^f$, $\xi_A^f$ through the Yukawa Lagrangian

```{math}
:label: YukawaLagrangian

\begin{aligned}
\mathcal{L}_{\text {Yukawa }}^{2 \mathrm{HDM}}=&-\sum_{f=u, d, \ell} \frac{m_{f}}{v}\left(\xi_{h}^{f} \bar{f} f h+\xi_{H}^{f} \bar{f} f H-i \xi_{A}^{f} \bar{f} \gamma_{5} f A\right) \\
&-\left\{\frac{\sqrt{2} V_{u d}}{v} \bar{u}\left(m_{u} \xi_{A}^{u} \mathrm{P}_{L}+m_{d} \xi_{A}^{d} \mathrm{P}_{R}\right) d H^{+}+\frac{\sqrt{2} m_{\ell} \xi_{A}^{\ell}}{v} \overline{\nu_{L}} \ell_{R} H^{+}+\text {H.c. }\right\}
\end{aligned}
```

where $P_{L/R}$ are projection operators for left-/right-handed fermions, and the factors $\xi$ for charged leptons are presented in the table  {numref}`xi_factors`(for the case of $\xi$ factors for quarks see {cite}`Branco:2011iw`).

```{list-table}
:header-rows: 1
:name: xi_factors

* - 
  - Type I 
  - Type II
  - Lepton-specific
  - Flipped 
* - $\xi_h^l$
  - $\cos{\alpha}/\sin{\beta}$
  - $- \sin{\alpha}/\cos{\beta}$
  - $- \sin{\alpha}/\cos{\beta}$
  - $\cos{\alpha}/\sin{\beta}$
* - $\xi_H^l$
  - $\sin{\alpha}/\sin{\beta}$
  - $\cos{\alpha}/\cos{\beta}$
  - $\cos{\alpha}/\cos{\beta}$
  - $\sin{\alpha}/\sin{\beta}$
* - $\xi_A^l$
  - $-\cot{\beta}$
  - $\tan{\beta}$
  - $\tan{\beta}$
  - $-\cot{\beta}$
```

## Couplings for LFVHD


|Vertex|coupling|Vertex|coupling|
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
|$h W^{+\mu} W^{-\nu}$|$i  \frac{m_{W}}{s_W}\sin{(\beta - \alpha)} g_{\mu \nu}$|$h G_{W}^{+} G_{W}^{-}$|$\frac{-i m_h^2}{2 s_W m_W} \sin{(\beta - \alpha)}$|
|$h G_{W}^{+} {W}^{-\mu}$|$\frac{ig}{2}(p_{+}- p_0)_{\mu}$|$h G_{W}^{-} W^{+\mu}$|$\frac{i g}{2}\left(p_{0}-p_{-}\right)_{\mu}$|
|$\bar{n}_{i} e_{a} W_{\mu}^{+}$|$\frac{i g}{\sqrt{2}} U_{a i}^{\nu} \gamma^{\mu} P_{L}$|$\overline{e_{a}} n_{j} W_{\mu}^{-}$|$\frac{i g}{\sqrt{2}} U_{a j}^{\nu *} \gamma^{\mu} P_{L}$|
|$\bar{n}_{i} e_{a} G_{W}^{+}$|$-\frac{i g}{\sqrt{2} m_{W}} U_{a i}^{\nu}\left(m_{e_{a}} P_{R}-m_{n, i} P_{L}\right)$|$\overline{e_{a}} n_{j} G_{W}^{-}$|$-\frac{i g}{\sqrt{2} m_{W}} U_{a j}^{\nu *}\left(m_{e_{a}} P_{L}-m_{n, j} P_{R}\right)$|
|$h\overline{n_i}n_j$|$\frac{-i g}{2 m_W}\left[C_{i j}\left(P_{L} m_{n_{i}}+P_{R} m_{n_{j}}\right) \quad+C_{i j}^{*}\left(P_{L} m_{n_{j}}+P_{R} m_{n_{i}}\right)\right]$|$h\overline{e_a}e_a$|$\frac{-ig m_{e_a}}{2 m_W}$|