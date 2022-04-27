#!/usr/bin/env python
# coding: utf-8

# # LFV Higgs decays in 2HDM with a seesaw type I

# In this model the couplings that allows LFVHD are given by 
# 
# | Vertex|coupling&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|Vertex|coupling&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
# |-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
# |$g_{\phi W^+ W^-}$|$ig m_W \Xi_\phi$|$g_{\phi G^+ G^-}$|$-i \frac{g m_\phi^2 \Xi_\phi}{2 m_W}$|
# |$g_{\phi G^+ W^-}$|$i \frac{g \Xi_\phi}{2}(p_+ - p_0)_\mu$|$g_{\phi W^+ G^-}$|$i \frac{g \Xi_\phi}{2}(p_0 - p_-)_\mu$|
# |$g_{\phi H^+ W^-}$|$i \frac{g \eta_\phi}{2}(k_+ - p_0)_\mu$|$g_{\phi W^+ H^-}$|$i \frac{g \eta_\phi}{2}(p_0 - k_-)_\mu$|
# | $g_{\phi H^{\pm}G^{\mp}}$|$i \frac{g \eta_\phi(m_{H^{\pm}}^2 - m_\phi^2)}{2 m_W}$|$g_{\phi H^{\pm}H^{\mp}}$| $ig \frac{\rho_\phi g_\phi - \Delta_\phi \mathcal{G}_\phi}{4 m_W \sin{2 \beta}} + i \frac{4 \lambda_5  m_W \rho_\phi}{g \sin{2 \beta}}$|
# |$g_{\phi l \overline{l}}$|$-ig \xi_\phi^{l}\frac{m_l}{2 m_W}$|$g_{\phi n_i n_j}$|$\frac{-i g \Xi_\phi}{2 m_W}\left[C_{i j}\left(P_{L} m_{n_{i}}+P_{R} m_{n_{j}}\right) \quad+C_{i j}^{*}\left(P_{L} m_{n_{j}}+P_{R} m_{n_{i}}\right)\right]$|
# |$\bar{n}_{i}e_{a} W_{\mu}^{+}$|$\frac{i g}{\sqrt{2}} U_{a i}^{\nu} \gamma^{\mu}P_{L}$|$\overline{e_{a}}n_{j}W_{\mu}^{-}$|$\frac{i g}{\sqrt{2}} U_{a j}^{\nu *}\gamma^{\mu} P_{L}$|
# |$\bar{n}_{i} e_{a} G_{W}^{+}$|$-\frac{i g}{\sqrt{2} m_{W}} U_{a i}^{\nu}\left(m_{e_{a}}P_{R}-m_{n, i} P_{L}\right)$|$\overline{e_{a}} n_{j} G_{W}^{-}$|$-\frac{i g}{\sqrt{2} m_{W}} U_{a j}^{\nu*}\left(m_{e_{a}} P_{L}-m_{n, j} P_{R}\right)$|
# |$\bar{n}_{i} e_{a} H^{+}$|$\frac{i g U^{\nu}_{a i}}{\sqrt{2} m_W}(\xi_{A}^{n}m_{n_i} P_L -  \xi_{A}^{l}m_{e_a} P_R)$|$\overline{e_{a}} n_{j} H^{-}$|$\frac{i g U^{\nu *}_{a i}}{\sqrt{2} m_W}(-  \xi_{A}^{l}m_{e_a}P_L + \xi_{A}^{n}m_{n_i} P_R)$|

# In[1]:


from sympy import symbols, init_printing, conjugate, I, pi,IndexedBase,sqrt,Add,simplify,factor,conjugate
from sympy import sin, cos, cot, tan
init_printing()


# **Assigning masses of initial and final particles**

# In[2]:


import OneLoopLFVHD as lfvhd


# **Defining symbolic variables**

# In[3]:


g = symbols('g',positive=True)
mW,mG, mHpm, mh, mH = symbols('m_W,m_G, {{m_{H^{\pm}}}}, m_h, m_H',positive=True)
Ξϕ = symbols('\Xi_{\phi}', real=True)
ξlϕ, ξnϕ, ξlA, ξnA = symbols(r'{{\xi^{l}_{\phi}}}, {{\xi^{n}_{\phi}}}, {{\xi^{l}_{A}}}, {{\xi^{n}_{A}}}', real=True)

Uν = IndexedBase(r'{{U^\nu}}')
Uνc = IndexedBase(r'{{U^{\nu *}}}')
mn = IndexedBase(r'{{m_n}}')
#me = IndexedBase(r'{{m_e}}')
C = IndexedBase(r'C')
Cc = IndexedBase(r'{{C^*}}')
a,b,i,j = symbols('a,b,i,j',integer=True)


# In[4]:


mϕ = lfvhd.ma
me = {a:lfvhd.mi,b:lfvhd.mj}
me


# **Defining vertexes**

# In[5]:


vertexϕWW = lfvhd.VertexHVV(I*g*mW*Ξϕ)
vertexϕGG = lfvhd.VertexHSS((-I*g*mϕ**2*Ξϕ)/(2*mW))

vertexϕWG = lfvhd.VertexHVpSm(I*g/2*Ξϕ)
vertexϕGW = lfvhd.VertexHSpVm(I*g/2*Ξϕ)

vertexneWu =lambda i,a: lfvhd.VertexVFF(0,I*g/sqrt(2)*Uν[a,i])
vertexenWd =lambda j,b: lfvhd.VertexVFF(0,I*g/sqrt(2)*Uνc[b,j])

vertexneGu = lambda i,a: lfvhd.VertexSFF((-I*g)/(sqrt(2)*mW)*me[a]*Uν[a,i],
                                         (I*g)/(sqrt(2)*mW)*mn[i]*Uν[a,i])

vertexenGd = lambda j,b: lfvhd.VertexSFF((I*g)/(sqrt(2)*mW)*mn[j]*Uνc[b,j],
                                         (-I*g)/(sqrt(2)*mW)*me[b]*Uνc[b,j])

##############3
Kϕ, Qϕ = symbols(r'K_{\phi},\mathcal{Q}_\phi', real=True)
ρϕ, Δϕ = symbols(r'\rho_{\phi},\Delta_\phi', real=True)
ηϕ = symbols(r'\eta_{\phi}', real=True)

λ5 = symbols(r'\lambda_5',real=True)
α, β = symbols(r'\alpha, \beta', real=True)

vertexϕHH = lfvhd.VertexHSS(I*g*(ρϕ*Kϕ - Δϕ*Qϕ)/(4*mW*sin(2*β)) + I*(4*λ5*mW*ρϕ)/(g*sin(2*β)))########

vertexϕHG = lfvhd.VertexHSS(I*(g*ηϕ*(mHpm**2 - mϕ**2))/(2*mW))

vertexϕHW = lfvhd.VertexHSpVm(I*(g*ηϕ/(2)))
vertexϕWH = lfvhd.VertexHVpSm(I*(g*ηϕ/(2)))



vertexneHu = lambda i,a: lfvhd.VertexSFF((-I*g*ξlA)/(sqrt(2)*mW)*me[a]*Uν[a,i],
                                         (I*g*ξnA)/(sqrt(2)*mW)*mn[i]*Uν[a,i])

vertexenHd = lambda j,b: lfvhd.VertexSFF((I*g*ξnA)/(sqrt(2)*mW)*mn[j]*Uνc[b,j],
                                         (-I*g*ξlA)/(sqrt(2)*mW)*me[b]*Uνc[b,j])
####################3

vertexϕnn = lambda i,j: lfvhd.VertexHF0F0((-I*g*ξnϕ)/(2*mW)*(mn[j]*C[i,j] + mn[i]*Cc[i,j]),
                                          (-I*g*ξnϕ)/(2*mW)*(mn[i]*C[i,j] + mn[j]*Cc[i,j]))

vertexϕee = lambda a:lfvhd.VertexHFF((-I*g*ξlϕ*me[a])/(2*mW))


# ## Form factors of SeeSaw model

# We are taking the diagrams of [Lepton flavor violating Higgs boson decays from massive seesaw neutrinos](https://arxiv.org/pdf/hep-ph/0407302.pdf), and we reproduce the form factor of [Lepton flavor violating Higgs boson decays in seesaw models: New discussions](https://inspirehep.net/files/b569c392f2240d487f9731316b2d5ffc)

# ### Triangle Gninj
# The **left** form factor

# In[6]:


A = g**3/(64*pi**2*mW**3) # Factor to simplify expressions
m = IndexedBase('m')
ϕ,ea,eb = symbols('\phi,e_a,e_b');
cambios = {lfvhd.ma:m[ϕ],lfvhd.mi:m[a],lfvhd.mj:m[b]}


# In[7]:


triangleGninj = lfvhd.TriangleSFF(vertexϕnn(i,j),vertexneGu(j,b),vertexenGd(i,a),[mW,mn[i],mn[j]])
AL1 = (triangleGninj.AL()/A).expand().collect([C[i,j],Cc[i,j]],simplify).collect(triangleGninj.Cs
                                                            ).simplify().subs(cambios)
AL1


# #### Divergent term

# In[8]:


DivGninjL = AL1.subs(lfvhd.cambiosDivFin(mW,mn[i],mn[j])).expand(
).collect([lfvhd.Δe],evaluate=False)[lfvhd.Δe].simplify()*lfvhd.Δe
DivGninjL 


# The **right** form factor is given by

# In[9]:


AR1 = (triangleGninj.AR()/A).expand().collect([C[i,j],Cc[i,j]],simplify).collect([lfvhd.C2(mW,mn[i],mn[j])]).simplify(
).simplify().subs(cambios)
AR1


# #### Divergent term

# In[10]:


DivGninjR = AR1.subs(lfvhd.cambiosDivFin(mW,mn[i],mn[j])).expand(
).collect([lfvhd.Δe],evaluate=False)[lfvhd.Δe].simplify()*lfvhd.Δe
DivGninjR


# ### Triangle Wninj
# 
# This is the diagram 2 of our reference 

# In[11]:


triangleWninj = lfvhd.TriangleVFF(vertexϕnn(i,j),vertexneWu(j,b),vertexenWd(i,a),[mW,mn[i],mn[j]])


# **Left form factor**

# In[12]:


AL2 = (triangleWninj.AL().subs(lfvhd.D,4)/A).expand().collect([C[i,j],Cc[i,j]],simplify).subs(
    cambios).simplify()
AL2


# **Right form factor**

# In[13]:


AR2 = (triangleWninj.AR().subs(lfvhd.D,4)/A).expand().collect([C[i,j],Cc[i,j]],simplify).subs(
    cambios).simplify()
AR2


# ### Triangle niWW
# 
# This is the diagram 3 of our reference 

# In[14]:


triangleniWW = lfvhd.TriangleFVV(vertexϕWW,vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW,mW])


# **Left form factor**

# In[15]:


AL3 = (triangleniWW.AL().subs(lfvhd.D,4)/A).subs(cambios)
AL3


# **Right form factor**

# In[16]:


(-triangleniWW.AR().subs(lfvhd.D,4)/A).subs(cambios)


# ### Triangle niWG
# 
# This is the diagram 4 of our reference

# In[17]:


triangleniWG = lfvhd.TriangleFVS(vertexϕWG,vertexneGu(i,b),vertexenWd(i,a),[mn[i],mW,mW])


# **Left form factor**

# In[18]:


AL4 = (triangleniWG.AL()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mW,mW),lfvhd.C1(mn[i],mW,mW),lfvhd.C2(mn[i],mW,mW)],simplify).simplify(
).subs(cambios)
AL4


# **Right form factor**

# In[19]:


AR4 = (triangleniWG.AR()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mW,mW),lfvhd.C1(mn[i],mW,mW),lfvhd.C2(mn[i],mW,mW)],simplify).simplify(
).subs(cambios)
AR4


# ### Triangle niGW
# 
# This is the diagram 5 of our reference

# In[20]:


triangleniGW = lfvhd.TriangleFSV(vertexϕGW,vertexneWu(i,b),vertexenGd(i,a),[mn[i],mW,mW])


# **Left form factor**

# In[21]:


AL5 = (triangleniGW.AL()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mW,mW),lfvhd.C1(mn[i],mW,mW),lfvhd.C2(mn[i],mW,mW)],simplify).simplify(
).subs(cambios)
AL5


# **Right form factor**

# In[22]:


AR5 = (triangleniGW.AR()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mW,mW),lfvhd.C1(mn[i],mW,mW),lfvhd.C2(mn[i],mW,mW)],simplify).simplify(
).subs(cambios)
AR5


# ### Triangle niGG
# 
# This is the diagram 6 of our reference

# In[23]:


triangleniGG = lfvhd.TriangleFSS(vertexϕGG,vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW,mW])


# **Left form factor**

# In[24]:


AL6 = (triangleniGG.AL()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AL6


# **Right form factor**

# In[25]:


ALR6 = (triangleniGG.AR()/A).expand().collect([mn[i]],simplify
                                  ).simplify().subs(cambios)
ALR6


# ### Bubble niW
# 
# This is the diagram 7 of our reference 

# In[26]:


bubbleniW = lfvhd.BubbleFV(vertexϕee(b),vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW])


# **Left form factor**

# In[27]:


AL7 = (bubbleniW.AL()/A).subs(lfvhd.D,4).subs(cambios)
AL7


# **Right form factor**

# In[28]:


AR7 = (bubbleniW.AR()/A).subs(lfvhd.D,4).subs(cambios)
AR7


# ### Bubble Wni
# 
# This is the diagram 9 of our reference 

# In[29]:


bubbleWni = lfvhd.BubbleVF(vertexϕee(a),vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW])


# **Left form factor**

# In[30]:


AL9 = (bubbleWni.AL()/A).subs(lfvhd.D,4).subs(cambios).simplify()
AL9


# **Right form factor**

# In[31]:


AR9 = (bubbleWni.AR()/A).subs(lfvhd.D,4).subs(cambios).simplify()
AR9


# **Adding bubble niW y Wni**

# In[32]:


BniW_L =  (bubbleniW.AL() + bubbleWni.AL()).subs(lfvhd.D,4).simplify()
BniW_L.subs(lfvhd.cambiosDivFin(mn[i],mW,mW))


# In[33]:


BniW_R =  (bubbleniW.AR() + bubbleWni.AR()).subs(lfvhd.D,4).simplify()
BniW_R.subs(lfvhd.cambiosDivFin(mn[i],mW,mW))


# ### Bubble niG
# 
# This is the diagram 8 of our reference

# In[34]:


bubbleniG = lfvhd.BubbleFS(vertexϕee(b),vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW])


# **Left form factor**

# In[35]:


AL8 = (bubbleniG.AL()/A).collect([
    lfvhd.B1_0(mn[i],mW),lfvhd.B1_1(mn[i],mW)]).simplify().subs(cambios)
AL8


# **Right form factor**

# In[36]:


AR8 = (bubbleniG.AR()/A).collect([
    lfvhd.B1_0(mn[i],mW),lfvhd.B1_1(mn[i],mW)]).simplify().subs(cambios)
AR8


# ### Bubble Gni
# 
# This is the diagram 10 of our reference

# In[37]:


bubbleGni = lfvhd.BubbleSF(vertexϕee(a),vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW])


# **Left form factor**

# In[38]:


AL10 = (bubbleGni.AL()/A).collect([
      lfvhd.B2_0(mn[i],mW),lfvhd.B2_1(mn[i],mW)]).simplify().subs(cambios)
AL10


# **Right form factor**

# In[39]:


AR10 = (bubbleGni.AR()/A).collect([
    lfvhd.B2_0(mn[i],mW),lfvhd.B2_1(mn[i],mW)]).simplify().subs(cambios)
AR10


# **Adding bubble niG y Gni**

# In[40]:


DivniGL = ((AL8 + AL10).subs(lfvhd.cambiosDivFin(mn[i],mW,mW)).expand(
).collect(lfvhd.Δe,evaluate=False)[lfvhd.Δe]*lfvhd.Δe).simplify()
DivniGL


# In[41]:


DivniGR = ((AR8 + AR10).subs(lfvhd.cambiosDivFin(mn[i],mW,mW)).expand(
).collect(lfvhd.Δe,evaluate=False)[lfvhd.Δe]*lfvhd.Δe).simplify()
DivniGR


# # Diagrams mediated by W and H

# ## Triangle Hninj
# 
# This is a new diagram

# In[42]:


triangleHninj = lfvhd.TriangleSFF(vertexϕnn(i,j),vertexneHu(j,b),vertexenHd(i,a),[mHpm,mn[i],mn[j]])
AL11 = (triangleHninj.AL()/A).expand().collect([C[i,j],Cc[i,j]],simplify).collect(triangleHninj.Cs
                                                            ).collect([ξlA,ξnA],simplify).subs(cambios)
AL11


# In[43]:


mab = m[a]**2 + m[b]**2
mij = mn[i]**2 + mn[j]**2
mab_ij = m[a]**2*mn[j]**2 + m[b]**2*mn[i]**2
Xbj = mHpm**2 + m[b]**2 + mn[j]**2
Xai = mHpm**2 + m[a]**2 + mn[i]**2
Yaij = mHpm**2*mn[i]**2 + m[a]**2*mn[j]**2 + mn[i]**2*mn[j]**2
Ybij = mHpm**2*mn[j]**2 + m[b]**2*mn[i]**2 + mn[i]**2*mn[j]**2
mabs, mijs, mab_ijs, Xbjs, Xais, Yaijs, Ybijs = symbols('m_{ab}, m_{ij}, {{m^{ab}_{ij}}}, X_{bj}, X_{ai}, Y_{aij}, Y_{bij} ',real = True)

cambiosmXY = {mab:mabs, mij:mijs, mab_ij:mab_ijs,
             Xbj:Xbjs, Xai:Xais, Yaij:Yaijs, Ybij:Ybijs}
AL11.subs(cambiosmXY)


# In[44]:


AL11body = AL11.args[2]


# In[45]:


PVHninj = [lfvhd.B12_0(mn[i],mn[j]), lfvhd.C0(mHpm,mn[i],mn[j]),
          lfvhd.C1(mHpm,mn[i],mn[j]), lfvhd.C2(mHpm,mn[i],mn[j])]

AL11coeff_Cij = AL11body.args[0].args[0]
AL11coeff_Cij.collect(PVHninj,evaluate=False)#.simplify()


# In[46]:


AL11coeff_Cijmnimnj = AL11body.args[1].args[0]
AL11coeff_Cijmnimnj.collect(PVHninj,evaluate=False)


# **Termino divergente**

# In[47]:


DivHninjL = AL11.subs(
    lfvhd.cambiosDivFin(mHpm,mn[i],mn[j])).expand(
).collect([lfvhd.Δe],evaluate=False)[lfvhd.Δe].simplify()*lfvhd.Δe
DivHninjL


# In[48]:


AR11 = (triangleHninj.AR()/A).expand().collect([C[i,j],Cc[i,j]],simplify).collect(triangleHninj.Cs
                                                            ).collect([ξlA,ξnA],simplify).subs(cambios)
AR11


# In[49]:


AR11body = AR11.args[1]
AR11body 


# In[50]:


AR11coeff_Cij = AR11body.args[0].args[0]
AR11coeff_Cij.collect(PVHninj,evaluate=False)


# In[51]:


AR11coeff_Cijmnimnj = AR11body.args[1].args[0]
AR11coeff_Cijmnimnj.collect(PVHninj,evaluate=False)


# In[52]:


DivHninjR = AR11.subs(lfvhd.cambiosDivFin(mHpm,mn[i],mn[j])).expand(
).collect([lfvhd.Δe],evaluate=False)[lfvhd.Δe].simplify()*lfvhd.Δe
DivHninjR


# ## Triangle niWH

# In[53]:


triangleniWH = lfvhd.TriangleFVS(vertexϕWH,vertexneHu(i,b),vertexenWd(i,a),[mn[i],mW,mHpm])


# In[54]:


AL12 = (triangleniWH.AL()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mW,mHpm),lfvhd.C1(mn[i],mW,mHpm),lfvhd.C2(mn[i],mW,mHpm)],simplify).simplify(
).subs(cambios)
AL12#AL12.args[0].func,AL12.args[2].func


# In[55]:


AR12 = (triangleniWH.AR()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mW,mHpm),lfvhd.C1(mn[i],mW,mHpm),lfvhd.C2(mn[i],mW,mHpm)],simplify).simplify(
).subs(cambios)
AR12


# ## Triangle niHW

# In[56]:


triangleniHW = lfvhd.TriangleFSV(vertexϕHW,vertexneWu(i,b),vertexenHd(i,a),[mn[i],mHpm,mW])


# In[57]:


AL13 = (triangleniHW.AL()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mHpm,mW),lfvhd.C1(mn[i],mHpm,mW),lfvhd.C2(mn[i],mHpm,mW)],simplify).simplify(
).subs(cambios)
AL13


# In[58]:


AR13 = (triangleniHW.AR()/A).subs(lfvhd.D,4).expand().collect(
    [lfvhd.C0(mn[i],mHpm,mW),lfvhd.C1(mn[i],mHpm,mW),lfvhd.C2(mn[i],mHpm,mW)],simplify).simplify(
).subs(cambios)
AR13


# ## Triangle niGH

# In[59]:


triangleniGH = lfvhd.TriangleFSS(vertexϕHG,vertexneHu(i,b),vertexenGd(i,a),[mn[i],mW,mHpm])


# In[60]:


AL14 = (triangleniGH.AL()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AL14


# In[61]:


AR14 = (triangleniGH.AR()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AR14


# ## Triangle niHG

# In[62]:


triangleniHG = lfvhd.TriangleFSS(vertexϕHG,vertexneGu(i,b),vertexenHd(i,a),[mn[i],mHpm,mW])


# In[63]:


AL15 = (triangleniHG.AL()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AL15


# In[64]:


AR15 = (triangleniHG.AR()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AR15


# ## Triangle niHH
# This is a new diagram 

# In[65]:


triangleniHH = lfvhd.TriangleFSS(vertexϕHH,vertexneHu(i,b),vertexenHd(i,a),[mn[i],mHpm,mHpm])


# In[66]:


AL16 = (triangleniHH.AL()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AL16


# In[67]:


AL16.expand().collect(triangleniHH.Cs,simplify).simplify()


# In[68]:


AR16 = (triangleniHH.AR()/A).expand().collect([mn[i]],simplify
                                      ).simplify().subs(cambios)
AR16


# In[69]:


AR16.expand().collect(triangleniHH.Cs,simplify).simplify()


# ### Bubble niH

# In[70]:


bubbleniH = lfvhd.BubbleFS(vertexϕee(b),vertexneHu(i,b),vertexenHd(i,a),[mn[i],mHpm])


# In[71]:


AL17 = (bubbleniH.AL()/A).collect([
    lfvhd.B1_0(mn[i],mHpm),lfvhd.B1_1(mn[i],mHpm)]).simplify().subs(cambios)
AL17


# In[72]:


AR17 = (bubbleniH.AR()/A).collect([
    lfvhd.B1_0(mn[i],mHpm),lfvhd.B1_1(mn[i],mHpm)]).simplify().subs(cambios)
AR17


# ## Triangle Hni

# In[73]:


bubbleHni = lfvhd.BubbleSF(vertexϕee(a),vertexneHu(i,b),vertexenHd(i,a),[mn[i],mHpm])


# In[74]:


AL18 = (bubbleHni.AL()/A).collect([
    lfvhd.B2_0(mn[i],mHpm),lfvhd.B2_1(mn[i],mHpm)]).simplify().subs(cambios)
AL18


# In[75]:


AR18 = (bubbleHni.AR()/A).collect([
    lfvhd.B2_0(mn[i],mHpm),lfvhd.B2_1(mn[i],mHpm)]).simplify().subs(cambios)
AR18


# **Adding bubbles niH and Hni**

# In[76]:


DivniHL = ((AL17 + AL18).subs(lfvhd.cambiosDivFin(mn[i],mHpm,mHpm)).expand(
).collect(lfvhd.Δe,evaluate=False)[lfvhd.Δe]*lfvhd.Δe).simplify()
DivniHL


# In[77]:


DivniHR = ((AR17 + AR18).subs(lfvhd.cambiosDivFin(mn[i],mHpm,mHpm)).expand(
).collect(lfvhd.Δe,evaluate=False)[lfvhd.Δe]*lfvhd.Δe).simplify()
DivniHR


# In[78]:


divAR17 = (AR17.subs(lfvhd.cambiosDivFin(mn[i],mHpm,mHpm)).expand(
).collect(lfvhd.Δe,evaluate=False)[lfvhd.Δe]*lfvhd.Δe).simplify()
divAR17


# In[79]:


divAR18 = (AR18.subs(lfvhd.cambiosDivFin(mn[i],mHpm,mHpm)).expand(
).collect(lfvhd.Δe,evaluate=False)[lfvhd.Δe]*lfvhd.Δe).simplify()
divAR18


# In[80]:


(divAR17 + divAR18).simplify()


# In[81]:


DivHninjR


# ## Working on divergencies

# In[82]:


typeI_ξh = {ξlφ:cos(α)/sin(β),ξnφ:cos(α)/sin(β),ξlA:-cot(β),ξnA:cot(β)}
typeI_ξH = {ξlφ:sin(α)/sin(β),ξnφ:sin(α)/sin(β),ξlA:-cot(β),ξnA:cot(β)}

typeII_ξh = {ξlφ:-sin(α)/cos(β),ξnφ:cos(α)/sin(β),ξlA:tan(β),ξnA:cot(β)}
typeII_ξH = {ξlφ:cos(α)/cos(β),ξnφ:sin(α)/sin(β),ξlA:tan(β),ξnA:cot(β)}

lepton_ξh = {ξlφ:-sin(α)/cos(β),ξnφ:-sin(α)/cos(β),ξlA:tan(β),ξnA:-tan(β)}
lepton_ξH = {ξlφ:cos(α)/cos(β),ξnφ:cos(α)/cos(β),ξlA:tan(β),ξnA:-tan(β)}


# In[83]:


def simplify_divXninj(Div):
    return Div.expand().subs(C[i,j],0).subs(cambios).subs({Cc[i,j]:1,Uν[b,j]:Uν[b,i],mn[j]:mn[i]})


# In[84]:


DivGR = (simplify_divXninj(DivGninjR) + DivniGR).simplify()
DivGR


# In[85]:


DivHR = (simplify_divXninj(DivHninjR) + DivniHR).simplify()
DivHR


# In the type I 2HDM we have

# In[86]:


DivGR.subs(typeI_ξh).simplify(), DivHR.subs(typeI_ξh).simplify()


# In[87]:


DivGR.subs(typeI_ξH).simplify(), DivHR.subs(typeI_ξH).simplify()


# In the type II 2HDM we have

# In[88]:


(DivGR.subs(typeII_ξh).simplify() + DivHR.subs(typeII_ξh).simplify()).simplify()


# In[89]:


(DivGR.subs(typeII_ξH).simplify() + DivHR.subs(typeII_ξH).simplify()).simplify()


# In[90]:


DivTotR = (DivGR + DivHR).simplify()
DivTotR


# In[91]:


DivGR.subs(typeII_ξH).simplify() + DivHR.subs(typeII_ξH).simplify()#).simplify()


# In[92]:


(DivGR.subs(typeII_ξh).simplify() + DivHR.subs(typeII_ξh).simplify()).simplify()


# In the Lepton-specific model we have

# In[93]:


DivGR.subs(lepton_ξh).simplify(), DivHR.subs(lepton_ξh).simplify()


# In[94]:


DivTotR.subs(lepton_ξh)


# In[ ]:




