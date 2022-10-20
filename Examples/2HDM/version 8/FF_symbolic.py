from sympy import symbols, conjugate,I,pi,IndexedBase,sqrt
from sympy import sin, cos, cot, tan
import OneLoopLFVHD as lfvhd

#####Variables definition ########
g = symbols('g',positive=True)
mW,mG, mHpm = symbols('m_W,m_G, {{m_{H^{\pm}}}}',positive=True)
Ξϕ = symbols('\Xi_{\phi}', real=True)
ξlϕ, ξnϕ, ξlA, ξnA = symbols(r'{{\xi^{l}_{\phi}}}, {{\xi^{n}_{\phi}}}, {{\xi^{l}_{A}}}, {{\xi^{n}_{A}}}', real=True)

Uν = IndexedBase(r'{{U^\nu}}')
Uνc = IndexedBase(r'{{U^{\nu *}}}')
mn = IndexedBase(r'{{m_n}}')
#me = IndexedBase(r'{{m_e}}')
C = IndexedBase(r'C')
Cc = IndexedBase(r'{{C^*}}')
a,b,i,j = symbols('a,b,i,j',integer=True)

mϕ = lfvhd.ma
me = {a:lfvhd.mi,b:lfvhd.mj}

########### Couplings #######################
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
Kϕ, Qϕ = symbols(r'K_{\phi},Q_\phi', real=True)
ρϕ, Δϕ = symbols(r'\rho_{\phi},\Delta_\phi', real=True)
ηϕ = symbols(r'\eta_{\phi}', real=True)

λ5 = symbols(r'\lambda_5',real=True)
α, β = symbols(r'\alpha, \beta', real=True)

vertexϕHH = lfvhd.VertexHSS(I*g*(ρϕ*Kϕ - Δϕ*Qϕ)/(4*mW*sin(2*β)) + I*(4*λ5*mW*ρϕ)/(g*sin(2*β)))########

vertexϕHG = lfvhd.VertexHSS(I*(g*ηϕ*(mHpm**2 - mϕ**2))/(2*mW))

vertexϕHW = lfvhd.VertexHSpVm(I*(g*ηϕ/(2)))
vertexϕWH = lfvhd.VertexHVpSm(I*(g*ηϕ/(2)))



vertexneHu = lambda i,a: lfvhd.VertexSFF((-I*g*ξlA)/(sqrt(2)*mW)*me[a]*Uν[a,i],#-
                                         (-I*g*ξnA)/(sqrt(2)*mW)*mn[i]*Uν[a,i])

vertexenHd = lambda j,b: lfvhd.VertexSFF((-I*g*ξnA)/(sqrt(2)*mW)*mn[j]*Uνc[b,j],
                                         (-I*g*ξlA)/(sqrt(2)*mW)*me[b]*Uνc[b,j])#-
####################3

vertexϕnn = lambda i,j: lfvhd.VertexHF0F0((-I*g*ξnϕ)/(2*mW)*(mn[j]*C[i,j] + mn[i]*Cc[i,j]),
                                          (-I*g*ξnϕ)/(2*mW)*(mn[i]*C[i,j] + mn[j]*Cc[i,j]))

vertexϕee = lambda a:lfvhd.VertexHFF((-I*g*ξlϕ*me[a])/(2*mW))

#############Definitions#####################3
A = g**3/(64*pi**2*mW**3) # Factor to simplify expressions
m = IndexedBase('m')
ϕ,ea,eb = symbols('\phi,e_a,e_b');
cambios = {lfvhd.ma:m[ϕ],lfvhd.mi:m[a],lfvhd.mj:m[b]}

##################### Diagrams ##############################

triangleGninj = lfvhd.TriangleSFF(vertexϕnn(i,j),vertexneGu(j,b),vertexenGd(i,a),[mW,mn[i],mn[j]])

triangleWninj = lfvhd.TriangleVFF(vertexϕnn(i,j),vertexneWu(j,b),vertexenWd(i,a),[mW,mn[i],mn[j]])

triangleniWW = lfvhd.TriangleFVV(vertexϕWW,vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW,mW])

triangleniWG = lfvhd.TriangleFVS(vertexϕWG,vertexneGu(i,b),vertexenWd(i,a),[mn[i],mW,mW])

triangleniGW = lfvhd.TriangleFSV(vertexϕGW,vertexneWu(i,b),vertexenGd(i,a),[mn[i],mW,mW])

triangleniGG = lfvhd.TriangleFSS(vertexϕGG,vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW,mW])

bubbleniW = lfvhd.BubbleFV(vertexϕee(b),vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW])

bubbleWni = lfvhd.BubbleVF(vertexϕee(a),vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW])

bubbleniG = lfvhd.BubbleFS(vertexϕee(b),vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW])

bubbleGni = lfvhd.BubbleSF(vertexϕee(a),vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW])

#########
triangleHninj = lfvhd.TriangleSFF(vertexϕnn(i,j),vertexneHu(j,b),vertexenHd(i,a),[mHpm,mn[i],mn[j]])

triangleniWH = lfvhd.TriangleFVS(vertexϕWH,vertexneHu(i,b),vertexenWd(i,a),[mn[i],mW,mHpm])

triangleniHW = lfvhd.TriangleFSV(vertexϕHW,vertexneWu(i,b),vertexenHd(i,a),[mn[i],mHpm,mW])

triangleniGH = lfvhd.TriangleFSS(vertexϕHG,vertexneHu(i,b),vertexenGd(i,a),[mn[i],mW,mHpm])

triangleniHG = lfvhd.TriangleFSS(vertexϕHG,vertexneGu(i,b),vertexenHd(i,a),[mn[i],mHpm,mW])

triangleniHH = lfvhd.TriangleFSS(vertexϕHH,vertexneHu(i,b),vertexenHd(i,a),[mn[i],mHpm,mHpm])

bubbleniH = lfvhd.BubbleFS(vertexϕee(b),vertexneHu(i,b),vertexenHd(i,a),[mn[i],mHpm])

bubbleHni = lfvhd.BubbleSF(vertexϕee(a),vertexneHu(i,b),vertexenHd(i,a),[mn[i],mHpm])
#####################################################################################################
##################################################################################################3
TrianglesTwoFermion = [triangleGninj,triangleWninj, triangleHninj]

TrianglesOneFermion = [triangleniWW, triangleniWG,triangleniGW,triangleniGG,
                      triangleniWH, triangleniHW, triangleniGH, triangleniHG,triangleniHH]

Bubbles = [bubbleniW,bubbleniG,bubbleWni,bubbleGni, bubbleniH, bubbleHni]

###############################################################
####### Clasification by the masses inside the loop
###############################################################
DiagramasWninj = [triangleGninj,triangleWninj]            #M0 = mW, M1 = mni, M2 = mnj ----->
DiagramasniWW = [triangleniWW,triangleniWG,triangleniGW,triangleniGG,
                 bubbleniW,bubbleniG,bubbleWni,bubbleGni] #M0 = mni, M1 = M2 = mW --->

DiagramasniWH = [triangleniWH,triangleniGH]               #M0 = mni, M1 = mW, M2 = mHpm --->
DiagramasniHW = [triangleniHW,triangleniHG]               #M0 = mni, M1 = mHpm, M2 = mW --->
DiagramasHninj = [triangleHninj]                         #M0 = mHpm, M1 = mni, M2 = mnj
DiagramasniHH = [triangleniHH,bubbleniH, bubbleHni]       #M0 = mni, M1 = mHpm, M2 = mHpm --->
#####################################################################################3
### 

