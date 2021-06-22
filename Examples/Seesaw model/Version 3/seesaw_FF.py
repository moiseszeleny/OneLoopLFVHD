from sympy import symbols, conjugate,I,pi,IndexedBase,sqrt
import OneLoopLFVHD as lfvhd

#####Variables definition ########
g = symbols('g',positive=True)
mW,mG = symbols('m_W,m_G',positive=True)

Uν = IndexedBase(r'{{U^\nu}}')
Uνc = IndexedBase(r'{{U^{\nu *}}}')
mn = IndexedBase(r'{{m_n}}')
#me = IndexedBase(r'{{m_e}}')
C = IndexedBase(r'C')
Cc = IndexedBase(r'{{C^*}}')
a,b,i,j = symbols('a,b,i,j',integer=True)

mh = lfvhd.ma
me = {a:lfvhd.mi,b:lfvhd.mj}
me

########### Couplings #######################
vertexhWW = lfvhd.VertexHVV(I*g*mW)
vertexhGG = lfvhd.VertexHSS((-I*g*mh**2)/(2*mW))

vertexhWG = lfvhd.VertexHVpSm(I*g/2)
vertexhGW = lfvhd.VertexHSpVm(I*g/2)

vertexneWu =lambda i,a: lfvhd.VertexVFF(0,I*g/sqrt(2)*Uν[a,i])
vertexenWd =lambda j,b: lfvhd.VertexVFF(0,I*g/sqrt(2)*Uνc[b,j])

vertexneGu = lambda i,a: lfvhd.VertexSFF((-I*g)/(sqrt(2)*mW)*me[a]*Uν[a,i],
                                         (I*g)/(sqrt(2)*mW)*mn[i]*Uν[a,i])

vertexenGd = lambda j,b: lfvhd.VertexSFF((I*g)/(sqrt(2)*mW)*mn[j]*Uνc[b,j],
                                         (-I*g)/(sqrt(2)*mW)*me[b]*Uνc[b,j])

vertexhnn = lambda i,j: lfvhd.VertexHF0F0((-I*g)/(2*mW)*(mn[j]*C[i,j] + mn[i]*Cc[i,j]),
                                          (-I*g)/(2*mW)*(mn[i]*C[i,j] + mn[j]*Cc[i,j]))

vertexhee = lambda a:lfvhd.VertexHFF((-I*g*me[a])/(2*mW))

#############Definitions#####################3
m = IndexedBase('m')
h = symbols('h');

##################### Diagrams ##############################

triangleGninj = lfvhd.TriangleSFF(vertexhnn(i,j),vertexneGu(j,b),vertexenGd(i,a),[mW,mn[i],mn[j]])

triangleWninj = lfvhd.TriangleVFF(vertexhnn(i,j),vertexneWu(j,b),vertexenWd(i,a),[mW,mn[i],mn[j]])

triangleniWW = lfvhd.TriangleFVV(vertexhWW,vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW,mW])

triangleniWG = lfvhd.TriangleFVS(vertexhWG,vertexneGu(i,b),vertexenWd(i,a),[mn[i],mW,mW])

triangleniGW = lfvhd.TriangleFSV(vertexhGW,vertexneWu(i,b),vertexenGd(i,a),[mn[i],mW,mW])

triangleniGG = lfvhd.TriangleFSS(vertexhGG,vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW,mW])

bubbleniW = lfvhd.BubbleFV(vertexhee(b),vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW])

bubbleWni = lfvhd.BubbleVF(vertexhee(a),vertexneWu(i,b),vertexenWd(i,a),[mn[i],mW])

bubbleniG = lfvhd.BubbleFS(vertexhee(b),vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW])

bubbleGni = lfvhd.BubbleSF(vertexhee(a),vertexneGu(i,b),vertexenGd(i,a),[mn[i],mW])

TrianglesTwoFermion = [triangleGninj,triangleWninj]
TrianglesOneFermion = [triangleniWW, triangleniWG,triangleniGW,triangleniGG]
Bubbles = [bubbleniW,bubbleniG,bubbleWni,bubbleGni]

DiagramsOneFermionW = [triangleniWW,bubbleniW,bubbleWni]
DiagramsOneFermionG = [triangleniWG,triangleniGW,triangleniGG,bubbleniG,bubbleGni]

#####################################################################################3
#####################################################################################
# Form factors in Unitary gauge from Lepton flavor violating Higgs boson decays in seesaw models: New discussions
B1_0 = lfvhd.B1_0
B1_1 = lfvhd.B1_1
B2_0 = lfvhd.B2_0
B2_1 = lfvhd.B2_1
B12_0 = lfvhd.B12_0
C0 = lfvhd.C0
C1 = lfvhd.C1
C2 = lfvhd.C2


AaL = - ((g**3*me[a])/(64*pi**2*mW**3))*Uν[b,i]*Uνc[a,i]*(
mn[i]**2*(B1_1(mn[i],mW) - B1_0(mn[i],mW) - B2_0(mn[i],mW)) - me[b]**2*B2_1(mn[i],mW) 
    + (2*mW**2 + mh**2)*mn[i]**2*C0(mn[i],mW,mW) - (2*mW**2*(2*mW**2 + mn[i]**2 + me[a]**2 - me[b]**2) + mn[i]**2*mh**2)*C1(mn[i],mW,mW) 
    + (2*mW**2*(me[a]**2 - mh**2) + me[b]**2*mh**2)*C2(mn[i],mW,mW)
)

AaR = - ((g**3*me[b])/(64*pi**2*mW**3))*Uν[b,i]*Uνc[a,i]*(
-mn[i]**2*(B2_1(mn[i],mW) + B1_0(mn[i],mW) + B2_0(mn[i],mW)) + me[a]**2*B1_1(mn[i],mW) 
    + (2*mW**2 + mh**2)*mn[i]**2*C0(mn[i],mW,mW) - (2*mW**2*(me[b]**2 - mh**2) + me[a]**2*mh**2)*C1(mn[i],mW,mW)
    + (2*mW**2*(2*mW**2 + mn[i]**2 - me[a]**2 + me[b]**2) + mn[i]**2*mh**2)*C2(mn[i],mW,mW)
)

AbL = - ((g**3*me[a])/(64*pi**2*mW**3))*Uν[b,j]*Uνc[a,i]*(
C[i,j]*(
    #mn[i]**2*B1_1(mW,mn[i]) + 
    mn[j]**2*B12_0(mn[i],mn[j]) - mn[j]**2*mW**2*C0(mW,mn[i],mn[j]) 
        + (2*mn[i]**2*mn[j]**2 + 2*mW**2*(mn[i]**2 + mn[j]**2) - (mn[i]**2*me[b]**2 + mn[j]**2*me[a]**2))*C1(mW,mn[i],mn[j])
)  
    + Cc[i,j]*mn[i]*mn[j]*(
        B12_0(mn[i],mn[j]) #+ B1_1(mW,mn[i]) 
        - mW**2*C0(mW,mn[i],mn[j]) 
        + (4*mW**2 + mn[i]**2 + mn[j]**2 - me[a]**2 - me[b]**2)*C1(mW,mn[i],mn[j])
    )
)

AbR = - ((g**3*me[b])/(64*pi**2*mW**3))*Uν[b,j]*Uνc[a,i]*(
C[i,j]*(
    #-mn[j]**2*B2_1(mW,mn[j]) 
    + mn[i]**2*B12_0(mn[i],mn[j]) - mn[i]**2*mW**2*C0(mW,mn[i],mn[j]) 
        - (2*mn[i]**2*mn[j]**2 + 2*mW**2*(mn[i]**2 + mn[j]**2) - (mn[i]**2*me[b]**2 + mn[j]**2*me[a]**2))*C2(mW,mn[i],mn[j])
)  
    + Cc[i,j]*mn[i]*mn[j]*(
        B12_0(mn[i],mn[j]) #- B2_1(mW,mn[j]) 
        - mW**2*C0(mW,mn[i],mn[j]) 
        - (4*mW**2 + mn[i]**2 + mn[j]**2 - me[a]**2 - me[b]**2)*C2(mW,mn[i],mn[j])
    )
)

AcdL = ((g**3*me[a])/(64*pi**2*mW**3))*Uν[b,i]*Uνc[a,i]*(me[b]**2/(me[a]**2 - me[b]**2))*(
    (2*mW**2 + mn[i]**2)*(B1_1(mn[i],mW) + B2_1(mn[i],mW)) 
    + me[a]**2*B1_1(mn[i],mW) + me[b]**2*B2_1(mn[i],mW) - 2*mn[i]**2*(B1_0(mn[i],mW) - B2_0(mn[i],mW))
)

AcdR = (me[a]/me[b])*AcdL