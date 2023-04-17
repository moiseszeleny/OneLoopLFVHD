'''
    Symbolic calculation of LFV Higgs decays diagrams with OneLoopLFVHD library.
'''


from sympy import symbols, I, IndexedBase, sqrt
import OneLoopLFVHD as lfvhd

# ####Variables definition ########
g = symbols('g', positive=True)
mW, mG = symbols('m_W,m_G', positive=True)

Uv = IndexedBase(r'{{U^\nu}}')
Uvc = IndexedBase(r'{{U^{\nu *}}}')
mn = IndexedBase(r'{{m_n}}')
# me = IndexedBase(r'{{m_e}}')
C = IndexedBase(r'C')
Cc = IndexedBase(r'{{C^*}}')
a, b, i, j = symbols('a,b,i,j', integer=True)

# ###### Masses of Ha and leptons
mh = lfvhd.ma
me = {a: lfvhd.mi, b: lfvhd.mj}

# ########## Couplings #######################
vertexhWW = lfvhd.VertexHVV(I*g*mW)
vertexhGG = lfvhd.VertexHSS((-I*g*mh**2)/(2*mW))

vertexhWG = lfvhd.VertexHVpSm(I*g/2)
vertexhGW = lfvhd.VertexHSpVm(I*g/2)


def vertexneWu(i, a):
    '''
        Interaction n e W+
    '''
    return lfvhd.VertexVFF(0, I*g/sqrt(2)*Uv[a, i])


def vertexenWd(j, b):
    '''
        Interaction n e W-
    '''
    return lfvhd.VertexVFF(0, I*g/sqrt(2)*Uvc[b, j])


def vertexneGu(i, a):
    '''
        Interaction n e G+
    '''
    return lfvhd.VertexSFF(
        (-I*g)/(sqrt(2)*mW)*me[a]*Uv[a, i],
        (I*g)/(sqrt(2)*mW)*mn[i]*Uv[a, i]
        )


def vertexenGd(j, b):
    '''
        Interaction n e G-
    '''
    return lfvhd.VertexSFF(
        (I*g)/(sqrt(2)*mW)*mn[j]*Uvc[b, j],
        (-I*g)/(sqrt(2)*mW)*me[b]*Uvc[b, j]
        )


def vertexhnn(i, j):
    '''
        Interaction h ni nj
    '''
    return lfvhd.VertexHF0F0(
        (-I*g)/(2*mW)*(mn[j]*C[i, j] + mn[i]*Cc[i, j]),
        (-I*g)/(2*mW)*(mn[i]*C[i, j] + mn[j]*Cc[i, j])
        )


def vertexhee(a):
    return lfvhd.VertexHFF((-I*g*me[a])/(2*mW))

# ############Definitions#####################3


m = IndexedBase('m')
h = symbols('h')

# #################### Diagrams ##############################

triangleGninj = lfvhd.TriangleSFF(
    vertexhnn(i, j), vertexneGu(j, b), vertexenGd(i, a), [mW, mn[i], mn[j]]
    )

triangleWninj = lfvhd.TriangleVFF(
    vertexhnn(i, j), vertexneWu(j, b), vertexenWd(i, a), [mW, mn[i], mn[j]]
    )

triangleniWW = lfvhd.TriangleFVV(
    vertexhWW, vertexneWu(i, b), vertexenWd(i, a), [mn[i], mW, mW]
    )

triangleniWG = lfvhd.TriangleFVS(
    vertexhWG, vertexneGu(i, b), vertexenWd(i, a), [mn[i], mW, mW]
    )

triangleniGW = lfvhd.TriangleFSV(
    vertexhGW, vertexneWu(i, b), vertexenGd(i, a), [mn[i], mW, mW]
    )

triangleniGG = lfvhd.TriangleFSS(
    vertexhGG, vertexneGu(i, b), vertexenGd(i, a), [mn[i], mW, mW]
    )

bubbleniW = lfvhd.BubbleFV(
    vertexhee(b), vertexneWu(i, b), vertexenWd(i, a), [mn[i], mW]
    )

bubbleWni = lfvhd.BubbleVF(
    vertexhee(a), vertexneWu(i, b), vertexenWd(i, a), [mn[i], mW]
    )

bubbleniG = lfvhd.BubbleFS(
    vertexhee(b), vertexneGu(i, b), vertexenGd(i, a), [mn[i], mW]
    )

bubbleGni = lfvhd.BubbleSF(
    vertexhee(a), vertexneGu(i, b), vertexenGd(i, a), [mn[i], mW]
    )

TrianglesTwoFermion = [triangleGninj, triangleWninj]
TrianglesOneFermion = [triangleniWW, triangleniWG, triangleniGW, triangleniGG]
Bubbles = [bubbleniW, bubbleniG, bubbleWni, bubbleGni]

DiagramsOneFermionW = [triangleniWW, bubbleniW, bubbleWni]
DiagramsOneFermionG = [
    triangleniWG, triangleniGW, triangleniGG,
    bubbleniG, bubbleGni
    ]
