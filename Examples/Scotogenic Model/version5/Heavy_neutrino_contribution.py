import OneLoopLFVHD as lfvhd
from sympy import symbols, conjugate,I,IndexedBase,Matrix,Add,MatrixSymbol, lambdify

# Symbolic variables

μ2,v = symbols(r'\mu_2,v',real=True)
Yν = IndexedBase(r'{{Y^\nu}}')
l,i,j = symbols('l,i,j',integer=True)

## Masses of heavy neutrino l and charged scalars
mNul,mη = symbols(r'm_{{N_l}},m_{{\eta}}',positive=True)
masasNηη = [mNul,mη,mη]

λ3 = (2/v**2)*(mη**2-μ2**2)

g,mW = symbols('g,m_W',positive=True)
vhll = lambda m:(I*g*m)/(2*mW)

# Vertexes

vertex_hηuηd = lfvhd.VertexHSS(-I*λ3*v)
vertex_ηuljNl = lfvhd.VertexSFF(0,I*Yν[l,j]) 
vertex_ηdliNl = lfvhd.VertexSFF(I*conjugate(Yν[l,i]),0)
vertex_hljlj = lfvhd.VertexHFF(vhll(lfvhd.mj))
vertex_hlili = lfvhd.VertexHFF(vhll(lfvhd.mi))

# Diagrams

TriangleNlηuηd = lfvhd.TriangleFSS(vertex_hηuηd,vertex_ηuljNl,vertex_ηdliNl,masasNηη)
BubbleNlη = lfvhd.BubbleFS(vertex_hljlj,vertex_ηuljNl,vertex_ηdliNl,[mNul,mη])
BubbleηNl = lfvhd.BubbleSF(vertex_hlili,vertex_ηuljNl,vertex_ηdliNl,[mNul,mη])