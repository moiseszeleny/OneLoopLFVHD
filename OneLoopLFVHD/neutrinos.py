from sympy import symbols,sqrt,Matrix,cos,sin,exp,I,im,conjugate,asin

############################################################################
# Mixing Matrix
############################################################################
class MixingMatrix(Matrix):
    '''
    Class that represents a leptonic mixing matrix
    
    Atributtes
    ----------
    This class is a subclass of sympy Matrix, then has the same atributes
    
    Methods
    -------
    eval(**Kwargs)
        kwargs: dict
        Return the mixing matrix evaluates in the values given in kwargs
    square_sin12()
        Return the expresion equivalent to sin(theta_12) from symmetric
        parametrization of U_PMNS
    square_sin23()
        Return the expresion equivalent to sin(theta_23) from symmetric
        parametrization of U_PMNS
    square_sin13()
        Return the expresion equivalent to sin(theta_13) from symmetric
        parametrization of U_PMNS
    JCP()
        Return Jarskog invariant
    sin_dcp()
        Return sin(dcp)
    '''
    def eval(self,**kwargs):
        '''
        Parameters
        ----------
        kwargs: dict

        Return the mixing matrix evaluates in the values given in kwargs
        '''
        return MixingMatrix(self.subs(kwargs))

    def square_sin12(self):
        '''
        Return the expresion equivalent to sin(theta_12) from symmetric
        parametrization of U_PMNS
        '''
        return abs(self[0,1])**2/(1-abs(self[0,2])**2)
    
    def square_sin23(self):
        '''
        Return the expresion equivalent to sin(theta_23) from symmetric
        parametrization of U_PMNS
        '''
        return abs(self[1,2])**2/(1-abs(self[0,2])**2)
    
    def square_sin13(self):
        '''
        Return the expresion equivalent to sin(theta_13) from symmetric
        parametrization of U_PMNS
        '''
        return abs(self[0,2])**2
    
    def JCP(self):
        '''
        Return Jarskog invariant
        '''
        return im(conjugate(self[0,0])*conjugate(self[1,2])*self[0,2]*self[1,0])
    
    def sin_dcp(self):
        '''
        Return sin(dcp)
        '''
        return self.JCP()*(1-abs(self[0,2])**2)/(abs(self[0,0])*abs(self[0,1])*abs(self[0,2])*abs(self[1,2])*abs(self[2,2]))
    
    #def Pmue(self):
    #    return Abs(2*self[1,2].conjugate()*self[0,2]*sin(factor(Delta[(3,1)]))*exp(-I*factor(Delta[(3,2)])) +\
    #               2*self[1,1].conjugate()*self[0,1]*sin(factor(Delta[(2,1)])))**2

############################################################################
# TBM correction
############################################################################
def UpmnsStandardParametrization(theta12,theta13,theta23,delta=0,alpha1=0,alpha2=0):
    '''
    Parameters
    ----------
    theta12,theta13,theta23: int,float,symbol
        Each of this are the angles of the standard parametrization of
        leptonic mixing matrix U_PMNS
    delta,alpha1,alpha2: int,float,symbol
        Each of this are the phases of the standard parametrization of
        leptonic mixing matrix U_PMNS. Dirac phase correspond to delta
        and alpha1 and alpha2 are Majorana phases. By default all phases
        are equal to zero.

    Returns the leptonic mixing matrix U_PMNS in the standard parametrization.
    ------
    '''
    c12,c13,c23 = cos(theta12), cos(theta13), cos(theta23)
    s12,s13,s23 = sin(theta12), sin(theta13), sin(theta23)
    UPMNS = Matrix(
        [[exp(I*(alpha1/2))*c12*c13,exp(I*(alpha2/2))*c13*s12,exp(-I*delta)*s13],
    [exp(I*(alpha1/2))*(-c23*s12-c12*s13*s23), exp(I*(alpha2/2))*(c12*c23 - s12*s13*s23), c13*s23],##corrección in U_21
    [exp(I*(alpha1/2))*(-c12*c23*s13 + s12*s23), exp(I*(alpha2/2))*(-c23*s12*s13-c12*s23), c13*c23]]
    )
    return UPMNS

#th12, th13, th23 = symbols(r'\theta_{12},\theta_{13},\theta_{23}',real=True)
#print('Upmns = \n',UpmnsStandardParametrization(th12, th13, th23))

def UTBM_correction(theta,xi,delta):
    r = 1/sqrt(2)
    A = Matrix([[cos(theta),sin(theta),0],[-r*sin(theta),r*cos(theta),r],[r*sin(theta),-r*cos(theta),r]])

    B = Matrix([[cos(xi),0,exp(I*delta)*sin(xi)],[0,1,0],[-exp(I*delta)*sin(xi),0,cos(xi)]])
    return MixingMatrix(A*B)

############################################################################
# Casas-Ibarra Parametrization
############################################################################
def parametrizationCI(sqrtM_inv,R,sqrtm,upmns):
    return sqrtM_inv*R*sqrtm*upmns
############################################################################
# Neutrino oscillations data
############################################################################
class Observable():
    '''
    Class that represents a experimental observable
    
    Atributtes
    ----------
    central: float
        Represents the central experimental value of this observable
    sigma1: tuple list

        This is a tuple or list with first sigma range of this observable.
            In our convention the first element is the name of the range. 
            To other side, second and third elements correspond to lower
            and upper bounds respectively.
    sigma2: tuple list
        default:None

        This is a tuple or list with second sigma range of this observable.
            In our convention the first element is the name of the range. 
            To other side, second and third elements correspond to lower
            and upper bounds respectively.
    
    Methods
    -------
    
    '''
    def __init__(self,central,sigma1,sigma2=None,name='Observable'):
        '''
        Parameters
        ----------
        central: float
            Represents the central experimental value of this observable
        sigma1: tuple list

            This is a tuple or list with first sigma range of this observable.
            In our convention the first element is the name of the range. 
            To other side, second and third elements correspond to lower
            and upper bounds respectively.
        sigma2: tuple list
            default:None

            This is a tuple or list with second sigma range of this observable.
            In our convention the first element is the name of the range. 
            To other side, second and third elements correspond to lower
            and upper bounds respectively.
        '''
        self.central = central
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.name = name
    
    def __str__(self):
        if self.sigma2==None:
            return f'{self.__class__.__name__}({self.central},({self.sigma1[0]!r},{self.sigma1[1]!r},{self.sigma1[2]}),{self.name!r})'
        else:
            string = (f'{self.__class__.__name__}(',
               f'{self.central!r},',
               f'({self.sigma1[0]!r},{self.sigma1[1]!r},{self.sigma1[2]!r}),',
               f'({self.sigma2[0]!r},{self.sigma2[1]!r},{self.sigma2[2]!r}),{self.name!r})')  
            s = ''
            for st in string:
                s+=st
            return s
    def __repr__(self):
        if self.sigma2==None:
            stringlist = (f'{self.name} with bounds:\n',
            f'{self.sigma1[1]} < {self.name} < {self.sigma1[2]} at {self.sigma1[0]}.')
        else:
            stringlist =  (f'{self.name} with bounds:\n',
            f'{self.sigma1[1]} < {self.name} < {self.sigma1[2]} at {self.sigma1[0]}\n',
            f'{self.sigma2[1]} < {self.name} < {self.sigma2[2]} at {self.sigma2[0]}.')
        s = ''
        for st in stringlist:
            s+=st
        return s
    
    def __mul__(self,other):
        if isinstance(other,(int,float)):
            newcentral = self.central*other
            newsigma1 = (self.sigma1[0],self.sigma1[1]*other,self.sigma1[2]*other)
            newsigma2 = (self.sigma2[0],self.sigma2[1]*other,self.sigma2[2]*other)
            return Observable(newcentral,newsigma1,newsigma2,name=self.name)
#O = Observable(0.5,('1 sigma',0.3,0.7),('2 sigma',0.1,0.9),name='NewObservable')
#print(O)

class SetObservables():
    pass

class NuOscObservables(SetObservables):
    sin2theta12 = Observable(0.310,('1 sigma',0.31-0.012,0.31+0.013),('3 sigma',0.275,0.350),name='sin(θ_12)^2')
    sin2theta13 = Observable(0.02241,('1 sigma',0.02241-0.00065,0.02241+0.00066),('3 sigma',0.02046,0.02440),name='sin(θ_13)^2')
    sin2theta23 = Observable(0.558,('1 sigma',0.558-0.033,0.558+0.020),('3 sigma',0.427,0.609),name='sin(θ_23)^2')
    deltaCP = Observable(222,('1 sigma',222-28,222+38),('3 sigma',141,370),name='delta_CP')
    squareDm21 = Observable(7.39,('1 sigma',7.30-0.20,7.39+0.21),('3 sigma',6.79,8.01),name='Dm21^2')*1e-5
    squareDm31 = Observable(2.523,('1 sigma',2.523-0.030,2.523+0.032),('3 sigma',2.432,2.618),name='Dm21^2')*1e-3
    
    def substitutions(self,th12,th13,th23):
        '''
        Parameters
        ----------
        th12,th13,th23: sympy symbols

        Return a dictionary with the keys th12,th13,th23 and as values the central
        experimental values of the mixings angles. 
        '''
        return {th12:asin(sqrt(self.sin2theta12.central)),
        th13:asin(sqrt(self.sin2theta13.central)),th23:asin(sqrt(self.sin2theta23.central))}
    



#print(NuOscObservables().substitutions(1,2,3))


    

