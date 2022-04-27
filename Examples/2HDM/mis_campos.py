#-*- coding: utf-8 -*-
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
#from IPython.display import display
#import pandas as pd
#pd.options.display.max_colwidth=1000 #ancho de las columnas de pandas
# lo usamos en la defifnición de la funcion terms_rep
from sympy import Function,factor,Matrix,latex,Mul,S,Add,symbols,Pow,im,conjugate,Abs,sin,exp,I
from sympy.physics.quantum import Dagger, Operator, HermitianOperator
from sympy.physics.matrices import msigma
#numpy
import numpy as np
#display muestra lo que necesite en el formato adecuado


from collections import defaultdict
#init_printing()

#Mis funciones

def latexificar(expr):
    if isinstance(expr,list):
        return np.reshape(list(map(lambda tc: '$'+latex(tc)+'$',np.ravel(expr))),np.shape(expr))
    else:
        return '$'+latex(expr)+'$'

def separar_campos(expr):
    '''Esta función toma un lagrangiano L y sustrae de cada termino la parte 
    que esta involucrada con los campos'''
    lista = [expr.args[i].args_cnc() for i in range(len(expr.args))]
    lista_campos = [Mul(*lista[i][1]) for i in range(0,len(lista))]
    lista = lista_campos
    for i in range(0,len(lista)):
        if lista[i]==1.0:
            lista[i]=S(1)
    return lista

def lista_campos_acoplamientos(L):
    '''Esta función toma un lagrangiano L y devuelve una lista de sublistas cada 
    una con un termino separado en campos y acoplamientos'''
    lista = [L.args[i].args_cnc() for i in range(len(L.args))]
    lista_acoplamientos = [[np.prod(np.array(lista[i][0])),np.prod(np.array(lista[i][1]))] 
                           for i in range(0,len(lista))]
    lista = lista_acoplamientos
    for i in range(0,len(lista)):
        if lista[i][1]==1.0:
            lista[i][1]=S(1)
    return lista

def dic_inv(dicc):
    '''
    Esta función toma un diccionario e invierte sus 
    claves y valores
    '''
    claves = list(dict.keys(dicc))
    return {dicc[i]:i for i in claves}


def terms_rep(lista):
    '''
    Esta función da como resultado un diccionario con claves igual a los elementos de 
    una lista y como valores una lista con los índices donde ella se repite. 
    Nosotros lo usamos para terminos en un expresion simbolica.
    '''
    FDE = lista

    aux = defaultdict(list)
    for index, item in enumerate(FDE):
        aux[item].append(index)
    result = {item: indexs for item, indexs in aux.items()}
    return result

def factorizar_campos(expr,funcion=factor):
    '''
    Esta función factoriza todos los campos o productos de campos.
    '''
    expr = expr.expand()
    campos = terms_rep(separar_campos(expr))
    claves = list(dict.keys(campos))
    terminos = lista_campos_acoplamientos(expr)
    return sum([funcion(sum([terminos[i][0] for i in campos[j]]))*j for j in claves])

def terminos_con(expr,lista):
    '''
    Esta función toma los terminos en "expr" que contienen todos los elementos de lista
    '''
    for termino in lista:
        expr = Add(*[argi for argi in expr.args if argi.has(termino)])
    return expr

def conmutar_campos(expr,lista):
    l = len(lista)
    cambios = {lista[i]:symbols('O_{a}'.format(a=i)) for i in range(l)}
    return expr.subs(cambios).subs(dic_inv(cambios))

def potencia(a):
    if isinstance(a,int):
        return 0
    elif isinstance(a,Pow):
        return a.args[1]
    elif len(a.args)==1:
        return 1
    else:
        print(f'{a} no es una potencia')


def clasificar_interacciones(lista):
    consts = []
    lista1 = []
    lista2 = []
    lista3 = []
    lista4 = []
    for i in lista:
        pot=0
        if isinstance(i[1],(int,float)):
            pot+=0
        elif isinstance(i[1],Pow):
            pot+=i[1].args[1]
        elif isinstance(i[1],(Operator,HermitianOperator)):
            pot+=1
        else:
            pot+= Add(*[potencia(term) for term in i[1].args])
        if pot==0:
            consts.append(i[0])
        elif pot==1:
            lista1.append(i)
        elif pot==2:
            lista2.append(i)
        elif pot==3:
            lista3.append(i)
        else:
            lista4.append(i)
    return consts,lista1,lista2,lista3,lista4 
   
    
    
# definiendo la clase Campo y las subclases Campo_escalar y Campo_fermionico.
class Campo(Operator):
    
    cuantos = 0
    lista_campos = []
    def __init__(self,nombre_entrada=r'\phi_1'):
        Campo.cuantos += 1
        Campo.lista_campos.append(nombre_entrada)
        self.__nombre = nombre_entrada  
        
    #@property
    #def nombre(self):
    #    return self.__nombre
    #@nombre.setter
    #def nombre(self):
    #    self.__nombre = self.nombre_entrada
    
    @classmethod
    def cuantos_hay(cls):
        return cls.cuantos
    
    @classmethod
    def lista(cls):
        return cls.lista_campos

class Campo_escalar_neutro(HermitianOperator):
    _diff_wrt = True
    
class Campo_escalar_cargado(Operator):
    _diff_wrt = True

class Campo_fermionico_neutro(Operator):
    _diff_wrt = True
    
class Campo_fermionico_cargado(Operator):
    _diff_wrt = True
    
class Campo_bosonico_neutro(HermitianOperator):
    _diff_wrt = True
    
class Campo_bosonico_cargado(Operator):
    _diff_wrt = True

    
    
##############################################

class Rep_gauge(Matrix):
    pass
    
def daga(rep_gauge):
    if isinstance(rep_gauge,Rep_gauge):
        return Rep_gauge(list(map(lambda x: Dagger(x),rep_gauge))).T

#class Rep_gauge(Matrix):
    
#    def daga(self):
#        n,m = self.shape
#        return Rep_gauge([[conjugate(self[i,j]) 
#                           if isinstance(self[i,j], (Campo_fermionico_cargado,Campo_fermionico_neutro)) else Dagger(self[i,j]) for i in range(n)] for j in range(m)])
print('All right mis_campos')