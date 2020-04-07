# -*- coding: utf-8 -*-

from sympy import Identity as sympyIdentity, MatMul, Mul, Symbol, Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract, Identity


class VevBetaFunction(BetaFunction):
        
    def compute(self, a, nLoops):
        return self.Beta(a, nLoops=nLoops)
    
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.V{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        xi = Symbol('_xiGauge', real=True)
        # Coeff = quartic * symmetryFactor * 2/24
        #       = quartic * symmetryFactor * 1/12
        
        ## 1-loop
        
        self.coefficients.append( [r(3) + xi, r(-1,2)] )
    
        # ## 2-loop
        
        self.coefficients.append( [r(35,3) - r(3,2)*xi  - r(3,2)*xi**2, r(-10,12), r(-11,12),
                                   r(-3,2) + r(2)*xi + r(2)*xi**2, r(-1,12),
                                   r(3,4), r(1,2), r(-10,4), r(-1,2)*xi] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def V1_1(self, a):
        return tensorContract(self.C2S(a,b_),
                              self.v(b_))
        
    def V1_2(self, a):
        return tensorContract(self.Y2S(a,b_),
                              self.v(b_))
        
    
    ######################
    #  2-loop functions  #
    ######################
    
    def V2_1(self, a):
        return tensorContract(self.C2SG(a,b_),
                              self.v(b_))
    def V2_2(self, a):
        return tensorContract(self.C2SF(a,b_),
                              self.v(b_))
    def V2_3(self, a):
        return tensorContract(self.C2SS(a,b_),
                              self.v(b_))
    def V2_4(self, a):
        return tensorContract(self.C2S(a,b_),
                              self.C2S(b_,c_),
                              self.v(c_))
    def V2_5(self, a):
        return tensorContract(self.l(a,b_,c_,d_),
                              self.l(b_,c_,d_,e_),
                              self.v(e_))
    def V2_6(self, a):
        return tensorContract(self.Y2SYF(b_,a),
                              self.v(b_))
    def V2_7(self, a):
        return tensorContract(self.Y4S(b_,a),
                              self.v(b_))
    def V2_8(self, a):
        return tensorContract(self.Y2SCF(b_,a),
                              self.v(b_))
    def V2_9(self, a):
        return tensorContract(self.C2S(a,b_),
                              self.Y2S(b_,c_),
                              self.v(c_))
    

class ScalarAnomalous(BetaFunction):
        
    def compute(self, a, b, nLoops):
        return self.Beta(a, b, nLoops=nLoops)
    
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.gS{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        xi = Symbol('_xiGauge', real=True)
        # Coeff = quartic * symmetryFactor * 2/24
        #       = quartic * symmetryFactor * 1/12
        
        ## 1-loop
        
        self.coefficients.append( [r(3) - xi, r(-1,2)] )
    
        # ## 2-loop
        
        self.coefficients.append( [r(35,3) - r(2)*xi  - r(1,4)*xi**2, r(-10,12), r(-11,12),
                                   r(-3,2), r(-1,12), r(3,4),
                                   r(1,2), r(-10,4)] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def gS1_1(self, a, b):
        return self.C2S[a,b]
        
    def gS1_2(self, a, b):
        return self.Y2S[a,b]
        
    
    ######################
    #  2-loop functions  #
    ######################
    
    def gS2_1(self, a, b):
        return self.C2SG[a,b]
    
    def gS2_2(self, a, b):
        return self.C2SF[a,b]
    
    def gS2_3(self, a, b):
        return self.C2SS[a,b]
    
    def gS2_4(self, a, b):
        return tensorContract(self.C2S(a,c_),
                              self.C2S(c_,b))
    def gS2_5(self, a, b):
        return tensorContract(self.l(a,e_,c_,d_),
                              self.l(e_,c_,d_,b))
    def gS2_6(self, a, b):
        return self.Y2SYF[b,a]
    
    def gS2_7(self, a, b):
        return self.Y4S[b,a]
    
    def gS2_8(self, a, b):
        return self.Y2SCF[b,a]
    

class FermionAnomalous(BetaFunction):
    
    def compute(self, i, j, nLoops):
        
        return self.Beta(i, j, nLoops=nLoops)
    
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.gF{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        xi = Symbol('_xiGauge', real=True)
        # Coeff = quartic * symmetryFactor * 2/24
        #       = quartic * symmetryFactor * 1/12
        
        ## 1-loop
        
        self.coefficients.append( [r(2)*xi, r(1)] )
    
        # ## 2-loop
        
        self.coefficients.append( [r(35,3) - r(3,2)*xi  - r(3,2)*xi**2, r(-10,12), r(-11,12),
                                   r(-3,2) + r(2)*xi + r(2)*xi**2, r(-1,12),
                                   r(3,4), r(1,2), r(-10,4), r(-1,2)*xi] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def gF1_1(self, i, j):
        # print('C2F : \n')
        # print(self.C2F[i,j], type(self.C2F[i,j]))
        # print(self.matrixToSymbol(self.C2F[i,j]), type(self.matrixToSymbol(self.C2F[i,j])))
        return self.matrixToSymbol(self.C2F[i,j])
        
    def gF1_2(self, i, j):
        # print('Y2F : \n')
        # print(self.Y2Ft[i,j], type(self.Y2Ft[i,j]))
        return self.Y2Ft[i,j]
        
    
    ######################
    #  2-loop functions  #
    ######################
    
    def gF2_1(self, i, j):
        return tensorContract(self.C2SG(a,b_),
                              self.v(b_))
    def gF2_2(self, i, j):
        return tensorContract(self.C2SF(a,b_),
                              self.v(b_))
    def gF2_3(self, i, j):
        return tensorContract(self.C2SS(a,b_),
                              self.v(b_))
    def gF2_4(self, i, j):
        return tensorContract(self.C2S(a,b_),
                              self.C2S(b_,c_),
                              self.v(c_))
    def gF2_5(self, i, j):
        return tensorContract(self.l(a,b_,c_,d_),
                              self.l(b_,c_,d_,e_),
                              self.v(e_))
    def gF2_6(self, i, j):
        return tensorContract(self.Y2SYF(b_,a),
                              self.v(b_))
    def gF2_7(self, i, j):
        return tensorContract(self.Y4S(b_,a),
                              self.v(b_))
    def gF2_8(self, i, j):
        return tensorContract(self.Y2SCF(b_,a),
                              self.v(b_))
    def gF2_9(self, i, j):
        return tensorContract(self.C2S(a,b_),
                              self.Y2S(b_,c_),
                              self.v(c_))
    
    
    ##
    
    def matrixToSymbol(self, expr):
        if isinstance(expr, MatMul):
            args = expr.args
            newArgs = []
            
            for el in args:
                if isinstance(el, sympyIdentity):
                    newArgs.append(Identity(el.args[0]))
                else:
                    newArgs.append(el)
            
            return Mul(*newArgs)
        
        if isinstance(expr, sympyIdentity):
            return Identity(expr.args[0])
        
        return expr
                    
    
    