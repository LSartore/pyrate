# -*- coding: utf-8 -*-

from sympy import transpose, Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract

class FermionMassBetaFunction(BetaFunction):        
        
    def compute(self, i,j, nLoops):
        ret = self.Beta(i,j, nLoops=nLoops)

        if i!=j:
            ret += transpose(self.Beta(j,i, nLoops=nLoops))
        else:
            ret += transpose(ret)

        return r(1,2)*ret
    
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.M{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        ## 1-loop
        
        self.coefficients.append( [r(-6), r(2), r(1)] )
    
        ## 2-loop    
    
        # self.coefficients.append( [r(-21,2), r(12), r(0), r(-3), r(49,4), r(-1,4),
        #                            r(-1,2), r(-97,3), r(11,6), r(5,3), r(1,12), r(12),
        #                            r(0), r(6), r(-12), r(10), r(6), r(5,2),
        #                            r(9), r(-1,2), r(-7,2), r(0), r(-2), r(2),
        #                            r(0), r(-2), r(0), r(-1,2), r(-2), r(-1,4),
        #                            r(-3,4), r(-1), r(-3,4)] )
        
        self.coefficients.append( [r(0), r(-3), r(-97,3), r(11,6), r(5,3), r(12),
                                   r(0), r(6), r(10), r(6), r(9), r(-1,2),
                                   r(-7,2), r(-2), r(2), r(0), r(-2), r(0),
                                   r(-2), r(-1,4), r(-1), r(-3,4)] )
    ######################
    #  1-loop functions  #
    ######################
        
    def M1_1(self, i,j):
        return tensorContract(self.C2F(k_,j),
                              self.M(i,k_))
        
    def M1_2(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Mt(k_,l_),
                              self.y(b_,l_,j))
        
    def M1_3(self, i,j):
        return tensorContract(self.M(i,k_),
                              self.Y2Ft(k_,j))
        
    
    ######################
    #  2-loop functions  #
    ######################
    
    def M2_1(self, i,j):
        return tensorContract(self.C2Ft(i,k_),
                              self.C2F(l_,j),
                              self.M(k_,l_))
    
    def M2_2(self, i,j):
        return tensorContract(self.C2F(l_,j),
                              self.C2F(k_,l_),
                              self.M(i,k_))
    
    def M2_3(self, i,j):
        return tensorContract(self.M(i,k_),
                              self.C2FG(k_,j))
    
    def M2_4(self, i,j):
        return tensorContract(self.M(i,k_),
                              self.C2FS(k_,j))
    
    def M2_5(self, i,j):
        return tensorContract(self.M(i,k_),
                              self.C2FF(k_,j))
    
    def M2_6(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.T(A_,k_,l_),
                              self.G(A_,B_),
                              self.yt(b_,l_,m_),
                              self.T(B_,n_,j),
                              self.M(m_,n_))
    
    def M2_7(self, i,j):
        return tensorContract(self.Y2F(i,_k_),
                              self.Tt(A_,k_,l_),
                              self.G(A_,B_),
                              self.M(l_,m_),
                              self.T(B_,m_,j))
    
    def M2_8(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Mt(k_,l_),
                              self.C2S(b_,c_),
                              self.y(c_,l_,j))
    
    def M2_9(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.C2F(k_,l_),
                              self.Mt(l_,m_),
                              self.y(b_,m_,j))
    
    def M2_10(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Mt(k_,l_),
                              self.y(b_,l_,m_),
                              self.C2F(m_,j))
    
    def M2_11(self, i,j):
        return tensorContract(self.Y2FCS(i,k_),
                              self.M(k_,j))
    
    def M2_12(self, i,j):
        return tensorContract(self.Y2FCF(i,k_),
                              self.M(k_,j))
    
    def M2_13(self, i,j):
        return tensorContract(self.M(i,k_),
                              self.Y2Ft(k_,l_),
                              self.C2F(l_,j))
    
    def M2_14(self, i,j):
        return tensorContract(self.h(b_,c_,d_),
                              self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.y(d_,l_,j))
    
    def M2_15(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.M(l_,m_),
                              self.yt(b_,m_,n_),
                              self.y(c_,n_,j))
    
    def M2_16(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Mt(k_,l_),
                              self.y(c_,l_,m_),
                              self.yt(b_,m_,n_),
                              self.y(c_,n_,j))
    
    def M2_17(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.M(l_,m_),
                              self.yt(c_,m_,n_),
                              self.y(b_,n_,j))
    
    def M2_18(self, i,j):
        return tensorContract(self.Y4F(i,k_),
                              self.M(k_,j))
    
    def M2_19(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Mt(k_,l_),
                              self.Y2F(l_,m_),
                              self.y(b_,m_,j))
    
    def M2_20(self, i,j):
        return tensorContract(self.M(i,k_),
                              self.Y2FYFt(k_,j))
    
    def M2_21(self, i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Y2S(b_,c_),
                              self.Mt(k_,l_),
                              self.y(c_,l_,j))
    
    def M2_22(self, i,j):
        return tensorContract(self.Y2FYS(i,k_),
                              self.M(k_,j))
    
    
    
    
    
    
    