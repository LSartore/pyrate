# -*- coding: utf-8 -*-

from sympy import transpose, Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract


class YukawaBetaFunction(BetaFunction):        
        
    def compute(self, a,i,j, nLoops):
        ret = self.Beta(a,i,j, nLoops=nLoops)
        if i!=j:
            ret += transpose(self.Beta(a,j,i, nLoops=nLoops))
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
                    self.functions[i].append(eval(f"self.y{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        ## 1-loop
        
        self.coefficients.append( [r(0), r(-6), r(2), r(1), r(1,2)] )
    
        ## 2-loop
    
        self.coefficients.append( [r(-21,2), r(12), r(0), r(-3), r(49,4), r(-1,4),
                                   r(-1,2), r(-97,3), r(11,6), r(5,3), r(1,12), r(12),
                                   r(0), r(6), r(-12), r(10), r(6), r(5,2),
                                   r(9), r(-1,2), r(-7,2), r(0), r(-2), r(2),
                                   r(0), r(-2), r(0), r(-1,2), r(-2), r(-1,4),
                                   r(-3,4), r(-1), r(-3,4)] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def y1_1(self, a,i,j):
        return tensorContract(self.C2S(b_,a),
                              self.y(b_,i,j))
        
    def y1_2(self, a,i,j):
        return tensorContract(self.C2F(k_,j),
                              self.y(a,i,k_))
        
    def y1_3(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(a,k_,l_),
                              self.y(b_,l_,j))
        
    def y1_4(self, a,i,j):
        return tensorContract(self.y(a,i,k_),
                              self.Y2Ft(k_,j))
        
    def y1_5(self, a,i,j):
        return tensorContract(self.Y2S(b_,a),
                              self.y(b_,i,j))
        
    
    ######################
    #  2-loop functions  #
    ######################
        
    def y2_1(self, a,i,j):
        return tensorContract(self.C2S(b_,a),
                              self.C2S(c_,b_),
                              self.y(c_,i,j))
    
    def y2_2(self, a,i,j):
        return tensorContract(self.C2S(b_,a),
                              self.C2F(k_,j),
                              self.y(b_,i,k_))
    
    def y2_3(self, a,i,j):
        return tensorContract(self.C2Ft(i,k_),
                              self.C2F(l_,j),
                              self.y(a,k_,l_))
    
    def y2_4(self, a,i,j):
        return tensorContract(self.C2F(l_,j),
                              self.C2F(k_,l_),
                              self.y(a,i,k_))
    
    def y2_5(self, a,i,j):
        return tensorContract(self.C2SG(b_,a),
                              self.y(b_,i,j))
    
    def y2_6(self, a,i,j):
        return tensorContract(self.C2SS(b_,a),
                              self.y(b_,i,j))
    
    def y2_7(self, a,i,j):
        return tensorContract(self.C2SF(b_,a),
                              self.y(b_,i,j))
    
    def y2_8(self, a,i,j):
        return tensorContract(self.y(a,i,k_),
                              self.C2FG(k_,j))
    
    def y2_9(self, a,i,j):
        return tensorContract(self.y(a,i,k_),
                              self.C2FS(k_,j))
    
    def y2_10(self, a,i,j):
        return tensorContract(self.y(a,i,k_),
                              self.C2FF(k_,j))
    
    def y2_11(self, a,i,j):
        return tensorContract(self.y(b_,i,j),
                              self.l(c_,d_,e_,a),
                              self.l(b_,c_,d_,e_))
    
    def y2_12(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.T(A_,k_,l_),
                              self.G(A_,B_),
                              self.yt(b_,l_,m_),
                              self.T(B_,n_,j),
                              self.y(a,m_,n_))
    
    def y2_13(self, a,i,j):
        return tensorContract(self.Y2F(i,_k_),
                              self.Tt(A_,k_,l_),
                              self.G(A_,B_),
                              self.y(a,l_,m_),
                              self.T(B_,m_,j))
    
    def y2_14(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(a,k_,l_),
                              self.C2S(b_,c_),
                              self.y(c_,l_,j))
    
    def y2_15(self, a,i,j):
        return tensorContract(self.C2S(c_,a),
                              self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.y(b_,l_,j))
    
    def y2_16(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.C2F(k_,l_),
                              self.yt(a,l_,m_),
                              self.y(b_,m_,j))
    
    def y2_17(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(a,k_,l_),
                              self.y(b_,l_,m_),
                              self.C2F(m_,j))
    
    def y2_18(self, a,i,j):
        return tensorContract(self.Y2SCF(b_,a),
                              self.y(b_,i,j))
    
    def y2_19(self, a,i,j):
        return tensorContract(self.Y2FCS(i,k_),
                              self.y(a,k_,j))
    
    def y2_20(self, a,i,j):
        return tensorContract(self.Y2FCF(i,k_),
                              self.y(a,k_,j))
    
    def y2_21(self, a,i,j):
        return tensorContract(self.y(a,i,k_),
                              self.Y2Ft(k_,l_),
                              self.C2F(l_,j))
    
    def y2_22(self, a,i,j):
        return tensorContract(self.C2S(b_,a),
                              self.Y2S(c_,b_),
                              self.y(c_,i,j))
    
    def y2_23(self, a,i,j):
        return tensorContract(self.l(b_,c_,d_,a),
                              self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.y(d_,l_,j))
    
    def y2_24(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.y(a,l_,m_),
                              self.yt(b_,m_,n_),
                              self.y(c_,n_,j))
    
    def y2_25(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(a,k_,l_),
                              self.y(c_,l_,m_),
                              self.yt(b_,m_,n_),
                              self.y(c_,n_,j))
    
    def y2_26(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(c_,k_,l_),
                              self.y(a,l_,m_),
                              self.yt(c_,m_,n_),
                              self.y(b_,n_,j))
    
    def y2_27(self, a,i,j):
        return tensorContract(self.Y4F(i,k_),
                              self.y(a,k_,j))
    
    def y2_28(self, a,i,j):
        return tensorContract(self.Y4S(b_,a),
                              self.y(b_,i,j))
    
    def y2_29(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.yt(a,k_,l_),
                              self.Y2F(l_,m_),
                              self.y(b_,m_,j))
    
    def y2_30(self, a,i,j):
        return tensorContract(self.y(a,i,k_),
                              self.Y2FYFt(k_,j))
    
    def y2_31(self, a,i,j):
        return tensorContract(self.Y2SYF(b_,a),
                              self.y(b_,i,j))
    
    def y2_32(self, a,i,j):
        return tensorContract(self.y(b_,i,k_),
                              self.Y2S(b_,c_),
                              self.yt(a,k_,l_),
                              self.y(c_,l_,j))
    
    def y2_33(self, a,i,j):
        return tensorContract(self.Y2FYS(i,k_),
                              self.y(a,k_,j))
    
    
    
    
    
    
    