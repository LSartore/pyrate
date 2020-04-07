# -*- coding: utf-8 -*-

from sympy import Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract


class GaugeBetaFunction(BetaFunction):
        
    def compute(self, A,B, nLoops):
        ret = 0
        
        for C,D in self.gaugeIndices(A,B):
            ret += self.G[A,C]*self.Beta(C,D, nLoops=nLoops)*self.G[D,B]
            ret += self.G[B,D]*self.Beta(D,C, nLoops=nLoops)*self.G[C,A]
            
        return r(1,2)*ret
        
        # for C,D in self.gaugeIndices(A,B):
        #     ret += self.G[A,C]*self.Beta(C,D, nLoops=nLoops)*self.G[D,B]
            
        return ret
    
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.g{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        ## 1-loop
        
        self.coefficients.append( [r(-22,3), r(2,3), r(1,3)] )
        
        ## 2-loop
        
        self.coefficients.append( [r(2), r(4), r(-68,3), r(10,3), r(2,3), r(-1), r(0)] )
    
        ## 3-loop
        
        self.coefficients.append( [r(-1), r(29,2), r(133,18), r(679,36), r(-11,18), r(-25,18),
                                   r(-23,36), r(-49,36), r(4), r(25,2), r(-2857,27), r(-79,108),
                                   r(1,54), r(1415,54), r(545,108), r(-29,54), r(1), r(-1,12),
                                   r(-5,4), r(-1,4), r(-1), r(-7), r(-7,2), r(-6),
                                   r(9,4), r(1), r(-1), r(3,2), r(7,8), r(1,2),
                                   r(1,8), r(3,8), r(-1,8)] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def g1_1(self, A,B):
        return self.C2G[A,B]
    
    def g1_2(self, A,B):
        return self.S2F[A,B]
    
    def g1_3(self, A,B):
        return self.S2S[A,B]
    
    
    ######################
    #  2-loop functions  #
    ######################
        
    def g2_1(self, A,B):
        return self.S2FCF[A,B]
    
    def g2_2(self, A,B):
        return self.S2SCS[A,B]
    
    def g2_3(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.C2G(D_,B))
    
    def g2_4(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2F(D_,B))
    
    def g2_5(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2S(D_,B))
    
    def g2_6(self, A,B):
        return self.S2FYF[A,B]
    
    def g2_7(self, A,B):
        return self.S2SYS[A,B]
    

    
    ######################
    #  3-loop functions  #
    ######################
    
    def g3_1(self, A,B):
        return tensorContract(self.T(A,i_,j_),
                              self.T(B,j_,k_),
                              self.C2F(k_,l_),
                              self.C2F(l_,i_),
                              doTrace=True)
    
    def g3_2(self, A,B):
        return tensorContract(self.Ts(A,a_,d_),
                              self.Ts(B,d_,b_),
                              self.C2S(b_,c_),
                              self.C2S(c_,a_))
    
    def g3_3(self, A,B):
        return tensorContract(self.T(A,i_,j_),
                              self.T(B,j_,k_),
                              self.C2FG(k_,i_),
                              doTrace=True)
    
    def g3_4(self, A,B):
        return tensorContract(self.Ts(A,a_,c_),
                              self.Ts(B,c_,b_),
                              self.C2SG(b_,a_))
    
    def g3_5(self, A,B):
        return tensorContract(self.T(A,i_,j_),
                              self.T(B,j_,k_),
                              self.C2FF(k_,i_),
                              doTrace=True)
    
    def g3_6(self, A,B):
        return tensorContract(self.Ts(A,a_,c_),
                              self.Ts(B,c_,b_),
                              self.C2SF(b_,a_))
    
    def g3_7(self, A,B):
        return tensorContract(self.T(A,i_,j_),
                              self.T(B,j_,k_),
                              self.C2FS(k_,i_),
                              doTrace=True)
    
    def g3_8(self, A,B):
        return tensorContract(self.Ts(A,a_,c_),
                              self.Ts(B,c_,b_),
                              self.C2SS(b_,a_))
    
    def g3_9(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2FCF(D_,B))
    
    def g3_10(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2SCS(D_,B))
    
    def g3_11(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.C2G(D_,E_),
                              self.G(E_,F_),
                              self.C2G(F_,B))
    
    def g3_12(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2F(D_,E_),
                              self.G(E_,F_),
                              self.S2F(F_,B))
    
    def g3_13(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2S(D_,E_),
                              self.G(E_,F_),
                              self.S2S(F_,B))
    
    def g3_14(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.C2G(D_,E_),
                              self.G(E_,F_),
                              self.S2F(F_,B))
    
    def g3_15(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.C2G(D_,E_),
                              self.G(E_,F_),
                              self.S2S(F_,B))
    
    def g3_16(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2F(D_,E_),
                              self.G(E_,F_),
                              self.S2S(F_,B))
    
    def g3_17(self, A,B):
        return tensorContract(self.Ts(A,a_,e_),
                              self.Ts(C_,e_,b_),
                              self.G(C_,D_),
                              self.l(a_,b_,c_,d_),
                              self.Ts(B,c_,f_),
                              self.Ts(D_,f_,d_))
    
    def g3_18(self, A,B):
        return tensorContract(self.Ts(A,a_,f_),
                              self.Ts(B,f_,b_),
                              self.l(b_,c_,d_,e_),
                              self.l(c_,d_,e_,a_))
    
    
    def g3_19(self, A,B):
        return tensorContract(self.T(A,i_,j_),
                              self.T(B,j_,k_),
                              self.C2F(k_,l_),
                              self.Y2Ft(l_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_20(self, A,B):
        return tensorContract(self.Tt(A,i_,j_),
                              self.Tt(B,j_,k_),
                              self.Y2FCF(k_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
                              
    
    def g3_21(self, A,B):
        return tensorContract(self.Ts(A,a_,e_),
                              self.Ts(B,e_,b_),
                              self.Y2SCF(b_,a_))
    
    def g3_22(self, A,B):
        return tensorContract(self.Tt(A,i_,j_),
                              self.Tt(B,j_,k_),
                              self.Y2FCS(k_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_23(self, A,B):
        return tensorContract(self.Ts(A,a_,e_),
                              self.Ts(B,e_,b_),
                              self.C2S(b_,c_),
                              self.Y2S(c_,a_))
    
    def g3_24(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2FYF(D_,B))
    
    def g3_25(self, A,B):
        return tensorContract(self.C2G(A,C_),
                              self.G(C_,D_),
                              self.S2SYS(D_,B))
    
    def g3_26(self, A,B):
        return tensorContract(self.T(A,i_,j_),
                              self.yt(a_,j_,k_),
                              self.T(B,l_,m_),
                              self.y(b_,k_,l_),
                              self.yt(b_,m_,n_),
                              self.y(a_,n_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_27(self, A,B):
        return tensorContract(self.Ts(A,a_,e_),
                              self.Ts(B,e_,b_),
                              self.Y4S(b_,a_))
    
    def g3_28(self, A,B):
        return tensorContract(self.Tt(A,i_,j_),
                              self.Tt(B,j_,k_),
                              self.Y4F(k_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_29(self, A,B):
        return tensorContract(self.Tt(A,i_,j_),
                              self.Tt(B,j_,k_),
                              self.Y2FYS(k_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_30(self, A,B):
        return tensorContract(self.Ts(A,a_,e_),
                              self.Ts(B,e_,b_),
                              self.Y2SYF(b_,a_))
    
    def g3_31(self, A,B):
        return tensorContract(self.Tt(A,i_,j_),
                              self.Tt(B,j_,k_),
                              self.Y2F(k_,l_),
                              self.Y2F(l_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_32(self, A,B):
        return tensorContract(self.Tt(A,i_,j_),
                              self.Tt(B,j_,k_),
                              self.Y2FYF(k_,i_),
                              doTrace=True,
                              yukSorting=self.model.YukPos)
    
    def g3_33(self, A,B):
        return tensorContract(self.Ts(A,a_,e_),
                              self.Ts(B,e_,b_),
                              self.Y2S(b_,c_),
                              self.Y2S(c_,a_))
    