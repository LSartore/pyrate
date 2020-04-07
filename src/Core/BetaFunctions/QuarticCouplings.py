# -*- coding: utf-8 -*-

from sympy import Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract

import itertools


class QuarticBetaFunction(BetaFunction):
        
    def compute(self, a,b,c,d, nLoops):
        perm = list(itertools.permutations([a, b, c, d], 4))
        permSet = set(perm)
        coeff = r(len(perm),len(permSet))

        ret = 0
        for s1,s2,s3,s4 in permSet:
            ret += coeff * self.Beta(s1,s2,s3,s4, nLoops=nLoops)
        
        return r(1,24)*ret
        
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.q{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        ## 1-loop
        
        self.coefficients.append( [r(36), r(-12), r(3), r(2), r(-12)] )
    
        ## 2-loop
    
        self.coefficients.append( [r(324), r(-684), r(646), r(-28), r(-32), r(12),
                                   r(60), r(0), r(6), r(-143,3), r(11,3), r(10,3),
                                   r(-18), r(24), r(-18), r(1,3), r(-6), r(0),
                                   r(-144),
                                   # Symbol('Q_19'), 
                                   r(60), r(10), r(0), r(-3), r(0),
                                   r(24), r(-48), r(12), r(0), r(-2), r(-3), 
                                   r(48), r(24), r(24)] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def q1_1(self, a,b,c,d):
        return tensorContract(self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(A_,a,e_),
                              self.Ts(C_,e_,b),
                              self.Ts(B_,c,f_),
                              self.Ts(D_,f_,d))
        
    def q1_2(self, a,b,c,d):
        return tensorContract(self.C2S(a,e_),
                              self.l(e_,b,c,d))
        
        
    def q1_3(self, a,b,c,d):
        return tensorContract(self.l(a,b,e_,f_),
                              self.l(e_,f_,c,d))
        
    def q1_4(self, a,b,c,d):
        return tensorContract(self.Y2S(a,e_),
                              self.l(e_,b,c,d))
        
    def q1_5(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(c,k_,l_),
                              self.yt(d,l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    ######################
    #  2-loop functions  #
    ######################
        
    def q2_1(self, a,b,c,d):
        return tensorContract(self.G(A_,B_),
                              self.Ts(A_,a,i_),
                              self.Ts(C_,i_,j_),
                              self.Ts(E_,j_,k_),
                              self.Ts(B_,k_,b),
                              self.G(C_,D_),
                              self.G(E_, F_),
                              self.Ts(D_,c,l_),
                              self.Ts(F_,l_,d))
    
    def q2_2(self, a,b,c,d):
        return tensorContract(self.C2S(a,i_),
                              self.Ts(A_,i_,j_),
                              self.Ts(C_,j_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,c,k_),
                              self.Ts(D_,k_,d))
    
    def q2_3(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,b),
                              self.G(A_,B_),
                              self.G(C_,E_),
                              self.C2G(E_,F_),
                              self.G(F_,D_),
                              self.Ts(B_,c,j_),
                              self.Ts(D_,j_,d))
    
    def q2_4(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,b),
                              self.G(A_,B_),
                              self.G(C_,E_),
                              self.S2S(E_,F_),
                              self.G(F_,D_),
                              self.Ts(B_,c,j_),
                              self.Ts(D_,j_,d))
    
    def q2_5(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,b),
                              self.G(A_,B_),
                              self.G(C_,E_),
                              self.S2F(E_,F_),
                              self.G(F_,D_),
                              self.Ts(B_,c,j_),
                              self.Ts(D_,j_,d))
    
    def q2_6(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,e_),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,b,j_),
                              self.Ts(D_,j_,f_),
                              self.l(e_,f_,c,d))
    
    def q2_7(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,e_,j_),
                              self.Ts(D_,j_,f_),
                              self.l(e_,f_,c,d))
    
    def q2_8(self, a,b,c,d):
        return tensorContract(self.C2S(a,e_),
                              self.C2S(b,f_),
                              self.l(e_,f_,c,d))
    
    def q2_9(self, a,b,c,d):
        return tensorContract(self.C2S(a,e_),
                              self.C2S(e_,f_),
                              self.l(f_,b,c,d))
    
    def q2_10(self, a,b,c,d):
        return tensorContract(self.C2SG(a,e_),
                              self.l(e_,b,c,d))
    
    def q2_11(self, a,b,c,d):
        return tensorContract(self.C2SS(a,e_),
                              self.l(e_,b,c,d))
    
    def q2_12(self, a,b,c,d):
        return tensorContract(self.C2SF(a,e_),
                              self.l(e_,b,c,d))
    
    def q2_13(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,e_),
                              self.Ts(B_,b,f_),
                              self.G(A_,B_),
                              self.l(e_,f_,g_,h_),
                              self.l(g_,h_,c,d))
    
    def q2_14(self, a,b,c,d):
        return tensorContract(self.l(a,b,e_,f_),
                              self.C2S(f_,g_),
                              self.l(e_,g_,c,d))
    
    def q2_15(self, a,b,c,d):
        return tensorContract(self.C2S(a,e_),
                              self.l(e_,b,f_,g_),
                              self.l(f_,g_,c,d))
    
    def q2_16(self, a,b,c,d):
        return tensorContract(self.l(a,e_,f_,g_),
                              self.l(e_,f_,g_,h_),
                              self.l(h_,b,c,d))
    
    def q2_17(self, a,b,c,d):
        return tensorContract(self.l(a,b,e_,f_),
                              self.l(e_,g_,h_,c),
                              self.l(f_,g_,h_,d))
    
    def q2_18(self, a,b,c,d):
        return tensorContract(self.l(a,b,e_,f_),
                              self.l(e_,f_,g_,h_),
                              self.l(g_,h_,c,d))
    
    def q2_19(self, a,b,c,d):
        return tensorContract(self.Ts(A_,a,e_),
                              self.Ts(C_,e_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.T(D_,i_,j_),
                              self.T(B_,j_,k_),
                              self.y(c,k_,l_),
                              self.yt(d,l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_20(self, a,b,c,d):
        return tensorContract(self.Y2S(a,e_),
                              self.Ts(A_,e_,f_),
                              self.Ts(C_,f_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,c,g_),
                              self.Ts(D_,g_,d))
    
    def q2_21(self, a,b,c,d):
        return tensorContract(self.Y2SCF(a,e_),
                              self.l(e_,b,c,d))
    
    def q2_22(self, a,b,c,d):
        return tensorContract(self.C2S(a,e_),
                              self.Y2S(e_,f_),
                              self.l(f_,b,c,d))
    
    def q2_23(self, a,b,c,d):
        return tensorContract(self.l(a,b,e_,f_),
                              self.Y2S(f_,g_),
                              self.l(e_,g_,c,d))
    
    def q2_24(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.T(A_,j_,k_),
                              self.yt(b,k_,l_),
                              self.y(c,l_,m_),
                              self.G(A_,B_),
                              self.T(B_,m_,n_),
                              self.yt(d,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_25(self, a,b,c,d):
        return tensorContract(self.C2S(a,e_),
                              self.y(e_,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(c,k_,l_),
                              self.yt(d,l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def q2_26(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(c,k_,l_),
                              self.yt(d,l_,m_),
                              self.C2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_27(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(e_,j_,k_),
                              self.y(b,k_,l_),
                              self.yt(f_,l_,i_),
                              self.l(e_,f_,c,d),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_28(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(f_,l_,i_),
                              self.l(e_,f_,c,d),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_29(self, a,b,c,d):
        return tensorContract(self.Y4S(a,e_),
                              self.l(e_,b,c,d))
    
    def q2_30(self, a,b,c,d):
        return tensorContract(self.Y2SYF(a,e_),
                              self.l(e_,b,c,d))
    
    def q2_31(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(c,k_,l_),
                              self.yt(e_,l_,m_),
                              self.y(d,m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_32(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(c,l_,m_),
                              self.y(d,m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def q2_33(self, a,b,c,d):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(c,k_,l_),
                              self.yt(d,l_,m_),
                              self.Y2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    
    
    
    
    
    