# -*- coding: utf-8 -*-

from sympy import Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract

import itertools


class TrilinearBetaFunction(BetaFunction):
        
    def compute(self, a,b,c, nLoops):
        perm = list(itertools.permutations([a, b, c], 3))
        permSet = set(perm)
        coeff = r(len(perm),len(permSet))

        ret = 0
        for s1,s2,s3 in permSet:
            ret += coeff * self.Beta(s1,s2,s3, nLoops=nLoops)
            
        return r(1,6)*ret
        
        # ret = []
        # for s1,s2,s3,s4 in permSet:
        #     ret.append(coeff*r(1,24) * self.Beta(s1,s2,s3,s4, nLoops=nLoops))
            
        # return ret
    
    def fDefinitions(self):
        """ Functions definition """
        
        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.h{i+1}_{count}"))
                    count += 1
                except:
                    break
        
        
    def cDefinitions(self):
        """ Coefficients definition """
        
        # Coeff = quartic * symmetryFactor * 6/24
        #       = quartic * symmetryFactor * 1/4
        
        ## 1-loop
        
        self.coefficients.append( [r(-9), r(3), r(3,2), r(-12)] )
    
        ## 2-loop
        
        self.coefficients.append( [r(6), r(30), r(0), r(9,2), r(-143,4), r(11,4),
                                   r(10,4), r(-9), r(24), r(-9,2), r(-9), r(1,4), 
                                   r(-3), r(-3), r(0), r(-36), r(15,2), r(0), 
                                   r(-3), r(0), r(6), r(-24), r(6), r(6), 
                                   r(0), r(0), r(-3,2), r(-9,4), r(12), r(24),
                                   r(12)] )
    
    ######################
    #  1-loop functions  #
    ######################
        
    def h1_1(self, a,b,c):
        return tensorContract(self.C2S(a,e_),
                              self.h(e_,b,c))
        
        
    def h1_2(self, a,b,c):
        return tensorContract(self.l(a,b,e_,f_),
                              self.h(e_,f_,c))
        
    def h1_3(self, a,b,c):
        return tensorContract(self.Y2S(a,e_),
                              self.h(e_,b,c))
        
    def h1_4(self, a,b,c):
        return tensorContract(self.M(i_,j_),
                              self.yt(a,j_,k_),
                              self.y(b,k_,l_),
                              self.yt(c,l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    ######################
    #  2-loop functions  #
    ######################
    
    def h2_1(self, a,b,c):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,e_),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,b,j_),
                              self.Ts(D_,j_,f_),
                              self.h(e_,f_,c))
    
    def h2_2(self, a,b,c):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,e_,j_),
                              self.Ts(D_,j_,f_),
                              self.h(e_,f_,c))
    
    def h2_3(self, a,b,c):
        return tensorContract(self.C2S(a,e_),
                              self.C2S(b,f_),
                              self.h(e_,f_,c))
    
    def h2_4(self, a,b,c):
        return tensorContract(self.C2S(a,e_),
                              self.C2S(e_,f_),
                              self.h(f_,b,c))
    
    def h2_5(self, a,b,c):
        return tensorContract(self.C2SG(a,e_),
                              self.h(e_,b,c))
    
    def h2_6(self, a,b,c):
        return tensorContract(self.C2SS(a,e_),
                              self.h(e_,b,c))
    
    def h2_7(self, a,b,c):
        return tensorContract(self.C2SF(a,e_),
                              self.h(e_,b,c))
    
    def h2_8(self, a,b,c):
        return tensorContract(self.Ts(A_,a,e_),
                              self.Ts(B_,b,f_),
                              self.G(A_,B_),
                              self.l(e_,f_,g_,h_),
                              self.h(g_,h_,c))
    
    def h2_9(self, a,b,c):
        return tensorContract(self.l(a,b,e_,f_),
                              self.C2S(f_,g_),
                              self.h(e_,g_,c))
    
    def h2_10(self, a,b,c):
        return tensorContract(self.C2S(a,e_),
                              self.h(e_,f_,g_),
                              self.l(f_,g_,b,c))
    
    def h2_11(self, a,b,c):
        return tensorContract(self.C2S(a,e_),
                              self.l(e_,b,f_,g_),
                              self.h(f_,g_,c))
    
    def h2_12(self, a,b,c):
        return tensorContract(self.l(a,e_,f_,g_),
                              self.l(e_,f_,g_,h_),
                              self.h(h_,b,c))
    
    def h2_13(self, a,b,c):
        return tensorContract(self.h(a,e_,f_),
                              self.l(e_,g_,h_,b),
                              self.l(f_,g_,h_,c))
    
    def h2_14(self, a,b,c):
        return tensorContract(self.l(a,b,e_,f_),
                              self.l(e_,g_,h_,c),
                              self.h(f_,g_,h_))
    
    def h2_15(self, a,b,c):
        return tensorContract(self.l(a,b,e_,f_),
                              self.l(e_,f_,g_,h_),
                              self.h(g_,h_,c))
    
    def h2_16(self, a,b,c):
        return ( tensorContract(self.Ts(A_,a,e_),
                                self.Ts(C_,e_,b),
                                self.G(A_,B_),
                                self.G(C_,D_),
                                self.T(D_,i_,j_),
                                self.T(B_,j_,k_),
                                self.M(k_,l_),
                                self.yt(c,l_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)
    
               + tensorContract(self.Ts(A_,a,e_),
                                self.Ts(C_,e_,b),
                                self.G(A_,B_),
                                self.G(C_,D_),
                                self.T(D_,i_,j_),
                                self.T(B_,j_,k_),
                                self.y(c,k_,l_),
                                self.Mt(l_,i_),
                                doTrace=True, yukSorting=self.model.YukPos) )
    
    def h2_17(self, a,b,c):
        return tensorContract(self.Y2SCF(a,e_),
                              self.h(e_,b,c))
    
    def h2_18(self, a,b,c):
        return tensorContract(self.C2S(a,e_),
                              self.Y2S(e_,f_),
                              self.h(f_,b,c))
    
    def h2_19(self, a,b,c):
        return tensorContract(self.l(a,b,e_,f_),
                              self.Y2S(f_,g_),
                              self.h(e_,g_,c))
    
    def h2_20(self, a,b,c):
        return ( tensorContract(self.M(i_,j_),
                                self.T(A_,j_,k_),
                                self.yt(a,k_,l_),
                                self.y(b,l_,m_),
                                self.G(A_,B_),
                                self.T(B_,m_,n_),
                                self.yt(c,n_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)
                
               + tensorContract(self.y(a,i_,j_),
                                self.T(A_,j_,k_),
                                self.Mt(k_,l_),
                                self.y(b,l_,m_),
                                self.G(A_,B_),
                                self.T(B_,m_,n_),
                                self.yt(c,n_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)
                
               + tensorContract(self.y(a,i_,j_),
                                self.T(A_,j_,k_),
                                self.yt(b,k_,l_),
                                self.M(l_,m_),
                                self.G(A_,B_),
                                self.T(B_,m_,n_),
                                self.yt(c,n_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)
                
               + tensorContract(self.y(a,i_,j_),
                                self.T(A_,j_,k_),
                                self.yt(b,k_,l_),
                                self.y(c,l_,m_),
                                self.G(A_,B_),
                                self.T(B_,m_,n_),
                                self.Mt(n_,i_),
                                doTrace=True, yukSorting=self.model.YukPos) )
    
    def h2_21(self, a,b,c):
        return ( 2*tensorContract(self.C2S(a,e_),
                                  self.y(e_,i_,j_),
                                  self.Mt(j_,k_),
                                  self.y(b,k_,l_),
                                  self.yt(c,l_,i_),
                                  doTrace=True, yukSorting=self.model.YukPos)
                
                 + tensorContract(self.C2S(a,e_),
                                  self.y(e_,i_,j_),
                                  self.yt(b,j_,k_),
                                  self.M(k_,l_),
                                  self.yt(c,l_,i_),
                                  doTrace=True, yukSorting=self.model.YukPos))

    def h2_22(self, a,b,c):
        return ( tensorContract(self.M(i_,j_),
                                self.yt(a,j_,k_),
                                self.y(b,k_,l_),
                                self.yt(c,l_,m_),
                                self.C2F(m_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)
                
               + tensorContract(self.y(a,i_,j_),
                                self.Mt(j_,k_),
                                self.y(b,k_,l_),
                                self.yt(c,l_,m_),
                                self.C2F(m_,i_),
                                doTrace=True, yukSorting=self.model.YukPos) )
    
    def h2_23(self, a,b,c):
        return tensorContract(self.M(i_,j_),
                              self.yt(e_,j_,k_),
                              self.y(a,k_,l_),
                              self.yt(f_,l_,i_),
                              self.l(e_,f_,b,c),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def h2_24(self, a,b,c):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(e_,j_,k_),
                              self.y(b,k_,l_),
                              self.yt(f_,l_,i_),
                              self.h(e_,f_,c),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def h2_25(self, a,b,c):
        return tensorContract(self.M(i_,j_),
                              self.yt(a,j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(f_,l_,i_),
                              self.l(e_,f_,b,c),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def h2_26(self, a,b,c):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(f_,l_,i_),
                              self.h(e_,f_,c),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def h2_27(self, a,b,c):
        return tensorContract(self.Y4S(a,e_),
                              self.h(e_,b,c))
    
    def h2_28(self, a,b,c):
        return tensorContract(self.Y2SYF(a,e_),
                              self.h(e_,b,c))
    
    def h2_29(self, a,b,c):
        return ( 2*tensorContract(self.M(i_,j_),
                                  self.yt(a,j_,k_),
                                  self.y(b,k_,l_),
                                  self.yt(e_,l_,m_),
                                  self.y(c,m_,n_),
                                  self.yt(e_,n_,i_),
                                  doTrace=True, yukSorting=self.model.YukPos)
                
                 + tensorContract(self.y(a,i_,j_),
                                  self.Mt(j_,k_),
                                  self.y(b,k_,l_),
                                  self.yt(e_,l_,m_),
                                  self.y(c,m_,n_),
                                  self.yt(e_,n_,i_),
                                  doTrace=True, yukSorting=self.model.YukPos)
                 
                 + tensorContract(self.y(a,i_,j_),
                                  self.yt(b,j_,k_),
                                  self.y(c,k_,l_),
                                  self.yt(e_,l_,m_),
                                  self.M(m_,n_),
                                  self.yt(e_,n_,i_),
                                  doTrace=True, yukSorting=self.model.YukPos) )
    
    def h2_30(self, a,b,c):
        return tensorContract(self.M(i_,j_),
                              self.yt(a,j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(b,l_,m_),
                              self.y(c,m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
    
    def h2_31(self, a,b,c):
        return ( tensorContract(self.M(i_,j_),
                                self.yt(a,j_,k_),
                                self.y(b,k_,l_),
                                self.yt(c,l_,m_),
                                self.Y2F(m_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)
                
               + tensorContract(self.y(a,i_,j_),
                                self.Mt(j_,k_),
                                self.y(b,k_,l_),
                                self.yt(c,l_,m_),
                                self.Y2F(m_,i_),
                                doTrace=True, yukSorting=self.model.YukPos) )
               

    