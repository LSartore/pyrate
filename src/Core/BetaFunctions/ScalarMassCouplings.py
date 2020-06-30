# -*- coding: utf-8 -*-

from sympy import Rational as r
from .BetaFunction import BetaFunction
from Definitions import tensorContract

import itertools


class ScalarMassBetaFunction(BetaFunction):

    def compute(self, a,b, nLoops):
        perm = list(itertools.permutations([a, b], 2))
        permSet = set(perm)
        coeff = r(len(perm),len(permSet))

        ret = 0
        for s1,s2 in permSet:
            ret += coeff * self.Beta(s1,s2, nLoops=nLoops)

        return r(1,2)*ret

    def fDefinitions(self):
        """ Functions definition """

        for i in range(self.nLoops):
            self.functions.append([])
            count = 1
            while True:
                try:
                    self.functions[i].append(eval(f"self.m{i+1}_{count}"))
                    count += 1
                except:
                    break


    def cDefinitions(self):
        """ Coefficients definition """

        ## 1-loop

        self.coefficients.append( [r(-6), r(1), r(1), r(1), r(-4), r(-2)] )

        ## 2-loop

        # self.coefficients.append( [r(2), r(10), r(0), r(3), r(-143,6), r(11,6),
        #                            r(10,6), r(-3), r(8), r(8), r(-3), r(-3),
        #                            r(1,6), r(-1), r(-1,2), r(-2), r(0), r(0),
        #                            r(-12), r(5), r(0), r(-1), r(-1), r(0),
        #                            r(2), r(-4), r(2), r(4), r(1), r(0),
        #                            r(0), r(0), r(-1), r(-3,2), r(4), r(4),
        #                            r(2)] )

        self.coefficients.append( [r(2), r(10), r(0), r(3), r(-143,6), r(11,6),
                                   r(10,6), r(-3), r(8), r(8), r(-3), r(-3),
                                   r(1,6), r(-1), r(-1,2), r(-2), r(0), r(0),
                                   r(-12), r(5), r(0), r(-1), r(-1), r(0),
                                   r(0), r(0), r(2), r(4), r(-8), r(-8),
                                   r(-4), r(-4), r(2), r(4), r(1), r(0),
                                   r(0), r(0), r(-1), r(-3,2), r(4), r(8),
                                   r(8), r(4), r(4), r(4), r(4), r(4),
                                   r(4), r(2), r(2)] )

    ######################
    #  1-loop functions  #
    ######################

    def m1_1(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.mu(e_,b))

    def m1_2(self, a,b):
        return tensorContract(self.l(a,b,e_,f_),
                              self.mu(e_,f_))
    def m1_3(self, a,b):
        return tensorContract(self.h(a,e_,f_),
                              self.h(b,e_,f_))

    def m1_4(self, a,b):
        return tensorContract(self.Y2S(a,e_),
                              self.mu(e_,b))

    def m1_5(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.M(k_,l_),
                              self.Mt(l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m1_6(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(b,k_,l_),
                              self.Mt(l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    ######################
    #  2-loop functions  #
    ######################

    def m2_1(self, a,b):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,e_),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,b,j_),
                              self.Ts(D_,j_,f_),
                              self.mu(e_,f_))

    def m2_2(self, a,b):
        return tensorContract(self.Ts(A_,a,i_),
                              self.Ts(C_,i_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.Ts(B_,e_,j_),
                              self.Ts(D_,j_,f_),
                              self.mu(e_,f_))

    def m2_3(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.C2S(b,f_),
                              self.mu(e_,f_))

    def m2_4(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.C2S(e_,f_),
                              self.mu(f_,b))

    def m2_5(self, a,b):
        return tensorContract(self.C2SG(a,e_),
                              self.mu(e_,b))

    def m2_6(self, a,b):
        return tensorContract(self.C2SS(a,e_),
                              self.mu(e_,b))

    def m2_7(self, a,b):
        return tensorContract(self.C2SF(a,e_),
                              self.mu(e_,b))

    def m2_8(self, a,b):
        return tensorContract(self.Ts(A_,a,e_),
                              self.Ts(B_,b,f_),
                              self.G(A_,B_),
                              self.l(e_,f_,g_,h_),
                              self.mu(g_,h_))

    def m2_9(self, a,b):
        return tensorContract(self.l(a,b,e_,f_),
                              self.C2S(f_,g_),
                              self.mu(e_,g_))

    def m2_10(self, a,b):
        return tensorContract(self.h(a,e_,f_),
                              self.C2S(f_,g_),
                              self.h(e_,g_,b))

    def m2_11(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.l(e_,b,f_,g_),
                              self.mu(f_,g_))

    def m2_12(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.h(e_,f_,g_),
                              self.h(f_,g_,b))

    def m2_13(self, a,b):
        return tensorContract(self.l(a,e_,f_,g_),
                              self.l(e_,f_,g_,h_),
                              self.mu(h_,b))

    def m2_14(self, a,b):
        return tensorContract(self.l(a,e_,g_,h_),
                              self.l(b,f_,g_,h_),
                              self.mu(e_,f_))

    def m2_15(self, a,b):
        return tensorContract(self.l(a,b,e_,f_),
                              self.h(e_,g_,h_),
                              self.h(f_,g_,h_))

    def m2_16(self, a,b):
        return tensorContract(self.h(a,e_,f_),
                              self.h(e_,g_,h_),
                              self.l(f_,g_,h_,b))

    def m2_17(self, a,b):
        return tensorContract(self.l(a,b,e_,f_),
                              self.l(e_,f_,g_,h_),
                              self.mu(g_,h_))

    def m2_18(self, a,b):
        return tensorContract(self.h(a,e_,f_),
                              self.l(e_,f_,g_,h_),
                              self.h(g_,h_,b))

    def m2_19(self, a,b):
        return tensorContract(self.Ts(A_,a,e_),
                              self.Ts(C_,e_,b),
                              self.G(A_,B_),
                              self.G(C_,D_),
                              self.T(D_,i_,j_),
                              self.T(B_,j_,k_),
                              self.Mt(k_,l_),
                              self.M(l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_20(self, a,b):
        return tensorContract(self.Y2SCF(a,e_),
                              self.mu(e_,b))

    def m2_21(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.Y2S(e_,f_),
                              self.mu(f_,b))

    def m2_22(self, a,b):
        return tensorContract(self.l(a,b,e_,f_),
                              self.Y2S(f_,g_),
                              self.mu(e_,g_))

    def m2_23(self, a,b):
        return tensorContract(self.h(a,e_,f_),
                              self.Y2S(f_,g_),
                              self.h(e_,g_,b))

    def m2_24(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.T(A_,j_,k_),
                              self.Mt(k_,l_),
                              self.M(l_,m_),
                              self.G(A_,B_),
                              self.T(B_,m_,n_),
                              self.yt(b,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_25(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.T(A_,j_,k_),
                              self.yt(b,k_,l_),
                              self.M(l_,m_),
                              self.G(A_,B_),
                              self.T(B_,m_,n_),
                              self.Mt(n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_26(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.T(A_,j_,k_),
                              self.Mt(k_,l_),
                              self.y(b,l_,m_),
                              self.G(A_,B_),
                              self.T(B_,m_,n_),
                              self.Mt(n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_27(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.y(e_,i_,j_),
                              self.Mt(j_,k_),
                              self.y(b,k_,l_),
                              self.Mt(l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_28(self, a,b):
        return tensorContract(self.C2S(a,e_),
                              self.y(e_,i_,j_),
                              self.yt(b,j_,k_),
                              self.M(k_,l_),
                              self.Mt(l_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_29(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                                self.yt(b,j_,k_),
                                self.M(k_,l_),
                                self.Mt(l_,m_),
                                self.C2F(m_,i_),
                                doTrace=True, yukSorting=self.model.YukPos)

    def m2_30(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(b,k_,l_),
                              self.Mt(l_,m_),
                              self.C2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_31(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.M(k_,l_),
                              self.yt(b,l_,m_),
                              self.C2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_32(self, a,b):
        return tensorContract(self.M(i_,j_),
                              self.yt(a,j_,k_),
                              self.y(b,k_,l_),
                              self.Mt(l_,m_),
                              self.C2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_33(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(e_,j_,k_),
                              self.y(b,k_,l_),
                              self.yt(f_,l_,i_),
                              self.mu(e_,f_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_34(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(e_,j_,k_),
                              self.M(k_,l_),
                              self.yt(f_,l_,i_),
                              self.h(e_,f_,b),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_35(self, a,b):
        return tensorContract(self.M(i_,j_),
                              self.yt(e_,j_,k_),
                              self.M(k_,l_),
                              self.yt(f_,l_,i_),
                              self.l(e_,f_,a,b),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_36(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(f_,l_,i_),
                              self.mu(e_,f_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_37(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(f_,l_,i_),
                              self.h(e_,f_,b),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_38(self, a,b):
        return tensorContract(self.M(i_,j_),
                              self.Mt(j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(f_,l_,i_),
                              self.l(e_,f_,a,b),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_39(self, a,b):
        return tensorContract(self.Y4S(a,e_),
                              self.mu(e_,b))

    def m2_40(self, a,b):
        return tensorContract(self.Y2SYF(a,e_),
                              self.mu(e_,b))

    def m2_41(self, a,b):
        return tensorContract(self.M(i_,j_),
                              self.yt(a,j_,k_),
                              self.M(k_,l_),
                              self.yt(e_,l_,m_),
                              self.y(b,m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_42(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.M(k_,l_),
                              self.yt(e_,l_,m_),
                              self.M(m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_43(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.M(k_,l_),
                              self.yt(e_, l_, m_),
                              self.y(b, m_, n_),
                              self.yt(e_, n_, i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_44(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(b,k_,l_),
                              self.yt(e_,l_,m_),
                              self.M(m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_45(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.y(e_,k_,l_),
                              self.Mt(l_,m_),
                              self.M(m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_46(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(e_,k_,l_),
                              self.Mt(l_,m_),
                              self.y(b,m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_47(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(e_,k_,l_),
                              self.yt(b,l_,m_),
                              self.M(m_,n_),
                              self.yt(e_,n_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_48(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.yt(b,j_,k_),
                              self.M(k_,l_),
                              self.Mt(l_,m_),
                              self.Y2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_49(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.y(b,k_,l_),
                              self.Mt(l_,m_),
                              self.Y2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_50(self, a,b):
        return tensorContract(self.y(a,i_,j_),
                              self.Mt(j_,k_),
                              self.M(k_,l_),
                              self.yt(b,l_,m_),
                              self.Y2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)

    def m2_51(self, a,b):
        return tensorContract(self.M(i_,j_),
                              self.yt(a,j_,k_),
                              self.y(b,k_,l_),
                              self.Mt(l_,m_),
                              self.Y2F(m_,i_),
                              doTrace=True, yukSorting=self.model.YukPos)
