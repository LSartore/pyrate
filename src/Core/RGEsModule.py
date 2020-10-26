# -*- coding: utf-8 -*-

from Logging import loggingInfo, loggingCritical
from sys import exit

from sympy import (I, Identity, Matrix, Rational, SparseMatrix, Symbol,
                   re, im, transpose)
import itertools

from Definitions import TensorDic, Tensor, tensorContract, tensorAdd, tensorMul
from Math import sMat, sEye as eye

import time

class RGEsModule():
    """ This class contains information about the various tensor quantities
        required to perform the RG computations. """

    def __init__(self, model):
        self.model = model
        self.nG = len(model.gaugeGroups)
        self.groupDims = tuple([g.dim for g in model.gaugeGroupsList])

        self.YDic = {} #Yukawa
        self.LambdaDic = {} #Quartic
        self.Hdic = {}  #Trilinear
        self.MSdic = {} #Scalar mass
        self.MFdic = {} #Fermion mass
        self.Vdic = {}  #Vevs
        self.gammaFdic = {} #Fermion anomalous
        self.gammaSdic = {} #Scalar anomalous

        self.nF = len(model.allFermions)
        self.nS = len(model.allScalars)

        self.TsDic = {}
        self.TDic = {}

        #Gauge indices
        self.gi = sum([[(u, A) for A in range(self.groupDims[u])] for u in range(self.nG)], [])
        self.nGi = len(self.gi)

        self.Ugauge = tuple([(pos,0) for pos,g in enumerate(model.gaugeGroupsList) if g.abelian])
        self.NUgauge = tuple([el for el in self.gi if model.gaugeGroupsList[el[0]].abelian == False])

        #Kinetic Mixing
        self.kinMix = model.kinMix
        if self.kinMix:
            self.kinMat = model.kinMat
            self.kinMat2 = model.kinMat2

    def initialize(self):
        loggingInfo("Initializing tensor quantities...", end=' ')
        t0 = time.time()

        # Build the representation matrices for the whole (semi)simple gauge group
        self.constructT()
        self.constructTs()

        # Remove 'True' from tuples with adjoint Yuk/FM matrices
        for k in list(self.YDic.keys()):
            if k[-1] is True:
                newK = k[:3]
                if newK not in self.YDic:
                    self.YDic[newK] = 0
                self.YDic[newK] += self.YDic[k]
                del self.YDic[k]

        for k in list(self.MFdic.keys()):
            if k[-1] is True:
                newK = k[:2]
                if newK not in self.MFdic:
                    self.MFdic[newK] = 0
                self.MFdic[newK] += self.MFdic[k]
                del self.MFdic[k]

        #Initialize tensors
        self.initTensors()

        loggingInfo("Done." + (f" ({time.time()-t0:.3f} seconds)" if self.model.times else ''))

        if self.model.runSettings['CheckGaugeInvariance'] is True:
            self.checkGaugeInvariance()
        else:
            loggingInfo("Skipping gauge invariance check.")


        # Close the DB, since all stored object must have been read by now
        self.model.idb.close()

    def constructT(self):
        """ Construct the fermion gauge generators """

        def identity(nGen):
            if isinstance(nGen, Symbol) or nGen > 1:
                return Identity(nGen)
            return 1

        model = self.model
        repMat = []
        dimR = 0
        nonAbelGroupNames = [gName for gName, g in model.gaugeGroups.items() if g.abelian is False]
        for gPos, (gName, g) in enumerate(model.gaugeGroups.items()):
            if g.abelian :
                for fName, f in model.allFermions.items():
                    if f[1].Qnb[gName] != 0:
                        self.TDic[((gPos,0), f[0], f[0])] = identity(f[1].gen)*Rational(f[1].Qnb[gName])

            else:
                nAbelPos = nonAbelGroupNames.index(gName)

                for fName, f in model.Fermions.items():
                    dimR = g.dimR(f.Qnb[gName])

                    if dimR == 1:
                        continue

                    padding = model.allFermions[fName + '[' + ', '.join(['0']*len(f.indexStructure)) + ']'][0]
                    repMat = g.repMat(f.Qnb[gName])

                    for A in range(g.dim):
                        kdMats = [(repMat[A] if iPos == nAbelPos else eye(iRange)) for iPos, iRange in enumerate(f.fullIndexStructure)]

                        rMat = kdMats[0]
                        for m in kdMats[1:]:
                            rMat = rMat.kroneckerProduct(m)

                        for (i, j), val in rMat._smat.items():
                            self.TDic[((gPos,A), padding + i, padding + j)] = identity(f.gen)*val

    def constructTs(self):
        """ Construct the scalar gauge generators """

        model = self.model
        repMat = []
        dimR = 0
        nonAbelGroupNames = [gName for gName, g in model.gaugeGroups.items() if g.abelian is False]

        O = sMat([[1, I], [-I, 1]])

        for gPos, (gName, g) in enumerate(model.gaugeGroups.items()):

            ###################
            #  Real  scalars  #
            ###################

            if not g.abelian:
                nAbelPos = nonAbelGroupNames.index(gName)
                for sName, s in model.Scalars.items():
                    if s.fromCplx:
                        continue

                    dimR = g.dimR(s.Qnb[gName])

                    if dimR == 1:
                        continue

                    padding = model.allScalars[sName + '[' + ', '.join(['0']*len(s.indexStructure)) + ']'][0]
                    repMat = g.repMat(s.Qnb[gName])

                    for A in range(g.dim):
                        kdMats = [(repMat[A] if iPos == nAbelPos else eye(iRange)) for iPos, iRange in enumerate(s.fullIndexStructure)]

                        rMat = kdMats[0]
                        for m in kdMats[1:]:
                            rMat = rMat.kroneckerProduct(m)

                        for (i, j), val in rMat._smat.items():
                            self.TsDic[((gPos,A), padding + i, padding + j)] = val

            ###################
            # Complex scalars #
            ###################

            for sName, s in model.ComplexScalars.items():
                if s.conj:
                    continue

                if not g.abelian:
                    nAbelPos = nonAbelGroupNames.index(gName)
                    dimR = g.dimR(s.Qnb[gName])

                    if dimR == 1:
                        continue

                padding = model.allScalars[str(s.realFields[0]) + '[' + ', '.join(['0']*len(s.indexStructure)) + ']'][0]

                if g.abelian:
                    t = s.Qnb[gName]

                    if t == 0:
                        continue

                    reRepMat = I * (O*t).imag()

                    rMat = reRepMat
                    for iRange in s.fullIndexStructure:
                        rMat = rMat.kroneckerProduct(eye(iRange))

                    for (i, j), val in rMat._smat.items():
                        self.TsDic[((gPos,0), padding + i, padding + j)] = val

                else:
                    repMat = g.repMat(s.Qnb[gName])

                    for A in range(g.dim):
                        kdMats = [(repMat[A] if iPos == nAbelPos else eye(iRange)) for iPos, iRange in enumerate(s.fullIndexStructure)]

                        rMat = O
                        for m in kdMats:
                            rMat = rMat.kroneckerProduct(m)

                        rMat = I*rMat.imag()
                        for (i, j), val in rMat._smat.items():
                            self.TsDic[((gPos,A), padding + i, padding + j)] = val

    def initTensors(self):
        ############################
        # Gauge-related quantities #
        ############################

        #G
        self.G = Tensor((self.nGi, self.nGi))
        for A in self.gi:
            for B in self.gi:
                tmp = self.G_(A,B)
                if tmp != 0:
                    self.G.dic[(A,B)] = tmp

        #f
        self.f = Tensor((self.nGi, self.nGi, self.nGi))
        for A in self.gi:
            for B in self.gi:
                if B[0] != A[0]:
                    continue
                for C in self.gi:
                    if C[0] != B[0]:
                        continue
                    tmp = self.f_(A,B,C)
                    if tmp != 0:
                        self.f.dic[(A,B,C)] = tmp

        #T (fermions)
        self.T = Tensor((self.nGi, self.nF, self.nF))
        for A in self.gi:
            for k,v in self.TDic.items():
                self.T.dic[k] = v

        self.Tt = Tensor((self.nGi, self.nF, self.nF))
        self.Tt.dic = self.dicTilde(self.T.dic, [1,2])

        #T (scalars)
        self.Ts = Tensor((self.nGi, self.nS, self.nS))
        for A in self.gi:
            for k,v in self.TsDic.items():
                self.Ts.dic[k] = v


        #########################
        # Lagrangian quantities #
        #########################

        # Yukawa

        self.y = Tensor((self.nS, self.nF, self.nF))
        for k,v in self.YDic.items():
            self.y.dic[k] = v
            self.y.dic[(k[0], k[2], k[1])] = transpose(v)

        self.yt = Tensor((self.nS, self.nF, self.nF))
        self.yt.dic = self.dicTilde(self.y.dic, [1,2])

        # Quartic

        self.l = Tensor((self.nS, self.nS, self.nS, self.nS), sym=True)
        for k,v in self.LambdaDic.items():
            self.l.dic[k] = v

        # Fermion Mass

        self.M = Tensor((self.nF, self.nF))
        for k,v in self.MFdic.items():
            self.M.dic[k] = v
            self.M.dic[(k[1], k[0])] = transpose(v)

        self.Mt = Tensor((self.nF, self.nF))
        self.Mt.dic = self.dicTilde(self.M.dic, [0,1])

        # Trilinear

        self.h = Tensor((self.nS, self.nS, self.nS), sym=True)
        for k,v in self.Hdic.items():
            self.h.dic[k] = v

        # Scalar Mass

        self.mu = Tensor((self.nS, self.nS), sym=True)
        for k,v in self.MSdic.items():
            self.mu.dic[k] = v

        # VeVs

        self.v = Tensor((self.nS,))
        for k,val in self.Vdic.items():
            self.v.dic[k] = val

        #############################################
        #####  Definition of 2-point functions  #####
        #############################################

        def requirement(gauge, yukawa, quartic):
            g = self.model.loopDic['GaugeCouplings'] >= gauge if gauge is not None else False
            y = max(self.model.loopDic['Yukawas'],
                    self.model.loopDic['FermionMasses']) >= yukawa if yukawa is not None else False
            q = max(self.model.loopDic['QuarticTerms'],
                    self.model.loopDic['TrilinearTerms'],
                    self.model.loopDic['ScalarMasses'],
                    self.model.loopDic['Vevs']) >= quartic if quartic is not None else False

            return (g or y or q)

        ################
        # 1-loop gauge #
        ################

        if requirement(1,2,2):
            self.S2F = Tensor((self.nGi, self.nGi),
                               tensorContract(self.T(A_,i_,j_), self.T(B_,j_,i_),
                                              doTrace=True))

            self.S2S = Tensor((self.nGi, self.nGi),
                               tensorContract(self.Ts(A_,i_,j_), self.Ts(B_,j_,i_), doTrace=True))

            self.C2G = Tensor((self.nGi, self.nGi),
                               tensorContract(self.f(A_,C_,D_), self.f(C_,D_,B_)))

        ##################
        # 1-loop fermion #
        ##################

        if requirement(1,1,2):
            self.C2F = Tensor((self.nF, self.nF),
                               tensorContract(self.G(A_,B_),
                                              self.T(A_,i_,k_),
                                              self.T(B_,k_,j_)))

            self.C2Ft = Tensor((self.nF, self.nF))
            self.C2Ft.dic = self.dicTilde(self.C2F.dic, [0,1])

            self.Y2F = Tensor((self.nF, self.nF),
                                tensorContract(self.y(a_,i_,k_),
                                               self.yt(a_,k_,j_)))

            self.Y2Ft = Tensor((self.nF, self.nF))
            self.Y2Ft.dic = self.dicTilde(self.Y2F.dic, [0,1])

        #################
        # 1-loop scalar #
        #################

        if requirement(1,1,1):
            self.C2S = Tensor((self.nS, self.nS),
                               tensorContract(self.G(A_,B_),
                                              self.Ts(A_,a_,c_),
                                              self.Ts(B_,c_,b_)))

            self.Y2S = Tensor((self.nS, self.nS),
                                tensorContract(self.y(a_,i_,j_),
                                               self.yt(b_,j_,i_),
                                               doTrace=True, yukSorting=self.model.YukPos))

        ################
        # 2-loop gauge #
        ################

        if requirement(2,None,None):
            self.S2FCF = Tensor((self.nGi, self.nGi),
                                tensorContract(self.C2F(i_,j_),
                                               self.T(A_,j_,k_),
                                               self.T(B_,k_,i_),
                                               doTrace=True))

            self.S2FYF = Tensor((self.nGi, self.nGi),
                                tensorContract(self.Y2F(i_,j_),
                                               self.T(A_,j_,k_),
                                               self.T(B_,k_,i_),
                                                doTrace=True, yukSorting=self.model.YukPos))

            self.S2SCS = Tensor((self.nGi, self.nGi),
                                tensorContract(self.C2S(a_,b_),
                                               self.Ts(A_,b_,c_),
                                               self.Ts(B_,c_,a_)))

            self.S2SYS = Tensor((self.nGi, self.nGi),
                                tensorContract(self.Y2S(a_,b_),
                                               self.Ts(A_,b_,c_),
                                               self.Ts(B_,c_,a_),
                                               freeDummies=[A_,B_]))

        ##################
        # 2-loop fermion #
        ##################

        if requirement(3,2,None):
            self.C2FG = Tensor((self.nF, self.nF),
                                tensorContract(self.G(A_,C_),
                                               self.C2G(C_,D_),
                                               self.G(D_,B_),
                                               self.T(A_,i_,j_),
                                               self.T(B_,j_,k_)))

            self.C2FS = Tensor((self.nF, self.nF),
                                tensorContract(self.G(A_,C_),
                                               self.S2S(C_,D_),
                                               self.G(D_,B_),
                                               self.T(A_,i_,j_),
                                               self.T(B_,j_,k_)))

            self.C2FF = Tensor((self.nF, self.nF),
                                tensorContract(self.G(A_,C_),
                                               self.S2F(C_,D_),
                                               self.G(D_,B_),
                                               self.T(A_,i_,j_),
                                               self.T(B_,j_,k_)))

            self.Y2FCF = Tensor((self.nF, self.nF),
                                tensorContract(self.y(a_,i_,j_),
                                               self.C2F(j_,k_),
                                               self.yt(a_,k_,l_)))

            self.Y2FCS = Tensor((self.nF, self.nF),
                                tensorContract(self.y(a_,i_,j_),
                                               self.yt(b_,j_,k_),
                                               self.C2S(a_,b_)))

            self.Y2FYF = Tensor((self.nF, self.nF),
                                tensorContract(self.y(a_,i_,j_),
                                               self.Y2Ft(j_,k_),
                                               self.yt(a_,k_,l_)))

            self.Y2FYFt = Tensor((self.nF, self.nF))
            self.Y2FYFt.dic = self.dicTilde(self.Y2FYF.dic, [0,1])


            self.Y2FYS = Tensor((self.nF, self.nF),
                                tensorContract(self.Y2S(a_,b_),
                                               self.y(a_,i_,j_),
                                               self.yt(b_,j_,k_) ))

            self.Y2FYSt = Tensor((self.nF, self.nF))
            self.Y2FYSt.dic = self.dicTilde(self.Y2FYS.dic, [0,1])


            self.Y4F = Tensor((self.nF, self.nF),
                              tensorContract(self.y(a_,i_,j_),
                                             self.yt(b_,j_,k_),
                                             self.y(a_,k_,l_),
                                             self.yt(b_,l_,m_)))

            if self.model.fermionAnomalous != {}:
                self.C2FGt = Tensor((self.nF, self.nF))
                self.C2FGt.dic = self.dicTilde(self.C2FG.dic, [0,1])

                self.C2FSt = Tensor((self.nF, self.nF))
                self.C2FSt.dic = self.dicTilde(self.C2FS.dic, [0,1])

                self.C2FFt = Tensor((self.nF, self.nF))
                self.C2FFt.dic = self.dicTilde(self.C2FF.dic, [0,1])

        #################
        # 2-loop scalar #
        #################

        if requirement(3,2,2):
            self.C2SG = Tensor((self.nS, self.nS),
                               tensorContract(self.G(A_,C_),
                                              self.C2G(C_,D_),
                                              self.G(D_,B_),
                                              self.Ts(A_,a_,b_),
                                              self.Ts(B_,b_,c_)))

            self.C2SS = Tensor((self.nS, self.nS),
                               tensorContract(self.G(A_,C_),
                                              self.S2S(C_,D_),
                                              self.G(D_,B_),
                                              self.Ts(A_,a_,b_),
                                              self.Ts(B_,b_,c_)))

            self.C2SF = Tensor((self.nS, self.nS),
                               tensorContract(self.G(A_,C_),
                                              self.S2F(C_,D_),
                                              self.G(D_,B_),
                                              self.Ts(A_,a_,b_),
                                              self.Ts(B_,b_,c_)))

            self.Y2SCF = Tensor((self.nS, self.nS),
                                tensorContract(self.y(a_,i_,j_),
                                               self.C2F(j_,k_),
                                               self.yt(b_,k_,i_),
                                               doTrace=True, yukSorting=self.model.YukPos))

            self.Y2SYF = Tensor((self.nS, self.nS),
                                tensorContract(self.y(a_,i_,j_),
                                               self.Y2Ft(j_,k_),
                                               self.yt(b_,k_,i_),
                                               doTrace=True, yukSorting=self.model.YukPos))

            self.Y4S = Tensor((self.nS, self.nS),
                              tensorContract(self.y(a_,i_,j_),
                                              self.yt(c_,j_,k_),
                                              self.y(b_,k_,l_),
                                              self.yt(c_,l_,i_),
                                              doTrace=True, yukSorting=self.model.YukPos))


    def G_(self, A, B):
        if self.nonZeroGauge(A,B):
            if not self.kinMix or A not in self.Ugauge or B not in self.Ugauge:
                return self.model.gaugeGroupsList[A[0]].g**2
            else:
                i,j = self.Ugauge.index(A),self.Ugauge.index(B)
                return self.kinMat2[i,j]
        else:
            return 0


    def f_(self, A, B, C):
        if not A[0] == B[0] == C[0]:
            return 0
        if self.model.gaugeGroupsList[A[0]].abelian:
            return 0

        return self.model.gaugeGroupsList[A[0]].structureConstants[A[1]][B[1],C[1]]


    # TESTS - DEBUG

    def Gmat(self):
        return SparseMatrix(self.nGi, self.nGi, {(self.gi.index(i), self.gi.index(j)):self.G_(i,j) for i in self.gi for j in self.gi})

    def ymat(self, a):
        return SparseMatrix(self.nF, self.nF, {k[1:]:v for k,v in self.y.dic.items() if k[0]==a})

    def ytmat(self, a):
        return SparseMatrix(self.nF, self.nF, {k[1:]:v for k,v in self.yt.dic.items() if k[0]==a})

    def Mmat(self):
        return SparseMatrix(self.nF, self.nF, {k:v for k,v in self.M.dic.items()})

    def Mtmat(self):
        return SparseMatrix(self.nF, self.nF, {k:v for k,v in self.Mt.dic.items()})

    def Y2Fmat(self):
        return SparseMatrix(self.nF, self.nF, {k:v for k,v in self.Y2F.dic.items()})

    def Y2Ftmat(self):
        return SparseMatrix(self.nF, self.nF, {k:v for k,v in self.Y2Ft.dic.items()})

    def Y2Smat(self):
        return SparseMatrix(self.nS, self.nS, {k:v for k,v in self.Y2S.dic.items()})

    def Tmat(self, A):
        return SparseMatrix(self.nF, self.nF, {k[1:]:v for k,v in self.T.dic.items() if k[0]==A})

    def Tsmat(self, A):
        return SparseMatrix(self.nS, self.nS, {k[1:]:v for k,v in self.Ts.dic.items() if k[0]==A})

    def S2Fmat(self):
        return SparseMatrix(self.nGi, self.nGi, {(self.gi.index(k[0]), self.gi.index(k[1])):v for k,v in self.S2F.dic.items()})

    def S2Smat(self):
        return SparseMatrix(self.nGi, self.nGi, {(self.gi.index(k[0]), self.gi.index(k[1])):v for k,v in self.S2S.dic.items()})

    def C2Fmat(self):
        return SparseMatrix(self.nF, self.nF, {k:v for k,v in self.C2F.dic.items()})

    def printLagrangian(self):
        ret = 0
        sList = [el[3] for el in self.model.AllScalars.values()]
        for a,b,c,d in itertools.product(*([range(self.nS)]*4)):
            ret += self.Lambda(a,b,c,d) * sList[a]*sList[b]*sList[c]*sList[d]

        return ret/24

    def printTerm(self, a,b,c,d):
        ret = 0
        sList = [el[3] for el in self.model.AllScalars.values()]
        ret += self.Lambda(a,b,c,d) * sList[a]*sList[b]*sList[c]*sList[d]

        return ret/24


    def checkGaugeInvariance(self):
        loggingInfo("Checking gauge invariance ...", end=' ')
        t0 = time.time()

        yuk = tensorAdd(tensorMul(-1, tensorContract(self.Tt(A_,i_,j_),
                                                     self.y(a_,j_,k_),
                                                     freeDummies=[A_,a_,i_,k_],
                                                     doit=True)) ,
                        tensorContract(self.y(a_,i_,j_),
                                       self.T(A_,j_,k_),
                                       freeDummies=[A_,a_,i_,k_],
                                       doit=True) ,
                        tensorContract(self.y(b_,i_,k_),
                                       self.Ts(A_,b_,a_),
                                       freeDummies=[A_,a_,i_,k_],
                                       doit=True) )

        fermionMass = tensorAdd(tensorMul(-1, tensorContract(self.Tt(A_,i_,j_),
                                                             self.M(j_,k_),
                                                             freeDummies=[A_,i_,k_],
                                                             doit=True)) ,
                                tensorContract(self.M(i_,j_),
                                               self.T(A_,j_,k_),
                                               freeDummies=[A_,i_,k_],
                                               doit=True) )

        quartics = tensorAdd(tensorContract(self.Ts(A_,a_,e_),
                                            self.l(e_,b_,c_,d_),
                                            freeDummies=[A_,a_,b_,c_,d_],
                                            doit=True) ,
                             tensorContract(self.Ts(A_,b_,e_),
                                            self.l(a_,e_,c_,d_),
                                            freeDummies=[A_,a_,b_,c_,d_],
                                            doit=True) ,
                             tensorContract(self.Ts(A_,c_,e_),
                                            self.l(a_,b_,e_,d_),
                                            freeDummies=[A_,a_,b_,c_,d_],
                                            doit=True) ,
                             tensorContract(self.Ts(A_,d_,e_),
                                            self.l(a_,b_,c_,e_),
                                            freeDummies=[A_,a_,b_,c_,d_],
                                            doit=True) )


        trilinears = tensorAdd(tensorContract(self.Ts(A_,a_,e_),
                                              self.h(e_,b_,c_),
                                              freeDummies=[A_,a_,b_,c_],
                                              doit=True) ,
                               tensorContract(self.Ts(A_,b_,e_),
                                              self.h(a_,e_,c_),
                                              freeDummies=[A_,a_,b_,c_],
                                              doit=True) ,
                               tensorContract(self.Ts(A_,c_,e_),
                                              self.h(a_,b_,e_),
                                              freeDummies=[A_,a_,b_,c_],
                                              doit=True) )


        scalarMass = tensorAdd(tensorContract(self.Ts(A_,a_,e_),
                                              self.mu(e_,b_),
                                              freeDummies=[A_,a_,b_],
                                              doit=True) ,
                               tensorContract(self.Ts(A_,b_,e_),
                                              self.mu(a_,e_),
                                              freeDummies=[A_,a_,b_],
                                              doit=True) )


        def which(dic):
            problematicCouplings = {}
            for k, el in dic.items():
                names = [obj for obj in el.atoms() if not obj.is_number and not (hasattr(obj, 'is_Identity') and obj.is_Identity)]
                for c in names:
                    if str(c) not in problematicCouplings:
                        problematicCouplings[str(c)] = set()
                    problematicCouplings[str(c)].add(k[0][0])

            return "\n\t" + "\n\t".join([str(k) + ' (' + ', '.join([self.model.gaugeGroupsList[g].name for g in sorted(list(v))]) + ')' for k,v in problematicCouplings.items()])


        if yuk != {}:
            loggingCritical("Gauge invariance is not satisfied by the following Yukawa couplings :" + which(yuk))
        if quartics != {}:
            loggingCritical("Gauge invariance is not satisfied by the following quartic couplings :" + which(quartics))
        if fermionMass != {}:
            loggingCritical("Gauge invariance is not satisfied by the following fermion mass couplings :" + which(fermionMass))
        if trilinears != {}:
            loggingCritical("Gauge invariance is not satisfied by the following trilinear couplings :" + which(trilinears))
        if scalarMass != {}:
            loggingCritical("Gauge invariance is not satisfied by the following scalar mass couplings :" + which(scalarMass))
        if any([el != {} for el in (yuk, quartics, fermionMass, trilinears, scalarMass)]):
            exit()

        loggingInfo("All OK !" + (f" ({time.time()-t0:.3f} seconds)" if self.model.times else ''))


    #########################
    # Some useful functions #
    #########################

    def nonZeroGauge(self, A, B):
        if not A in self.Ugauge or not B in self.Ugauge or not self.kinMix:
            return (A==B)
        return True

    def gaugeIndices(self, *args):
        """ Constructs an iterator to sum on gauge indices, taking correctly into
            account kinetic mixing and optimizing performance
            Usage :
                - gaugeIndices(1) creates 1 index
                - gaugeIndices(2) create 2 indices both appearing in G(A,B)
                - gaugeIndices(1,2,1) creates 4 indices with indices 2 & 3 in G(A,B)
                - gaugeIndices( A ) creates 1 index B appearing in G(A,B) with A fixed
        """

        totalComb = []

        for n in args:
            comb = iter([])

            if type(n) == int:
                # Abelian indices combinations
                if n == 1:
                    comb = iter(self.gi)
                else:
                    comb = itertools.product(*(n*[self.Ugauge]))

                    # Non-abelian indices
                    comb = itertools.chain(comb, iter([tuple(n*[el]) for el in self.NUgauge]))

            if type(n) == tuple:
                if n not in self.Ugauge:
                    comb = iter([n])
                else:
                    comb = iter(self.Ugauge)

            totalComb.append(comb)

        if len(args) > 1:
            totalComb = itertools.product(*totalComb)
        else:
            totalComb = totalComb[0]

        return totalComb


    def dicTilde(self, dic, indPos):
        if not isinstance(dic, TensorDic):
            dicTilde = {}
            for k,v in dic.items():
                newInds = list(k)
                for p in indPos:
                    if k[p] < self.nF//2:
                        newInds[p] = k[p] + self.nF//2
                    else:
                        newInds[p] = k[p] - self.nF//2
                dicTilde[tuple(newInds)] = v

        else:
            dicTilde = TensorDic()
            for k in dic:
                newInds = list(k)
                for p in indPos:
                    if k[p] < self.nF//2:
                        newInds[p] = k[p] + self.nF//2
                    else:
                        newInds[p] = k[p] - self.nF//2
                dicTilde[tuple(newInds)] = k
            dicTilde.tilde = True
            dicTilde.tildeRef = dic

        return dicTilde

