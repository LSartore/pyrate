# -*- coding: utf-8 -*-

from sys import exit
import sys
import os

wd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(wd)

import copy as cp
from functools import reduce, cmp_to_key
import operator

import numpy as np

from sympy import (GramSchmidt, I, IndexedBase, KroneckerDelta,
                   Rational, Wild)

from sympy import (conjugate, exp, expand, factorial, flatten, matrix2numpy,
                   pi, sqrt)


from Math import MathGroup, Sn, sMat, sEye as eye
from Math import sTensor

import time

import re as reg

from sympy import init_printing
init_printing(forecolor='White', wrap_line=False, use_unicode=False)

def timer(func):
    def inner(*args, **kwargs):
        t = time.time()
        ret = func(*args, **kwargs)
        print(func.__name__, f" executed in {time.time()-t:.3f} s.")

        return ret
    return inner

class CartanMatrix(object):
    """
    Represents a Cartan Matrix.
    Constructor : (name, id) or (string)
        e.g. (SO, 10) or ("SU6")
    """
    class AlgebraNameError(BaseException):
        pass

    def __init__(self, *args):
        self._translation = {'SU': 'A', 'SP': 'C', 'SO': ('B', 'D')}
        self._classicalLieAlgebras = ['A', 'B', 'C', 'D']

        if len(args) == 2:
            name, Id = args
            name = name.upper()
        elif len(args) == 1:
            Id = reg.search(r'\d+$', args[0]).group()
            name, Id = args[0].replace(Id, '').upper(), int(Id)

        self._validateAlgebra(name, Id)
        self.cartan = self._constructCartanMatrix()


    def _validateAlgebra(self, name, Id):
        err = CartanMatrix.AlgebraNameError
        self._fullName = name + str(Id)
        self._id = Id

        if name not in self._classicalLieAlgebras:
            if name in self._translation:
                self._name = self._translation[name]

                if name == 'SU':
                    if self._id < 2:
                        raise err("For SU[n], 'n' must be >= 2")
                    self._id -= 1
                elif name == 'SO' and self._id % 2 != 0:
                    if self._id < 3:
                        raise err("For SO[n], odd 'n' must be >= 3")
                    self._id = (self._id - 1) // 2
                    self._name = self._name[0]
                elif name == 'SO' and self._id % 2 == 0:
                    if self._id < 5:
                        raise err("For SO[n], even 'n' must be >= 6")
                    self._id //= 2
                    self._name = self._name[1]
                elif name == 'SP':
                    if self._id % 2 != 0:
                        raise err("For SP[n], 'n' must be even")
                    self._id //= 2

            # Exceptional algebras
            else:
                self._name = name
                if name == 'E':
                    if self._id not in (6, 7, 8):
                        raise err("Error : for E[n], 'n' must equal 6, 7 or 8.")
                elif name == 'F':
                    if self._id != 4:
                        raise err("Error : for F[n], 'n' can only equal 4.")
                elif name == 'G':
                    if self._id != 2:
                        raise err("Error : for G[n], 'n' can only equal 2.")
                else:
                    errorStr = f"Unknown Lie Algebra : {name}{Id}. Valid inputs are:\n"
                    errorStr += '\t- SU[n] ; n >= 2\n'
                    errorStr += '\t- SO[n] ; n >= 3 (odd) or n >= 6 (even)\n'
                    errorStr += '\t- SP[n] ; n >= 2 (even)\n'
                    errorStr += '\t- G2\n'
                    errorStr += '\t- F4\n'
                    errorStr += '\t- E6, E7, E8'

                    raise err(errorStr)
        else:
            if self._id <= 0:
                exit("Id must be a positive integer")
            elif name == 'D' and self._id == 1:
                exit("For 'D' algebra, Id must be >= 1")

            self._name = name



    def _constructCartanMatrix(self):
        fillUp = eval('self._fillUpFunction' + self._name)

        return sMat(self._id, self._id, lambda i, j: fillUp(i, j))

    def _fillUpFunctionA(self, i, j):
        if i == j:
            return 2
        elif i == j + 1 or j == i + 1:
            return -1
        else:
            return 0

    def _fillUpFunctionB(self, i, j):
        if (i,j) == (self._id - 2, self._id - 1):
            return -2
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionC(self, i, j):
        if (i,j) == (self._id - 1, self._id - 2):
            return -2
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionD(self, i, j):
        if (i,j) == (self._id - 1, self._id - 2):
            return 0
        if (i,j) == (self._id - 2, self._id - 1):
            return 0
        if (i,j) == (self._id - 1, self._id - 3):
            return -1
        if (i,j) == (self._id - 3, self._id - 1):
            return -1
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionE(self, i, j):
        if (i,j) == (self._id - 2, self._id - 1):
            return 0
        if (i,j) == (self._id - 1, self._id - 2):
            return 0
        if (i,j) == (2, self._id - 1):
            return -1
        if (i,j) == (self._id - 1, 2):
            return -1
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionF(self, i, j):
        if (i,j) == (1, 2):
            return -2
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionG(self, i, j):
        if (i,j) == (0, 1):
            return -3
        return self._fillUpFunctionA(i, j)


class LieAlgebra(object):
    """
    This is the central class implmenting all the method that one can perform on the lie algebra

    Constructor : (cartanMatrix) or constructor of CartanMatrix
    """

    def __init__(self, *args, realBasis=None):
        if not (len(args) == 1 and isinstance(args[0], CartanMatrix)):
            cartanMatrix = CartanMatrix(*args)
        else:
            cartanMatrix = args[0]

        self.cartan = cartanMatrix
        self.cm = self.cartan.cartan
        self._n = self.cm.shape[0]  # size of the n times n matrix = rank of the algebra
        self.cminv = self.cm.inv()  # inverse of the Cartan matrix
        self.ncm = matrix2numpy(self.cm)  # numpy version of the Cartan Matrix
        self.ncminv = matrix2numpy(self.cminv)  # numpy version of the inverse of the Cartan matrix

        self._matD = self._matrixD()  # D matrix
        self._smatD = self._specialMatrixD()  # Special D matrix

        self._cmID = np.dot(self.ncminv, self._matD)  # matrix product of the inverse Cartan matrix and D matrix
        self._cmIDN = self._cmID / np.max(self._matD)  # same as cmID but normalized to the max of matD

        self.proots = self._positiveRoots()  # compute the positive roots
        self._deltaTimes2 = self.proots.sum(axis=0) # sum the positive roots

        self._rho = self._sumPositiveCoRoots()

        self._dimR = {}

        self.adjoint = self._getAdjoint()
        # self.fond = self._getFond()
        self.dimAdj = self._getDimAdj()
        self.longestWeylWord = self._longestWeylWord()

        # store the matrices for speeding up multiple calls
        self._repMinimalMatrices = {}
        self._repMatrices = {}
        self._dominantWeightsStore = {}
        self._invariantsStore = {}
        self._struc = []

        self.a, self.b, self.c, self.d, self.e = map(IndexedBase, ['a', 'b', 'c', 'd', 'e'])
        self.f, self.g, self.h, self.i = map(IndexedBase, ['f', 'g', 'h', 'i'])
        self._symblist = [self.a, self.b, self.c, self.d, self.e]
        self._symbdummy = [self.f, self.g, self.h, self.i]
        self.p, self.q = map(Wild, ['p', 'q'])
        self.pp = Wild('pp', exclude=[IndexedBase])

        # create an Sn object for all the manipulation on the  Sn  group
        self.Sn = Sn()
        # create a MathGroup object for the auxiliary functions
        self.math = MathGroup()

        # which reps need to be translated to real basis is a property of the algebra
        self._realBasisDic = {}
        self._computeReal = False

    def _matrixD(self):
        """
        Returns a diagonal matrix with the values <root i, root i>
        """
        positions = sum(
            [[(irow, icol) for icol, col in enumerate(row) if (col in [-1, -2, -3]) and (irow < icol)]
             for irow, row in enumerate(self.ncm)], [])
        result = np.ones((1, self._n), dtype=object)[0]

        for coord1, coord2 in positions:
            result[coord2] = Rational(self.cm[coord2, coord1], self.cm[coord1, coord2]) * result[coord1]
        return np.diagflat(result)


    def _specialMatrixD(self):
        result = sMat(self._n, 4)
        for i in range(1, self._n + 1):
            k = 1
            for j in range(1, self._n + 1):
                if self.cm[i - 1, j - 1] == -1:
                    result[i - 1, k - 1] = j
                    k += 1
                if self.cm[i - 1, j - 1] == -2:
                    result[i - 1, k - 1] = j
                    result[i - 1, k - 1 + 1] = j
                    k += 2
                if self.cm[i - 1, j - 1] == -3:
                    result[i - 1, k - 1] = j
                    result[i - 1, k - 1 + 1] = j
                    result[i - 1, k - 1 + 2] = j
                    k += 3
        return matrix2numpy(result)

    def _findM(self, ex, el, ind):
        aux1 = cp.copy(el[ind - 1])
        aux2 = cp.copy(el)
        aux2[ind - 1] = 0
        auxMax = 0
        for ii in range(1, aux1 + 2):
            if ex.count(aux2) == 1:
                auxMax = aux1 - ii + 1
                return auxMax
            aux2[ind - 1] = cp.copy(aux2[ind - 1] + 1)
        return auxMax

    def _simpleProduct(self, v1, v2, cmID):
        # Scalar product from two vector and a matrix
        if type(v1) == list:
            v1 = np.array(v1)
        if type(v2) == list:
            v2 = np.array(v2)
        return Rational(1, 2) * (np.dot(np.dot(v1, cmID), v2.transpose())[0, 0])

    def _positiveRoots(self):
        """
        Returns the positive roots of a given group
        """
        aux1 = [[KroneckerDelta(i, j) for j in range(1, self._n + 1)] for i in range(1, self._n + 1)]
        count = 0
        weights = cp.copy(self.cm)
        while count < weights.rows:
            count += 1
            aux2 = cp.copy(aux1[count - 1])
            for inti in range(1, self._n + 1):
                aux3 = cp.copy(aux2)
                aux3[inti - 1] += 1
                if self._findM(aux1, aux2, inti) - weights[count - 1, inti - 1] > 0 and aux1.count(aux3) == 0:
                    weights = weights.col_join(weights.row(count - 1) + self.cm.row(inti - 1))
                    aux1.append(aux3)
        return matrix2numpy(weights)

    def _positiveCoRoots(self):
        r = self._positiveRoots()
        cr = [2*el / self._simpleProduct([el], [el], self._cmID) for el in r]
        return cr

    def _sumPositiveCoRoots(self):
        return sum(self._positiveCoRoots())/2

    def _getAdjoint(self):
        # returns the adjoint of the gauge group
        return self._tolist(self.proots[-1])

    def _getDimAdj(self):
        return self.dimR(self.adjoint)

    def dynkinIndex(self, rep):
        """
        returns the dynkin index of the corresponding representation
        """
        return self.casimir(rep) * Rational(self.dimR(rep), self.dimR(self.adjoint))

    def structureConstants(self, factor=1):
        # About 100 times faster than the old function for SU6 and SO10

        if self._struc != []:
            return self._struc

        d = self.dimAdj
        fondRep = self.repsUpToDimN(self.dimAdj)[1]
        mat_fond = self.repMatrices(fondRep)
        srep = self.dynkinIndex(fondRep)

        def trAcommBC(A,B,C):
        # Compute Tr(A*[B,C]) in an efficient manner, using matrix sparsity
        # = A[i,j]B[j,k]C[k,i] - A[i,j]C[j,k]B[k,i]

            res = 0
            Brows, Bcols = {}, {}
            Crows, Ccols = {}, {}

            for k,v in B.todok().items():
                if k[0] not in Brows:
                    Brows[k[0]] = {}
                if k[1] not in Bcols:
                    Bcols[k[1]] = {}
                Brows[k[0]][k[1]] = v
                Bcols[k[1]][k[0]] = v

            for k,v in C.todok().items():
                if k[0] not in Crows:
                    Crows[k[0]] = {}
                if k[1] not in Ccols:
                    Ccols[k[1]] = {}
                Crows[k[0]][k[1]] = v
                Ccols[k[1]][k[0]] = v

            for (i,j), va in A.todok().items():
                if j in Brows and i in Ccols:
                    for k in set(Brows[j]).intersection(Ccols[i]):
                        res += va*Brows[j][k]*Ccols[i][k]
                if j in Crows and i in Bcols:
                    for k in set(Crows[j]).intersection(Bcols[i]):
                        res -= va*Crows[j][k]*Bcols[i][k]

            return res

        struc = []

        for i in range(d):
            f_i = sMat(d,d,{})
            for j in range(d):
                for k in range(d):
                    if j==k:
                        continue
                    if j>k:
                        f_i[j,k] = -f_i[k,j]
                        continue
                    f_i[j,k] = -I/srep * trAcommBC(mat_fond[i], mat_fond[j], mat_fond[k]) * factor
            struc.append(f_i)

        self._struc = struc
        return struc

    def frobeniusSchurIndicator(self, rep):
        if self._nptokey(self.conjugateIrrep(rep)) != tuple(rep):
            return 1

        highestWeight = self._dominantWeights(rep)[0][0].ravel()
        p = self._simpleProduct([self._rho], [highestWeight], self._cmID)
        indic = exp(2*I*pi*p)

        if indic == 1:
            return 0
        return indic

    def _longestWeylWord(self):
        # returns the longest Weyl word: from the Lie manual see Susyno
        weight = [-1] * self._n
        result = []
        while list(map(abs, weight)) != weight:
            for iel, el in enumerate(weight):
                if el < 0:
                    break
            weight = self._reflectWeight(weight, iel + 1)
            result.insert(0, iel + 1)
        return result

    def _reflectWeight(self, weight, i):
        """
        Reflects a given weight. WARNING The index i is from 1 to n
        """
        result = cp.deepcopy(weight)
        result[i - 1] = -weight[i - 1]
        for ii in range(1, 5):
            if self._smatD[i - 1, ii - 1] != 0:
                result[self._smatD[i - 1, ii - 1] - 1] += weight[i - 1]
        return result

    def _weylOrbit(self, weight):
        """
        Creates the weyl orbit i.e. the system of simple root
        """
        counter = 0
        result, wL = [], []
        wL.append([weight])
        result.append(weight)
        while len(wL[counter]) != 0:
            counter += 1
            wL.append([])
            for j in range(1, len(wL[counter - 1]) + 1):
                for i in range(1, self._n + 1):
                    if wL[counter - 1][j - 1][i - 1] > 0:
                        aux = self._reflectWeight(wL[counter - 1][j - 1], i)[i + 1 - 1:self._n + 1]
                        if aux == list(map(abs, aux)):
                            wL[counter].append(self._reflectWeight(wL[counter - 1][j - 1], i))
            result = result + wL[counter]  # Join the list
        return result

    def _dominantConjugate(self, weight):
        weight = weight[0]
        if self.cm == np.array([[2]]):  # SU2 code
            if weight[0] < 0:
                return [-weight, 1]
            else:
                return [weight, 0]
        else:
            index = 0
            dWeight = weight
            i = 1
            while i <= self._n:
                if (dWeight[i - 1] < 0):
                    index += 1
                    dWeight = self._reflectWeight(dWeight, i)
                    i = min([self._smatD[i - 1, 0], i + 1])
                else:
                    i += 1
            return [dWeight, index]

    def _representationIndex(self, irrep):
        delta = np.ones((1, self._n), dtype=int)
        # Factor of 2 ensures is due to the fact that SimpleProduct is defined such that Max[<\[Alpha],\[Alpha]>]=1 (considering all positive roots), but we would want it to be =2
        return Rational(self.dimR(irrep), self.dimR(self.adjoint)) * 2 * self._simpleProduct(irrep, irrep + 2 * delta,
                                                                                             self._cmID)

    def casimir(self, irrep):
        """
        Returns the casimir of a given irrep
        """
        irrep = np.array([irrep])
        return self._simpleProduct(irrep, irrep + self._deltaTimes2, self._cmIDN)

    def dimR(self, irrep):
        """
        Returns the dimention of representation irrep
        """
        if type(irrep) == np.ndarray and len(irrep.shape) == 1:
            irrep = np.array([irrep])
        if type(irrep) == np.ndarray:
            keydimR = tuple([int(el) for el in irrep.tolist()[0]])
        else:
            keydimR = tuple(irrep)
        if keydimR in self._dimR:
            return self._dimR[keydimR]
        if not (type(irrep) == np.ndarray):
            irrep = np.array([irrep])
        delta = Rational(1, 2) * self._deltaTimes2

        if self.cartan._name in 'ABC' and self.cartan._id == 1 and type(delta) != np.ndarray:
            delta = np.array([delta])

        result = np.prod([self._simpleProduct([self.proots[i - 1]], irrep + delta, self._cmID) /
                          self._simpleProduct([self.proots[i - 1]], [delta], self._cmID)
                             for i in range(1, len(self.proots) + 1)], axis=0)

        result = round(result)
        self._dimR[keydimR] = result
        return result

    def _conjugacyClass(self, irrep):
        if not (type(irrep) == np.ndarray):
            irrep = np.array(irrep)

        series, n = self.cartan._name, self.cartan._id
        if series == "A":
            return [np.sum([i * irrep[i - 1] for i in range(1, n + 1)]) % (n + 1)]
        if series == "B":
            return [irrep[n - 1] % 2]
        if series == "C":
            return [np.sum([irrep[i - 1] for i in range(1, n + 1, 2)]) % 2]
        if series == "D" and n % 2 == 1:
            return [(irrep[-2] + irrep[-1]) % 2,
                    (2 * np.sum([irrep[i - 1] for i in range(1, n - 1, 2)])
                     + (n - 2) * irrep[-2] + n * irrep[-1]) % 4]
        if series == "D" and n % 2 == 0:
            return [(irrep[-2] + irrep[-1]) % 2,
                    (2 * np.sum([irrep[i - 1] for i in range(1, n - 2, 2)])
                     + (n - 2) * irrep[-2] + n * irrep[-1]) % 4]
        if series == "E" and n == 6:
            return [(irrep[0] - irrep[1] + irrep[3] - irrep[4]) % 3]
        if series == "E" and n == 7:
            return [(irrep[3] + irrep[5] + irrep[6]) % 2]
        if series == "E" and n == 8:
            return [0]
        if series == "F":
            return [0]
        if series == "G":
            return [0]


    def _dominantWeights(self, weight):
        """
        Generate the dominant weights without dimentionality information
        """
        keyStore = tuple(weight)
        if keyStore in self._dominantWeightsStore:
            return self._dominantWeightsStore[keyStore]
        # convert the weight
        weight = np.array([weight], dtype=int)
        listw = [weight]
        counter = 1
        while counter <= len(listw):
            aux = [listw[counter - 1] - self.proots[i] for i in range(len(self.proots))]
            aux = [el for el in aux if np.all(el == abs(el))]
            listw = listw + aux
            tp = []
            listw = [self._nptokey(el) for el in listw]
            for el in listw:
                if not (el) in tp:
                    tp.append(el)
            listw = [np.array([el], dtype=int) for el in tp]
            counter += 1

        # need to sort listw
        def sortList(a, b):
            tp1 = list(np.dot(-(a - b), self.ncminv)[0])
            return self._cmp(tp1, [0] * a.shape[1])

        listw.sort(key=cmp_to_key(sortList))

        functionaux = {self._nptokey(listw[0]): 1}
        result = [[listw[0], 1]]
        for j in range(2, len(listw) + 1):
            for i in range(1, len(self.proots) + 1):
                k = 1
                aux1 = self._indic(functionaux,
                                   tuple(self._dominantConjugate(k * self.proots[i - 1] + listw[j - 1])[0]))
                key = self._nptokey(listw[j - 1])
                while aux1 != 0:
                    aux2 = k * self.proots[i - 1] + listw[j - 1]
                    if key in functionaux:
                        functionaux[key] += 2 * aux1 * self._simpleProduct(aux2, [self.proots[i - 1]], self._cmID)
                    else:
                        functionaux[key] = 2 * aux1 * self._simpleProduct(aux2, [self.proots[i - 1]], self._cmID)
                    k += 1
                    # update aux1 value
                    kkey = tuple(self._dominantConjugate(k * self.proots[i - 1] + listw[j - 1])[0])
                    if kkey in functionaux:
                        aux1 = functionaux[kkey]
                    else:
                        aux1 = 0
            functionaux[key] /= self._simpleProduct(listw[0] + listw[j - 1] + self._deltaTimes2,
                                                    listw[0] - listw[j - 1], self._cmID)
            result.append([listw[j - 1], self._indic(functionaux, self._nptokey(listw[j - 1]))])
        self._dominantWeightsStore[keyStore] = result
        return result

    def _weights(self, weights):
        """
        Reorder the weights of conjugate representations
        so that RepMatrices[group,ConjugateIrrep[group,w]]=-Conjugate[RepMatrices[group,w]]
        and Invariants[group,{w,ConjugateIrrep[group,w]},{False,False}]=a[1]b[1]+...+a[n]b[n]
        """
        if (self._cmp(list(weights), list(self.conjugateIrrep(weights))) in [-1, 0]) and not (np.all(
                    (self.conjugateIrrep(weights)) == weights)):
            return [[-1*el[0], el[1]] for el in self._weights(self.conjugateIrrep(weights))]
        else:
            dw = self._dominantWeights(weights)
            result = sum(
                [[[np.array(el, dtype=int), dw[ii][1]] for el in self._weylOrbit(self._tolist(dw[ii][0][0]))] for ii in
                 range(len(dw))], [])

            def sortList(a, b):
                tp1 = list(np.dot(-(a[0] - b[0]), self.ncminv).ravel())
                return self._cmp(tp1, [0] * a[0].shape[0])

            result.sort(key=cmp_to_key(sortList))
            return result

    ############################
    # Generate representations #
    ############################

    def _repsUpToDimNAuxMethod(self, weight, digit, max, reap):
        waux = cp.deepcopy(weight)
        waux[digit] = 0
        while self.dimR(np.array([waux])) <= max:
            if digit == len(weight) - 1:
                reap.append([int(el) for el in waux.ravel()])
            else:
                self._repsUpToDimNAuxMethod(waux, digit + 1, max, reap)
            waux[digit] += 1

    def repsUpToDimN(self, maxdim):
        """ Returns the list of irreps of dim less or equal to maxdim """
        reap = []
        self._repsUpToDimNAuxMethod(np.zeros((1, self._n))[0], 0, maxdim, reap)

        def sortByDimension(x):
            dim = self.dimR(x)
            rep = self._representationIndex(np.array([x]))
            conj = self._conjugacyClass(x)
            return tuple(flatten([dim, rep, conj]))

        reap.sort(key = sortByDimension)

        return reap

    def conjugateIrrep(self, irrep, u1in=False):
        """
        returns the conjugated irrep
        """

        if u1in:
            irrep, u1 = irrep
        lbd = lambda weight, ind: self._reflectWeight(weight, ind)
        res = -reduce(lbd, [np.array([irrep])[0]] + self.longestWeylWord)
        if u1in:
            u1 = - u1
            return [res, u1]
        else:
            return res


    ###########################
    # Representation matrices #
    ###########################

    def repMinimalMatrices(self, maxW):
        """
        1) The output of this function is a list of sets of 3 matrices:
            {{E1, F1, H1},{E2, F2, H2},...,{En, Fn, Hn}}, where n is the group's rank.
            Hi are diagonal matrices, while Ei and Fi are raising and lowering operators.
            These matrices obey the Chevalley-Serre relations: [Ei, Ej]=delta_ij Hi, [Hi,Ej]= AjiEj, [Hi,Fj]= -AjiFj and [Hi,Hj]=0
            here A is the Cartan matrix of the group/algebra.
        2) With the exception of SU(2) [n=1], these 3n matrices Ei, Fi, Hi do not generate the Lie algebra,
            which is bigger, as some raising and lowering operators are missing.
            However, these remaining operators can be obtained through simple commutations: [Ei,Ej], [Ei,[Ej,Ek]],...,[Fi,Fj], [Fi,[Fj,Fk]].
        3) This method clearly must assume a particular basis for each representation so the results are basis dependent.
        4) Also, unlike RepMatrices, the matrices given by this function are not Hermitian and therefore they do not conform with the usual requirements of model building in particle physics.
            However, for some applications, they might be all that is needed.
        """

        # check whether it s not been calculated already
        if type(maxW) == np.ndarray:
            tag = self._nptokey(maxW)
        else:
            tag = tuple(maxW)
            maxW = np.array([maxW])
        if tag in self._repMinimalMatrices:
            return self._repMinimalMatrices[tag]

        # auxiliary function for the repMatrices method base on the Chevalley-Serre relations
        cmaxW = self.conjugateIrrep(self._tolist(maxW))

        if self._cmp(self._tolist(maxW), self._tolist(cmaxW)) in [-1, 0] and not (np.all(cmaxW == maxW)):
            return [[-1 * el[1], -1 * el[0], -1 * el[2]] for el in
                    self.repMinimalMatrices(cmaxW)]
        else:
            listw = self._weights(self._tolist(maxW))
            up, dim, down = {}, {}, {}
            for i in range(len(listw)):
                dim[self._nptokey(listw[i][0])] = listw[i][1]
            up[self._nptokey(listw[0][0])] = sMat(1, self._n)

            for element in range(1, len(listw)):
                matrixT = [[]]
                for j in range(self._n):
                    col = [[]]
                    for i in range(self._n):
                        key1 = self._nptokey(listw[element][0] + self.ncm[i])
                        key2 = self._nptokey(listw[element][0] + self.ncm[i] + self.ncm[j])
                        key3 = self._nptokey(listw[element][0] + self.ncm[j])
                        dim1 = self._indic(dim, key1)
                        dim2 = self._indic(dim, key2)
                        dim3 = self._indic(dim, key3)
                        ax = 1 if col == [[]] else 0
                        if dim1 != 0 and dim3 != 0:
                            if dim2 != 0:
                                aux1 = up[self._nptokey(listw[element][0] + self.ncm[i])][j]
                                aux2 = down[self._nptokey(listw[element][0] + self.ncm[i] + self.ncm[j])][i]
                                if i != j:
                                    if col == [[]]:
                                        col = aux1*aux2
                                    else:
                                        col = col.append(aux1*aux2, axis=ax)

                                else:
                                    tmp = aux1*aux2 + eye(dim1, listw[element][0][i] + self.ncm[i, i])
                                    if col == [[]]:
                                        col = tmp
                                    else:
                                        col = col.append(tmp, axis=ax)

                            else:
                                if i != j:
                                    if col == [[]]:
                                        col = sMat(dim1, dim3)
                                    else:
                                        col = col.append(sMat(dim1, dim3), axis=ax)
                                else:
                                    tmp = eye(dim1, listw[element][0][i] + self.ncm[i, i])
                                    if col == [[]]:
                                        col = tmp
                                    else:
                                        col = col.append(tmp, axis=ax)
                    if col != [[]]:
                        if matrixT == [[]]:
                            matrixT = col.transpose()
                        else:
                            matrixT = matrixT.append(col.transpose(), axis=0)


                if matrixT == [[]]:
                    matrix = sMat(1, 1)
                else:
                    matrix = matrixT.transpose()

                aux1 = sum([self._indic(dim, self._nptokey(listw[element][0] + self.ncm[i])) for i in range(self._n)])
                aux2 = self._indic(dim, self._nptokey(listw[element][0]))
                cho = self.math._decompositionTypeCholesky(matrix)
                if cho.shape == (0,):
                    aux3 = sMat(1,1)
                else:
                    aux3 = cho.copy()
                    if not ( aux1 - cho.shape[0] == 0 and aux2 - cho.shape[1] == 0 ):
                        exit()
                        aux3 = aux3.pad(pad_width=((0, max(aux1 - cho.shape[0], 0)), (0, max(aux2 - cho.shape[1], 0))))

                aux4 = aux3.transpose()
                if aux3*aux4 != matrix:
                    print("Error in repminimal matrices:", aux3, " ", aux4, " ", matrix)
                    return
                # Obtain the blocks in  (w+\[Alpha]i)i and wj. Use it to feed the recursive algorith so that we can calculate the next w's
                aux1 = np.array([[0, 0]])  # format (+-): (valid cm raise index i - 1, start position of weight w+cm[[i-1]]-1)
                for i in range(self._n):
                    key = self._nptokey(listw[element][0] + self.ncm[i])
                    if key in dim:
                        aux1 = np.append(aux1, np.array([[i + 1, aux1[-1, 1] + dim[key]]], dtype=object), axis=0)
                for i in range(len(aux1) - 1):
                    index = aux1[i + 1, 0]
                    posbegin = aux1[i, 1] + 1
                    posend = aux1[i + 1, 1]
                    key = self._nptokey(listw[element][0] + self.ncm[index - 1])
                    aux2 = down[key] if key in down else [[]] * self._n
                    aux2[index - 1] = aux3[posbegin - 1:posend]
                    down[key] = aux2
                    key2 = self._nptokey(listw[element][0])
                    aux2 = up[key2] if key2 in up else [[]] * self._n
                    aux2[index - 1] = (aux3[posbegin - 1:posend]).transpose()
                    up[key2] = aux2

            # Put the collected pieces together and build the 3n matrices: hi,ei,fi
            begin, end = {self._nptokey(listw[0][0]): 1}, {self._nptokey(listw[0][0]): listw[0][1]}
            for element in range(1, len(listw)):
                key = self._nptokey(listw[element][0])
                key1 = self._nptokey(listw[element - 1][0])
                begin[key] = begin[key1] + listw[element - 1][1]
                end[key] = end[key1] + listw[element][1]
            aux2 = sum([listw[i][1] for i in range(len(listw))])
            matrixE, matrixF, matrixH = [], [], []
            for i in range(self._n):
                aux6, aux7, aux8 = ( sMat(aux2, aux2),   # e[i]
                                     sMat(aux2, aux2),   # f[i]
                                     sMat(aux2, aux2) )  # h[i]

                for element in range(len(listw)):
                    key = self._nptokey(listw[element][0] + self.ncm[i])
                    key2 = self._nptokey(listw[element][0])
                    key3 = self._nptokey(listw[element][0] - self.ncm[i])
                    if key in dim:
                        b1, e1 = begin[key], end[key]
                        b2, e2 = begin[key2], end[key2]
                        aux6[b1 - 1:e1, b2 - 1:e2] = (up[key2][i]).transpose()
                    if key3 in dim:
                        b1, e1 = begin[key3], end[key3]
                        b2, e2 = begin[key2], end[key2]
                        aux7[b1 - 1:e1, b2 - 1:e2] = (down[key2][i]).transpose()
                    b1, e1 = begin[key2], end[key2]
                    aux8[b1 - 1:e1, b1 - 1:e1] = listw[element][0][i] * eye(listw[element][1])
                matrixE.append(aux6)
                matrixF.append(aux7)
                matrixH.append(aux8)
            aux1 = [[matrixE[i], matrixF[i], matrixH[i]] for i in range(self._n)]

            self._repMinimalMatrices[tag] = aux1
            return aux1

    def repMatrices(self, maxW, rep=None, conj=False, realBasis=None):
        """
        This method returns the complete set of matrices that make up a representation, with the correct casimir and trace normalizations
        1) The matrices {M_i} given by this method are in conformity with the usual requirements in particle physics: \!\(
            M_a^\Dagger = M_a ; Tr(M_a M_b = S(rep) \Delta_ab; Sum_a M_a M_a = C(rep) 1.
        """

        # check if its been calculated already
        if isinstance(maxW, np.ndarray):
            tag = self._nptokey(maxW)
        else:
            tag = tuple(maxW)

        if not tag in self._repMatrices:
            # Let's gather the minimal rep matrices
            if not isinstance(maxW, np.ndarray):
                maxW = np.array([maxW])
            if rep == None:
                rep = self.repMinimalMatrices(maxW)
            else:
                rep = [rep]

            dimG = 2 * len(self.proots) + len(self.ncm)
            dimR = self.dimR(maxW.tolist()[0])
            sR = Rational(self.casimir(self._tolist(maxW)) * dimR, dimG)
            if dimR == 1:
                #  Trivial representation, the matrices are null
                listTotal = [rep[0][0] for i in range(dimG)]
                return listTotal
            listE, listF, listH = [el[0] for el in rep], [el[1] for el in rep], [el[2] for el in rep]
            # If it's not the trivial rep, generate the matrices of the remaining algebra elements.
            #  The positive roots of the algebra serve as a guide in this process of doing comutators
            for i in range(self._n, len(self.proots)):
                j = 0
                aux = []
                while aux == []:
                    aux = [iel for iel, el in enumerate(self.proots[:i]) if np.all(el == self.proots[i] - self.proots[j])]
                    if aux == []:
                        j += 1
                listE.append(listE[aux[0]]*listE[j] - listE[j]*listE[aux[0]])
                listF.append(listF[aux[0]]*listF[j] - listF[j]*listF[aux[0]])
            for i, aux in enumerate(listE):
                # Change from the operators T+, T- to Tx,Ty
                listE[i] = aux + listF[i]
                listF[i] = aux - listF[i]

                # Control the normalization of the Tx,Ty matrices with the trace condition
                listE[i] = listE[i] * ( sqrt(sR) / sqrt( (listE[i]*listE[i]).trace() ) )
                listF[i] = listF[i] * ( sqrt(sR) / sqrt( (listF[i]*listF[i]).trace() ) )

            matrixCholesky = np.dot(self.ncminv, self._matD)  # See the casimir expression in a book on lie algebras
            aux = (sMat(matrixCholesky).cholesky()).transpose()  # get the actual cholesky decomposition from sympy
            listH = [reduce(operator.add, [listH[j] * aux[i, j] for j in range(self._n)]) for i in range(self._n)]
            # Up to multiplicative factors, Tz are now correct. We fix again the normalization with the trace condition
            listH = [listH[i] * (sqrt(sR) / sqrt( (listH[i]*listH[i]).trace() )) for i in range(self._n)]
            listTotal = sum([listE, listF, listH], [])

            self._repMatrices[tag] = listTotal

        # Now that either the repMats are in the dic for sure, handle the keywords args
        ret = [el for el in self._repMatrices[tag]]

        if self._goToRealBasis(tag, realBasis):
            # Rotate to real basis if needed
            if conj:
                conj == False

            realMat = self._realBasisRotation(tag)
            realMatAdj = realMat.adjoint()

            for i, el in enumerate(ret):
                ret[i] = realMatAdj*el*realMat

        if conj:
            # Get the rep matrices of the conjugate irrep
            if self.frobeniusSchurIndicator(tag) != 1:
                for i, el in enumerate(ret):
                    ret[i] = -el.transpose()
            else:
                print("Warning in repMatrices: conjugation is only meant for real and pseudo-real representations. Skipping.")

        return ret



    ################
    # Rep products #
    ################

    def reduceRepProduct(self, repslist):
        """
        Reduces a direct product of representation to its irreducible parts
        """
        if len(repslist) == 1:
            return [[repslist, 1]]

        # order the list by dimension
        orderedlist = sorted(repslist, key=lambda x: self.dimR(x))
        n = len(orderedlist)

        result = self._reduceRepProductBase2(orderedlist[n - 2], orderedlist[n - 1])
        for i in range(2, n):
            result = self._reduceRepProductBase1(orderedlist[n - i - 1], result)
        return result

    def _reduceRepProductBase1(self, rep1, listReps):
        res = sum([[(ell[0], el[1] * ell[1]) for ell in self._reduceRepProductBase2(rep1, el[0])] for el in listReps],
                  [])

        final = []
        togather = cp.deepcopy(res)
        while togather != []:
            gathering = togather.pop(0)
            temp = [gathering[1]]
            for iel, el in enumerate(togather):
                if el[0] == gathering[0]:
                    temp.append(el[1])
            togather = [el for el in togather if el[0] != gathering[0]]
            final.append((gathering[0], sum(temp)))
        return final

    def _reduceRepProductBase2(self, w1, w2):
        l1 = self._dominantWeights(w1)

        delta = np.ones(self._n, dtype=int)
        dim = {}
        allIrrep = []
        for i in range(len(l1)):
            wOrbit = np.array(self._weylOrbit(self._tolist(l1[i][0])))
            for j in range(len(wOrbit)):
                aux = self._dominantConjugate([wOrbit[j] + np.array(w2) + delta])
                if np.all(aux[0] - 1 == abs(aux[0] - 1)):
                    key = self._nptokey(aux[0] - delta)
                    if key in dim:
                        dim[key] += (-1) ** aux[1] * l1[i][1]
                    else:
                        dim[key] = (-1) ** aux[1] * l1[i][1]
                    val = self._tolist(aux[0] - delta)
                    if not (val in allIrrep):
                        allIrrep.append(val)
        result = [(el, self._indic(dim, tuple(el))) for el in allIrrep]
        result = [el for el in result if el[1] != 0]
        return result



    ###############################
    #  Computation of invariants  #
    ###############################

    def invariants(self, reps, conj=[], skipSymmetrize=False, pyrateNormalization=False, realBasis=None):
        """
        Calculates the linear combinations of the components of rep1 x rep2 x ... which are invariant under the action of the group.
        These are also known as the Clebsch-Gordon coefficients.
        The invariants/Clebsch-Gordon coefficients returned by this function follow the following general normalization convention.
        Writing each invariant as Sum_i,j,...c^ij... rep1[i] x rep2[j] x ..., then the normalization convention is  Sum_i,j,...|c_ij...|^2=Sqrt[dim(rep1)dim(rep2)...]. Here, i,j, ... are the components of each representation.
        conj represents wether or not the irrep should be conjugated.
        """

        if conj == []:
            conj = [False]*len(reps)

        if len(reps) > len(conj):
            conj = list(conj) + [False]*(len(reps)-len(list(conj)))
            print("Warning : length of conjugations is lower than length of reps.")
            print("\tAssuming conj =", conj)

        elif len(reps) < len(conj):
            conj = conj[:len(reps)]
            print("Warning : length of conjugations is larger than length of reps.")
            print("\tAssuming conj =", conj)

        # If some real reps are to be rotated later, disable the 'conj' keyword which is useless
        if realBasis is not None:
            conj = [el if not self._goToRealBasis(reps[i], realBasis) else False for i,el in enumerate(conj)]

        originalReps = tuple([tuple(el) for el in reps])
        originalCjs = tuple(conj)

        # Sort the input according to (rep, conj)
        ordering = sorted(range(len(originalReps)), key=lambda x: (self.dimR(reps[x]), tuple(reps[x]), conj[x]))

        reps = [originalReps[i] for i in ordering]
        conj = [originalCjs[i] for i in ordering]

        # Final permutations (= inverse of 'ordering' permutation)
        perm = [ordering.index(i) for i in range(len(reps))]

        storeKey = tuple([(r,c) for r,c in zip(reps, conj)])
        if storeKey in self._invariantsStore:
            # The invariant was computed earlier
            invs, maxinds = self._invariantsStore[storeKey]

            if invs == []:
                return []
        else:
            if len(reps) == 2:
                cjs = (conj[0] != conj[1])
                invs, maxinds = self._invariants2Irrep(reps, cjs)
            elif len(reps) == 3:
                cjs = False
                permThree = [0,1,2]

                if (conj[0] and conj[1] and not (conj[2])) or (not (conj[0]) and not (conj[1]) and conj[2]):
                    cjs = True
                if (conj[0] and not (conj[1]) and conj[2]) or (not (conj[0]) and conj[1] and not (conj[2])):
                    cjs = True
                    permThree = [0,2,1]
                if (not (conj[0]) and conj[1] and conj[2]) or (conj[0] and not (conj[1]) and not (conj[2])):
                    cjs = True
                    permThree = [2,1,0]

                invs, maxinds = self._invariants3Irrep([reps[permThree[i]] for i, rep in enumerate(reps)], cjs)
                invs = [el.permute(permThree) for el in invs]
                maxinds = [maxinds[permThree[i]] for i,m in enumerate(maxinds)]
            elif len(reps) == 4:
                invs, maxinds = self._invariants4Irrep([], reps, conj)
            else:
                raise TypeError("Error, only 2, 3 or 4 irrep should be passed.")

            if invs == []:
                self._invariantsStore[storeKey] = (invs, maxinds)
                return []

            # Compute the normalization factor related the dimensions of the irreps
            repDims = 1
            for rep, cj in zip(reps, conj):
                if cj:
                    rep = self.conjugateIrrep(rep)
                repDims *= self.dimR(rep)
            repDims = sqrt(repDims)

            # Normalize the invariants
            invs = self._normalizeInvariants(invs, repDims)

            # Store the invariants
            self._invariantsStore[storeKey] = (invs, maxinds)

        if not skipSymmetrize:
            invs = self._symmetrizeInvariants(reps, invs, maxinds, conj)

        if realBasis is not None:
            invs = self._rotateInvariants(reps, realBasis, invs, conj, maxinds)

        if pyrateNormalization:
            invs = self._pyrateNormalization(invs)

        # Reorder the fields before returning the result
        ret = [el.permute(perm) for el in invs]
        return ret

    def _invariants2Irrep(self, reps, cjs):
        """
        return the invariants of the the irreps
        """
        w1, w2 = self._weights(reps[0]), self._weights(reps[1])
        reps = [np.array([el]) for el in reps if type(el) != np.array]
        r1, r2 = [cp.deepcopy(self.repMinimalMatrices(rep)) for rep in reps]

        if cjs:
            for i in range(len(w2)):
                w2[i][0] = - w2[i][0]
            for i in range(self._n):
                for j in range(3):
                    r2[i][j] = - r2[i][j].transpose()

        array1, array2 = {}, {}
        for i in range(len(w1)):
            array1[self._nptokey(w1[i][0])] = w1[i][1]
        for i in range(len(w2)):
            array2[self._nptokey(w2[i][0])] = w2[i][1]

        aux1 = []
        for i in range(len(w1)):
            if self._indic(array2, self._nptokey(-w1[i][0])) != 0:
                aux1.append([w1[i][0], -w1[i][0]])
        dim1 = [0]
        for i in range(len(aux1)):
            dim1.append(dim1[i] + self._indic(array1, self._nptokey(aux1[i][0]))*
                                  self._indic(array2, self._nptokey(aux1[i][1])))
        b1, e1 = {}, {}
        for i in range(len(aux1)):
            key = tuple([self._nptokey(el) for el in aux1[i]])
            b1[key] = dim1[i] + 1
            e1[key] = dim1[i + 1]

        bigMatrix = []
        for i in range(self._n):
            aux2 = {}
            for j in range(len(aux1)):
                if self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])) != 0:
                    val = [aux1[j][0] + self.ncm[i], aux1[j][1]]
                    key = tuple([self._nptokey(el) for el in val])
                    if key not in aux2:
                        aux2[key] = val
                        # keysaux2.append(key)
                if self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])) != 0:
                    val = [aux1[j][0], aux1[j][1] + self.ncm[i]]
                    key = tuple([self._nptokey(el) for el in val])
                    if key not in aux2:
                        aux2[key] = val

            if len(w1) == 1 and len(w2) == 1:  # Special care is needed if both reps are singlets
                aux2 = aux1
            else:
                aux2 = aux2.values()

            dim2 = [0]
            for k, val in enumerate(aux2):
                dim2.append(dim2[k] + self._indic(array1, self._nptokey(val[0]))*
                                      self._indic(array2, self._nptokey(val[1])))

            b2, e2 = {}, {}
            for k, val in enumerate(aux2):
                key = tuple([self._nptokey(el) for el in val])
                b2[key] = dim2[k] + 1
                e2[key] = dim2[k + 1]
            if dim2[len(aux2)] != 0 and dim1[len(aux1)] != 0:
                matrixE = sMat(dim2[len(aux2)], dim1[len(aux1)])
            else:
                matrixE = []
            for j in range(len(aux1)):
                i1, i2 = (self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])),
                          self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])))

                if i1 != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0] + self.ncm[i], aux1[j][1]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])

                    slice1 = slice(self._indic(b2, kaux4) - 1, self._indic(e2, kaux4))
                    slice2 = slice(self._indic(b1, kaux3) - 1, self._indic(e1, kaux3))

                    m1 = self._blockW(aux1[j][0] + self.ncm[i], aux1[j][0], w1, r1[i][0])
                    m2 = eye(self._indic(array2, self._nptokey(aux1[j][1])))

                    matrixE[slice1, slice2] = m1.kroneckerProduct(m2)

                if i2 != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0], aux1[j][1] + self.ncm[i]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])

                    slice1 = slice(self._indic(b2, kaux4) - 1, self._indic(e2, kaux4))
                    slice2 = slice(self._indic(b1, kaux3) - 1, self._indic(e1, kaux3))

                    m1 = eye(self._indic(array1, self._nptokey(aux1[j][0])))
                    m2 = self._blockW(aux1[j][1] + self.ncm[i], aux1[j][1], w2, r2[i][0])

                    matrixE[slice1, slice2] = m1.kroneckerProduct(m2)

            if bigMatrix == [] and matrixE != []:
                bigMatrix = matrixE

            elif bigMatrix != [] and matrixE != []:
                bigMatrix = bigMatrix.append(matrixE, axis=0)

        if len(bigMatrix) == 0:
            return [], [0, 0, 0]

        dim1 = [0]
        dim2 = [0]
        for i in range(len(w1)):
            dim1.append(dim1[i] + w1[i][1])
        for i in range(len(w2)):
            dim2.append(dim2[i] + w2[i][1])
        for i in range(len(w1)):
            b1[self._nptokey(w1[i][0])] = dim1[i]
        for i in range(len(w2)):
            b2[self._nptokey(w2[i][0])] = dim2[i]

        aux4 = bigMatrix.nullSpace()

        # let's construct the invariant combination from the null space solution
        # declare the symbols for the output of the invariants
        expr = []
        resTensor = []
        maxInds = [0, 0]

        for i0, nsDic in enumerate(aux4.values()):
            expr.append(0)
            resTensor.append(sTensor(*[self.dimR(r) for r in reps]))
            count = 0
            for i in range(len(aux1)):
                r1, r2 = (self._indic(array1, self._nptokey(aux1[i][0])),
                          self._indic(array2, self._nptokey(aux1[i][1])))

                bi1, bi2 = (b1[self._nptokey(aux1[i][0])],
                            b2[self._nptokey(aux1[i][1])])

                for j1 in range(r1):
                    aInd = bi1 + j1
                    if aInd > maxInds[0]:
                        maxInds[0] = aInd

                    for j2 in range(r2):
                        bInd = bi2 + j2
                        if bInd > maxInds[1]:
                            maxInds[1] = bInd

                        if count in nsDic:
                            resTensor[i0][aInd, bInd] = nsDic[count]

                        count += 1

        return resTensor, maxInds


    def _invariants3Irrep(self, reps, cjs):
        """
        Returns the invariant for three irreps
        """
        w1, w2, w3 = self._weights(reps[0]), self._weights(reps[1]), self._weights(reps[2])
        reps = [np.array([el]) for el in reps if type(el) != np.array]
        r1, r2, r3 = [cp.deepcopy(self.repMinimalMatrices(rep)) for rep in reps]

        if cjs:
            for i in range(len(w3)):
                w3[i][0] = - w3[i][0]
            for i in range(self._n):
                for j in range(3):
                    r3[i][j] = - r3[i][j].transpose()

        array1, array2, array3 = {}, {}, {}
        for i in range(len(w1)):
            array1[self._nptokey(w1[i][0])] = w1[i][1]
        for i in range(len(w2)):
            array2[self._nptokey(w2[i][0])] = w2[i][1]
        for i in range(len(w3)):
            array3[self._nptokey(w3[i][0])] = w3[i][1]

        aux1 = []
        for i in range(len(w1)):
            for j in range(len(w2)):
                if self._indic(array3, self._nptokey(-w1[i][0] - w2[j][0])) != 0:
                    aux1.append([w1[i][0], w2[j][0], -w1[i][0] - w2[j][0]])
        dim1 = [0]
        for i in range(len(aux1)):
            dim1.append(dim1[i] + self._indic(array1, self._nptokey(aux1[i][0]))*
                                  self._indic(array2, self._nptokey(aux1[i][1]))*
                                  self._indic(array3, self._nptokey(aux1[i][2])))
        b1, e1 = {}, {}
        b3 = {}
        for i in range(len(aux1)):
            key = tuple([self._nptokey(el) for el in aux1[i]])
            b1[key] = dim1[i] + 1
            e1[key] = dim1[i + 1]

        bigMatrix = []
        for i in range(self._n):
            aux2 = {}

            for j in range(len(aux1)):
                if self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])) != 0:
                    val = [aux1[j][0] + self.ncm[i], aux1[j][1], aux1[j][2]]
                    key = tuple([self._nptokey(el) for el in val])
                    if key not in aux2:
                        aux2[key] = val
                if self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])) != 0:
                    val = [aux1[j][0], aux1[j][1] + self.ncm[i], aux1[j][2]]
                    key = tuple([self._nptokey(el) for el in val])
                    if key not in aux2:
                        aux2[key] = val
                if self._indic(array3, self._nptokey(aux1[j][2] + self.ncm[i])) != 0:
                    val = [aux1[j][0], aux1[j][1], aux1[j][2] + self.ncm[i]]
                    key = tuple([self._nptokey(el) for el in val])
                    if key not in aux2:
                        aux2[key] = val

            if len(w1) == 1 and len(w2) == 1 and len(w3) == 1:  # Special care is needed if all reps are singlets
                aux2 = aux1
            else:
                aux2 = aux2.values()

            dim2 = [0]
            for k, val in enumerate(aux2):
                dim2.append(dim2[k] + self._indic(array1, self._nptokey(val[0]))*
                                      self._indic(array2, self._nptokey(val[1]))*
                                      self._indic(array3, self._nptokey(val[2])))

            b2, e2 = {}, {}
            for k, val in enumerate(aux2):
                key = tuple([self._nptokey(el) for el in val])
                b2[key] = dim2[k] + 1
                e2[key] = dim2[k + 1]
            if dim2[len(aux2)] != 0 and dim1[len(aux1)] != 0:
                matrixE = sMat(dim2[len(aux2)], dim1[len(aux1)])
            else:
                matrixE = []

            for j in range(len(aux1)):
                i1, i2, i3 = (self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])),
                              self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])),
                              self._indic(array3, self._nptokey(aux1[j][2] + self.ncm[i])))

                if i1 != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0] + self.ncm[i], aux1[j][1], aux1[j][2]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])

                    slice1 = slice(self._indic(b2, kaux4) - 1, self._indic(e2, kaux4))
                    slice2 = slice(self._indic(b1, kaux3) - 1, self._indic(e1, kaux3))

                    m1 = self._blockW(aux1[j][0] + self.ncm[i], aux1[j][0], w1, r1[i][0])
                    m2 = eye(self._indic(array2, self._nptokey(aux1[j][1])))
                    m3 = eye(self._indic(array3, self._nptokey(aux1[j][2])))

                    matrixE[slice1, slice2] = m1.kroneckerProduct(m2).kroneckerProduct(m3)

                if i2 != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0], aux1[j][1] + self.ncm[i], aux1[j][2]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])

                    slice1 = slice(self._indic(b2, kaux4) - 1, self._indic(e2, kaux4))
                    slice2 = slice(self._indic(b1, kaux3) - 1, self._indic(e1, kaux3))

                    m1 = eye(self._indic(array1, self._nptokey(aux1[j][0])))
                    m2 = self._blockW(aux1[j][1] + self.ncm[i], aux1[j][1], w2, r2[i][0])
                    m3 = eye(self._indic(array3, self._nptokey(aux1[j][2])))

                    matrixE[slice1, slice2] = m1.kroneckerProduct(m2).kroneckerProduct(m3)

                if i3 != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0], aux1[j][1], aux1[j][2] + self.ncm[i]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])

                    slice1 = slice(self._indic(b2, kaux4) - 1, self._indic(e2, kaux4))
                    slice2 = slice(self._indic(b1, kaux3) - 1, self._indic(e1, kaux3))

                    m1 = eye(self._indic(array1, self._nptokey(aux1[j][0])))
                    m2 = eye(self._indic(array2, self._nptokey(aux1[j][1])))
                    m3 = self._blockW(aux1[j][2] + self.ncm[i], aux1[j][2], w3, r3[i][0])

                    matrixE[slice1, slice2] = m1.kroneckerProduct(m2).kroneckerProduct(m3)

            if bigMatrix == [] and matrixE != []:
                bigMatrix = matrixE
            elif bigMatrix != [] and matrixE != []:
                bigMatrix = bigMatrix.append(matrixE, axis=0)

        if len(bigMatrix) == 0:
            return [], [0, 0, 0]

        dim1 = [0]
        dim2 = [0]
        dim3 = [0]
        for i in range(len(w1)):
            dim1.append(dim1[i] + w1[i][1])
        for i in range(len(w2)):
            dim2.append(dim2[i] + w2[i][1])
        for i in range(len(w3)):
            dim3.append(dim3[i] + w3[i][1])

        for i in range(len(w1)):
            b1[self._nptokey(w1[i][0])] = dim1[i]
        for i in range(len(w2)):
            b2[self._nptokey(w2[i][0])] = dim2[i]
        for i in range(len(w3)):
            b3[self._nptokey(w3[i][0])] = dim3[i]

        aux4 = bigMatrix.nullSpace()
        # aux4 = bigMatrix.nullSpace(progress=True)

        resTensor = []
        maxInds = [0, 0, 0]

        for i0, nsDic in enumerate(aux4.values()):
            resTensor.append(sTensor(*[self.dimR(r) for r in reps]))
            count = 0
            for i in range(len(aux1)):
                r1, r2, r3 = (self._indic(array1, self._nptokey(aux1[i][0])),
                              self._indic(array2, self._nptokey(aux1[i][1])),
                              self._indic(array3, self._nptokey(aux1[i][2])))

                bi1, bi2, bi3 = (b1[self._nptokey(aux1[i][0])],
                                 b2[self._nptokey(aux1[i][1])],
                                 b3[self._nptokey(aux1[i][2])])

                for j1 in range(r1):
                    aInd = bi1 + j1
                    if aInd > maxInds[0]:
                        maxInds[0] = aInd

                    for j2 in range(r2):
                        bInd = bi2 + j2
                        if bInd > maxInds[1]:
                            maxInds[1] = bInd

                        for j3 in range(r3):
                            cInd = bi3 + j3
                            if cInd > maxInds[2]:
                                maxInds[2] = cInd

                            if count in nsDic:
                                resTensor[i0][aInd, bInd, cInd] = nsDic[count]

                            count += 1

        return resTensor, maxInds

    def _invariants4Irrep(self, otherStuff, reps, cjs):
        """
        Returns the invariants for four irreps
        """
        result = []
        if len(reps) == 3:
            if len(otherStuff) > 1:
                exit(" LENGTH OTHER STUFF > 1 ! Should not get here. Exiting")

            aux1 = self.invariants(reps, cjs, skipSymmetrize=True, realBasis=None)
            subs = len(otherStuff)*(None,) + (0, 1, 2)

            # do the permutations
            aux1 = [el.permute(subs) for el in aux1]
            aux2 = otherStuff[0]

            for i, el in enumerate(aux1):
                for j, elem in enumerate(aux2):
                    tp = el.sub(elem)
                    result.append(tp)

            return result

        trueReps = [tuple(self.conjugateIrrep(el)) if cjs[iel] else el for iel, el in enumerate(reps)]

        # find the irreps in the product of the first two representations
        aux1 = [el[0] for el in self.reduceRepProduct([trueReps[0], trueReps[1]])]
        # conjugate them
        aux1 = [self.conjugateIrrep(el) for el in aux1]
        # do the same for the rest of the irreps
        aux2 = [el[0] for el in self.reduceRepProduct([el for el in trueReps[2:]])]

        # get the intersection and sort it by (dimension, reverse[DynkinLabels])
        aux1 = sorted([list(elem) for
                       elem in list(set([tuple(el) for el in aux1]).intersection(set([tuple(el) for el in aux2])))],
                      key=lambda x: (self.dimR(x), x[::-1]))

        for i in range(len(aux1)):
            # Fix from Susyno v3.7
            aux2 = self._irrepInProduct([reps[0], reps[1], aux1[i]], cjs=[cjs[0], cjs[1], True])
            # aux2 = self._irrepInProduct([reps[0], reps[1], aux1[i]], cjs=[cjs[0], cjs[1], False])

            subs = len(otherStuff)*(None,) + (0, 1)

            aux2 = [[inv.permute(subs) for inv in el] for el in aux2]
            aux3 = [(None,)*(len(otherStuff)+1) + (el,) + (None,)*(2-len(otherStuff)) for el in range(self.dimR(aux1[i]))]
            aux2 = [{el1: el2 for el1, el2 in zip(aux3, elem)} for elem in aux2]

            # warning otherstuff should not be appended in this scope
            otherStuffcp = cp.deepcopy(otherStuff)
            otherStuffcp.append(aux2)

            tp1 = reps[2:]
            tp1.insert(0, aux1[i])
            # do some type conversion
            tp1 = [list(el) for el in tp1]
            tp2 = cjs[2:]
            tp2.insert(0, True)

            res = self._invariants4Irrep(otherStuffcp, tp1, tp2)

            for resTensor in res:
                resTensor.dim = [self.dimR(el) for el in reps]

            result.append(res)

        return flatten(result, cls=list), [self.dimR(el) for el in reps]


    def _irrepInProduct(self, reps, cjs=[]):
        """
        calculate the combination of rep1xrep2 that transform as rep3
        """
        if cjs == []:
            cjs = [False] * len(reps)

        # Fix from Susyno v3.6
        cjs[2] = not(cjs[2])

        aux = self.invariants(reps, cjs, skipSymmetrize=True, realBasis=None)

        vector = set()
        for inv in aux:
            vector = vector.union(inv.subDics[2].keys())
        vector = sorted(vector)

        return [[inv.subTensors[(None,None,v,None)] for v in vector] for inv in aux]


    def _pyrateNormalization(self, tensor):
        # Normalize the invariants in a simpler way
        # For that we look at the first element in the invariants and divide by its value

        def minKey(inv):
            absMin = (0,)*inv.rank+(None,)*(4-inv.rank)
            if absMin in inv.dic:
                return absMin
            return min(inv.dic.keys())

        for inv in tensor:
            firstEl = inv.dic[minKey(inv)]

            for k,v in inv.dic.items():
                # inv[k] = simplify(v/firstEl)
                inv[k] = v/firstEl

        return tensor

    def _normalizeInvariants(self, invs, repDims, pyNorm=False):
        """ Normalize the invariants according to sqrt(Prod_n Dim(rep_n)).
            Note that it also orthogonalize them!! """

        N = len(invs)
        aux = sMat(N, N)

        for i, el1 in enumerate(invs):
            for j, el2 in enumerate(invs):
                # The resulting matrix is symmetric
                if j >= i:
                    aux[i, j] = (el1*el2).sum()
                else:
                    aux[i, j] = aux[j, i]

        aux = self.math._decompositionTypeCholesky(aux)

        # for k,v in aux._smat.items():
        #     aux[k] = expand(v)

        aux = aux.inv() * sqrt(repDims)

        for k,v in list(aux.todok().items()):
            aux[k] = expand(v)

        # Finally perform matrix multiplication aux*invs
        normalizedInvs = [0]*N
        for i in range(N):
            normalizedInvs[i] = aux[i,0]*invs[0]

            for j, tensor in enumerate(invs[1:]):
                normalizedInvs[i] += aux[i,j+1]*tensor

        return normalizedInvs


    def _goToRealBasis(self, rep, realBasis):
        """ Returns whether a given rep should be rotated to a real basis """

        if realBasis is False:
            realBasis = None
        if realBasis is True:
            realBasis = 'all'

        if realBasis is None:
            return False
        if realBasis == 'adjoint':
            return tuple(rep) == tuple(self.adjoint)
        if realBasis == 'all':
            return tuple(rep) == tuple(self.adjoint) or self.frobeniusSchurIndicator(rep) == 0

    def _realBasisRotation(self, rep):
        tag = tuple(rep)

        if self.frobeniusSchurIndicator(rep) == 1:
            print(f"Representation {rep} is complex.")
            return
        if self.frobeniusSchurIndicator(rep) == -1:
            print(f"Representation {rep} is pseudo-real.")
            return

        if tag in self._realBasisDic:
            return self._realBasisDic[tag]

        # Lock
        self._computeReal = True

        if tag != tuple(self.adjoint):
            inv = self.invariants([rep, rep], skipSymmetrize=True)
            mat = sMat(self.dimR(rep), self.dimR(rep), {k[:2]:v for k,v in inv[0].dic.items()})

            taka = mat.takagi()
            if tag not in self._realBasisDic:
                self._realBasisDic[tag] = taka

            self._computeReal = False
            return taka

        # For adjoint rep, make sure that (T_adj^i)_jk = -I f[i,j,k]
        struc = self.structureConstants()
        adjRepMat = self.repMatrices(self.adjoint)

        bigMatrix = []
        idMat = eye(self.dimAdj)

        for i, mat in enumerate(adjRepMat):
            tmp = mat.kroneckerProduct(idMat) - I*idMat.kroneckerProduct(struc[i])

            if bigMatrix == []:
                bigMatrix = tmp
            else:
                bigMatrix = bigMatrix.append(tmp, axis=0)

        vecForm = bigMatrix.nullSpace()[0]
        norm = sum([el*conjugate(el) for el in vecForm.values()])

        rotMat = sMat(self.dimAdj, self.dimAdj)

        for k,v in vecForm.items():
            rotMat[k//self.dimAdj, k%self.dimAdj] = v*sqrt(self.dimAdj/norm)

        if tag not in self._realBasisDic:
            self._realBasisDic[tag] = rotMat

        self._computeReal = False

        return rotMat

    def _pseudoMetric(self, rep):
        if rep[-1] is True:
            cj = True
            rep = rep[:-1]
        else:
            cj = False

        if self.frobeniusSchurIndicator(rep) != -1:
            print(f"Representation {rep} is not pseudo-real.")
            return

        quadraticInv = self.invariants([rep, rep])[0]
        metric = sMat(*quadraticInv.dim[:2], {k[:2]: v for k,v in quadraticInv.dic.items()})

        if not cj:
            return metric
        return -1*metric.transpose()

    def _rotateInvariants(self, reps, realBasis, invs, conj, maxinds):
        """ Rotate invariants to real basis """

        toReal = [self._goToRealBasis(el, realBasis) for el in reps]

        if self._computeReal or not any(toReal):
            return invs

        # Build the newInvs dic
        newInvs = []
        for i, el in enumerate(invs):
            newInvs.append(sTensor(*el.dim, forceSub=True))
            for k,v in el.dic.items():
                newInvs[i][k] = v

        for i, real in enumerate(toReal):
            if real:
                subs = {}
                rotMat = self._realBasisRotation(reps[i])
                for k,v in rotMat.todok().items():
                    subFrom = tuple([k[0] if p==i else None for p in range(4)])
                    subTo = tuple([k[1] if p==i else None for p in range(4)])

                    if subFrom not in subs:
                        subs[subFrom] = sTensor(*[maxinds[p] if p==i else None for p in range(4)])

                    if not conj[i]:
                        subs[subFrom][subTo] = v
                    else:
                        subs[subFrom][subTo] = conjugate(v)

                newInvs = [el.sub(subs) for j, el in enumerate(newInvs)]

        return newInvs

    def _symmetrizeInvariants(self, reps, invs, maxinds, cjs):
        if len(invs) <= 1:
            return invs

        #Remove duplicate reps
        aux1 = []
        for el in reps:
            if el not in aux1:
                aux1.append(el)

        #Positions of the reps
        aux2 = [[ix for ix, x in enumerate(reps) if x == y] for y in aux1]

        cjs = np.array(cjs)
        # collect the position of the False and True cjs
        aux3 = [[[np.array(el)[pos] for pos in [ix for ix, x in enumerate(cjs[el]) if x]],
                 [np.array(el)[pos] for pos in [ix for ix, x in enumerate(cjs[el]) if not (x)]]]
                for el in aux2]

        fakeConjugationCharges = np.zeros(len(reps), dtype=int)

        for i in range(len(aux3)):
            fakeConjugationCharges[aux3[i][0]] = len(aux3[i][1])
            fakeConjugationCharges[aux3[i][1]] = len(aux3[i][0])

        representations = list(zip(reps, fakeConjugationCharges))
        representations = [self.conjugateIrrep(representations[i], u1in=True) if cjs[i] else representations[i] for i in
                           range(len(representations))]
        representations = [(tuple(el[0]),el[1]) for el in representations] #force the conjugated irreps to be tuples for tally below

        if (len(self.math.tally(representations)) == len(representations)):
            return invs

        symmetries = self.permutationSymmetryOfInvariants(representations, u1in=True)
        flattenedInvariants, columns = self.flattenInvariants(invs, maxinds)

        maxRank = 0
        columnsToTrack = []

        for i, c in enumerate(columns):
            aux = flattenedInvariants[:, columnsToTrack + [c]].rank()
            if aux > maxRank:
                columnsToTrack.append(c)
                maxRank = aux

                if aux == len(invs):
                    break
        flattenedInvariants = flattenedInvariants[:, columnsToTrack]

        try:
            invRef = flattenedInvariants.pinv()
        except:
            exit("impossible to calculate the pseudo inverse in `symmetrizeInvariants`.")

        # [END] Don't handle the complete invariants: just find a minimum set of entries which reveal the linear independence of the invariants
        # [START]Generate the Sn transformation matrices of the invariants under permutations of each set of equal representations

        permuteInvs12 = []
        permuteInvs12n = []

        for groupOfIndicesI in range(len(symmetries[0])):
            groupOfIndices = np.array(symmetries[0][groupOfIndicesI], dtype=int) - 1
            if len(groupOfIndices) > 1:
                aux = np.arange(len(representations), dtype=int)
                aux[groupOfIndices[[0, 1]]] = aux[groupOfIndices[[0, 1]]][::-1]

                perm = [el[0] for el in sorted([(iel, el) for iel, el in enumerate(aux)], key=lambda x: x[1])]
                p = [inv.permute(perm) for inv in invs]
                p = self.flattenInvariants(p, maxinds)[0][:, columnsToTrack]
                permuteInvs12.append(p)

                aux = np.arange(len(representations), dtype=int)
                aux[groupOfIndices] = aux[self.math._rotateleft(groupOfIndices, 1, numpy=True)]


                perm = [el[0] for el in sorted([(iel, el) for iel, el in enumerate(aux)], key=lambda x: x[1])]
                p = [inv.permute(perm) for inv in invs]
                p = self.flattenInvariants(p, maxinds)[0][:, columnsToTrack]
                permuteInvs12n.append(p)

        refP12 = [el * invRef for el in permuteInvs12]
        refP12n = [el * invRef for el in permuteInvs12n]

        # [END]Generate the Sn transformation matrices of the invariants under permutations of each set of equal representations
        # The standardized Sn irrep generators
        newStates = []
        for snIrrep in symmetries[1]:
            aux0 = []
            for groupOfIndicesI in range(len(snIrrep[0])):
                if len(symmetries[0][groupOfIndicesI]) > 1:
                    ids = [eye(self.Sn.snIrrepDim(el)) for el in snIrrep[0]]
                    aux = self.Sn.snIrrepGenerators(snIrrep[0][groupOfIndicesI])
                    aux2 = [el for el in ids]
                    aux2[groupOfIndicesI] = aux[0]
                    if len(symmetries[0]) > 1:
                        tmp = aux2[0]
                        for el in aux2[1:]:
                            tmp = tmp.kroneckerProduct(el)
                        aux2 = tmp
                    else:
                        aux2 = aux2[0]
                    aux3 = [el for el in ids]
                    aux3[groupOfIndicesI] = aux[1]
                    if len(symmetries[0]) > 1:
                        tmp = aux3[0]
                        for el in aux3[1:]:
                            tmp = tmp.kroneckerProduct(el)
                        aux3 = tmp
                    else:
                        aux3 = aux3[0]

                    aux0.append([aux2, aux3])

            aux4 = [x.kroneckerProduct(y, simplify=True).transpose() for x, y in zip([ell[0] for ell in aux0], refP12)]
            aux5 = [x.kroneckerProduct(y, simplify=True).transpose() for x, y in zip([ell[1] for ell in aux0], refP12n)]

            aux4 = aux4 + aux5
            bigMatrix = aux4[0] - eye(aux4[0].shape[0])

            for i in range(1, len(aux4)):
                bigMatrix = bigMatrix.append(aux4[i] - eye(aux4[i].shape[0]), axis=0)

            aux4 = [self.math._inverseFlatten(el, [len(aux0[0][0]), refP12[0].shape[0]])
                                          for el in bigMatrix.nullSpace(vecForm=True)]

            aux4 = flatten(aux4, cls=list)

            aux4 = GramSchmidt(aux4, True)

            # Sympy 1.6 compatibility
            for i, el in enumerate(aux4):
                if not isinstance(el, sMat):
                    aux4[i] = sMat(el)

            if newStates == []:
                newStates = aux4[0]
                for el in aux4[1:]:
                    newStates = newStates.append(el, axis=1)
            else:
                for el in aux4:
                    newStates = newStates.append(el, axis=1)

        result = [0]*len(invs)

        # Performing matrix multiplication transpose(newStates)*invs
        for k,v in newStates.todok().items():
            if result[k[1]] == 0:
                result[k[1]] = v*invs[k[0]]
            else:
                result[k[1]] += v*invs[k[0]]

        return result


    def flattenInvariants(self, invs, maxinds):
        nMax = 1

        for m in maxinds:
            nMax *= (1+m)

        mat = sMat(len(invs), nMax)

        def posFuc(tup):
            p = 0
            for i, m in enumerate(maxinds):
                p *= (m+1)
                p += tup[i]
            return int(p)

        nonZeroCols = dict()
        for i, inv in enumerate(invs):
            for k, v in inv.dic.items():
                p = posFuc(k)
                mat[i, p] = v
                if p not in nonZeroCols:
                    nonZeroCols[p] = 1
                else:
                    nonZeroCols[p] += 1

        return mat, sorted(nonZeroCols, key=lambda x:nonZeroCols[x])



    def permutationSymmetryOfInvariants(self, listofreps, u1in=False):
        """
        Computes how many invariant combinations there are in the product of the representations of the gauge group
         provided, together with the information on how these invariants change under a permutation of the representations
         - The output is rather complex (see the examples below). It is made of two lists: {indices, SnRepresentations}.
          The first one (indices) indicates the position of equal representations in the  input list. So indices={G1, G2, \[CenterEllipsis]}
          where each GI lists the positions of a group of equal representations. For example, if the input list is {Subscript[R, 1], Subscript[R, 2],Subscript[R, 1], Subscript[R, 2]} for some representation Subscript[R, 1], Subscript[R, 2] of the gauge group, indices will be {{1,3},{2,4}} (the representations in positions 1 and 3 are the same, as well as the ones in the positions 2 and 4). The second list (SnRepresentations) is itself a list {SnRep1, SnRep2, \[CenterEllipsis]} with the break down of the gauge invariants according to how they change under permutations of equal representations. Specifically, each SnRepI is of the form {{SnRepIG1, SnRepIG2, \[CenterEllipsis]}, multiplicity} where each SnRepIGJ is the irreducible representation of an Subscript[S, n] induced when the same fields in the grouping GJ are permuted. multiplicity indicates how many times such a gauge invariant is contained in the product of the representations of the gauge group provided.
        :param listofreps:
        :return:
        """
        indices, invariants = self._permutationSymmetryOfInvariantsProductParts(listofreps, u1in=u1in)
        invariants = [el for el in invariants if np.all(np.array(el[0][0]) * 0 == np.array(el[0][0]))]
        invariants = [[el[0][1], el[1]] for el in invariants]
        return [indices, invariants]

    def _permutationSymmetryOfInvariantsProductParts(self, listofreps, u1in=False):
        """
        This calculates the Plethysms in a tensor product of different fields/representations *)
        """
        if not (u1in):
            listofreps = [[el] for el in listofreps]
        aux1 = self.math.tally(listofreps)
        plesthysmFields = [[i + 1 for i, el in enumerate(listofreps) if el == ell[0]] for ell in aux1]
        aux2 = [self._permutationSymmetryOfInvariantsProductPartsAux(aux1[i][0], aux1[i][1]) for i in range(len(aux1))]
        aux2 = self.math._tuplesWithMultiplicity(aux2)
        aux3 = [self.reduceRepProduct([ell[0][0] for ell in el[0]]) for el in aux2]
        aux3 = sum([[[[aux3[i][j][0], [el[1] for el in aux2[i][0]]], aux3[i][j][1] * aux2[i][1]]
                     for j in range(len(aux3[i]))]
                    for i in range(len(aux3))], [])
        aux3 = self.math._tallyWithMultiplicity(aux3)
        return [plesthysmFields, aux3]

    def _permutationSymmetryOfInvariantsProductPartsAux(self, rep, n):
        intPartitionsN = list(self.math._partitionInteger(n))
        # this differs from the original algo because we only consider a single group factor
        aux = self.math._tuples(intPartitionsN, 1)
        snPart = [self.Sn.decomposeSnProduct(el) for el in aux]
        aux = [self._plethysms(rep[j], i[j]) for i in aux for j in range(len(i))]
        aux = [self.math._tuplesWithMultiplicity([el]) for el in aux]
        aux = [[[[[aux[i][j][0], intPartitionsN[k]], aux[i][j][1] * snPart[i][k]]
                 for k in range(len(intPartitionsN))]
                for j in range(len(aux[i]))]
               for i in range(len(aux))]
        aux = [el for el in sum(sum(aux, []), []) if not (el[-1] == 0)]
        result = self.math._tallyWithMultiplicity(aux)
        return result

    def _plethysms(self, weight, partition):
        n = sum(partition)
        kList = list(self.math._partitionInteger(n))

        summing = []
        for i in range(len(kList)):
            factor = 1 / factorial(n) * self.Sn.snClassOrder(kList[i]) * self.Sn.snClassCharacter(partition, kList[i])
            aux = [self._adams(el, weight) for el in kList[i]]
            aux = self._reduceRepPolyProduct(aux)
            aux = [(el[0], factor * el[1]) for el in aux]
            summing.append(aux)
        summing = self._gatherWeightsSingle(summing)
        return summing

    def _reduceRepPolyProduct(self, polylist):
        """
        (* This method calculates the decompositions of a product of sums of irreps: (R11+R12+R13+...) x (R21+R22+R23+...) x ... *)
        (* polyList = list of lists of representations to be multiplied. The method outputs the decomposition of such a product *)
        """
        n = len(polylist)
        aux = polylist[0]
        if n <= 1:
            return aux
        for i in range(n - 1):
            aux = list(self.math._tuplesList([aux, polylist[i + 1]]))
            aux2 = [self.reduceRepProduct([ell[0] for ell in el[0:2]]) for el in aux]
            aux = self._gatherWeights(aux2, [el[0][1] * el[1][1] for el in aux])
        return aux

    def _gatherWeightsSingle(self, llist):
        aux = sum(llist, [])
        aux = self.math._gatherAux(aux)
        aux = [[el[0][0], sum([ell[1] for ell in el])] for el in aux]
        aux = [el for el in aux if el[1] != 0]
        return aux

    def _gatherWeights(self, listW, listMult):
        aux = [[[el[0], listMult[i] * el[1]] for el in listW[i]] for i in range(len(listW))]
        aux = sum(aux, [])
        aux = self.math._gatherAux(aux)
        aux = [[el[0][0], sum([ell[1] for ell in el[0:]])] for el in aux]
        aux = [el for el in aux if el[1] != 0]
        return aux

    def _adams(self, n, rep):
        aux = self._dominantWeights(rep)
        aux = [((el[0] * n).tolist()[0], el[1]) for el in aux]
        result = [[self._vdecomp(aux[i][0]), aux[i][1]] for i in range(len(aux))]
        result = [[[result[i][0][j][0], result[i][0][j][1] * result[i][1]]
                   for j in range(len(result[i][0]))]
                  for i in range(len(result))]
        result = sum(result, [])
        return result

    def _vdecomp(self, dominantWeight):
        return self._altDom1Arg(self._weylOrbit(dominantWeight))

    def _altDom1Arg(self, weights):
        return self._altDom(weights, self.longestWeylWord)

    def _altDom(self, weights, weylWord):
        prov = [[el, 1] for el in weights]
        for i in range(len(weylWord)):
            for j in range(len(prov)):
                if prov[j][1] != 0:
                    if prov[j][0][weylWord[i] - 1] >= 0:
                        pass
                    elif prov[j][0][weylWord[i] - 1] == -1:
                        prov[j][1] = 0
                    elif prov[j][0][weylWord[i] - 1] <= -2:
                        prov[j][1] = - prov[j][1]
                        provMat = sMat([prov[j][0]]) if not isinstance(prov[j][0], sMat) else prov[j][0]
                        prov[j][0] = provMat - int((prov[j][0][weylWord[i] - 1] + 1)) * self.cm[weylWord[i] - 1, :]
        prov = [[list(el[0]), el[1]] for el in prov]
        prov = [el for el in prov if not (el[1] == 0)]
        return prov


    def _blockW(self, w1, w2, listW, repMat):
        """
        aux function to construct the invariants
        """
        dim = [0]
        for i in range(len(listW)):
            dim.append(dim[i] + listW[i][1])
        b, e = {}, {}
        for i in range(len(listW)):
            key = self._nptokey(listW[i][0])
            b[key] = dim[i] + 1
            e[key] = dim[i + 1]
        aux1 = repMat[b[self._nptokey(w1)] - 1:e[self._nptokey(w1)],
                      b[self._nptokey(w2)] - 1:e[self._nptokey(w2)]]

        return aux1


    #  AUXILIARY FUNCTIONS #

    # This is a fix for Python3 compatibility (cmp built-in function removed)
    def _cmp(self, a,b):
        if sorted([a,b]) == [a,b]:
            return -1
        elif a==b:
            return 0
        return 1

    def _nptokey(self, array):
        return tuple(array.ravel())

    def _tolist(self, array):
        return list(array.ravel())

    def _indic(self, dic, key):
        if key in dic:
            return dic[key]
        else:
            return 0

    def _empty(self, array):
        if isinstance(array, list):
            return array == [[]]
        if isinstance(array, np.ndarray):
            return array.size == 0
        if array == 0:
            return True
        if isinstance(array, sMat):
            return array.todok() == {}

        exit("What type of array ?")
