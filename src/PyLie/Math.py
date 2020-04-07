from sys import exit
import sys

import copy as cp
import numpy as np

import operator
import itertools
from functools import reduce

from sympy.combinatorics import Permutation
from sympy import (Abs, Add, I, Matrix, Mul, SparseMatrix, Symbol, Rational,
                   conjugate, eye, diag, factorial, flatten, im, sqrt, zeros)

from sympy import expand, symbols, solve, simplify as sSimplify, linsolve, pprint, GramSchmidt

import time

class Sn:
    def __init__(self):
        # declare a MathGroup object to access the standard method
        self.math = MathGroup()

    def snIrrepGenerators(self, Lambda, orthogonalize=True):
        """
        returns the matrices that generate Sn
        - representation must be a partition of some integer n, as irreducible representations of  Subscript[S, n] are specified in this way;
        - Note with the (12) and (12...n) elements of the Subscript[S, n] group alone, it is possible to generate all remaining group elements, for any n;
        - This function returns two real orthogonal/unitary matrices which are the representation matrices of the elements (12) and (12...n) elements of the Sn group. 
        If orthogonality is not required, the option OrthogonalizeGenerators->False can be used -- the resulting matrices have less complicated values,
        and the code is executed faster.
        """
        n = sum(Lambda)
        sts = self.generateStandardTableaux(Lambda)
        
        basicPermutations = [Permutation(1, 2), Permutation(*range(1, n + 1))]
        # because the length of the lists on which we apply the permutations is not constant we need to resize them for each element
        tabloids, stsX = [], []
        
        for perm in basicPermutations:
            stscp = cp.deepcopy(sts)
            sts_nosort = cp.deepcopy(sts)
            for iel, el in enumerate(stscp):
                for iell, ell in enumerate(el):
                    for ixel, xel in enumerate(ell):
                        # if xel in perm.args[0]:
                        if xel in perm._array_form:
                            stscp[iel][iell][ixel] = perm(xel)
                            sts_nosort[iel][iell][ixel] = perm(xel)
                    stscp[iel][iell] = sorted(stscp[iel][iell])
            # TODO IMPLEMENT THE DELETE DUPLICATES
            tabloids.append(stscp)
            stsX.append(sts_nosort)
        
        X = [sMat(len(sts), len(tabloids[0])) for _ in range(2)]
        Y = [sMat(len(sts), len(tabloids[0])) for _ in range(2)]
        
        for alpha in range(2):
            for i in range(len(sts)):
                for j in range(len(tabloids[alpha])):
                    startingTableauxY = sts[i]
                    startingTableauxX = stsX[alpha][i]
                    targetTabloid = tabloids[alpha][j]
                    tmp = [[self.math._position_in_array(targetTabloid, ell)[0][0] for ell in el] for el in
                           self._transposeTableaux(startingTableauxY)]
                    Y[alpha][i, j] = reduce(operator.mul,
                                            [0 if sorted(el) != list(range(len(el))) else Permutation(el).signature() for el
                                             in tmp])
                    tmp = [[self.math._position_in_array(targetTabloid, ell)[0][0] for ell in el] for el in
                           self._transposeTableaux(startingTableauxX)]
                    X[alpha][i, j] = reduce(operator.mul,
                                            [0 if sorted(el) != list(range(len(el))) else Permutation(el).signature() for el
                                             in tmp])
        
        result = [(X[i] * Y[i].inv()).transpose() for i in range(2)]
        
        # Finally let's orthogonalize the generators P_i
        # Oi = B.Pi.Inverse[B], Oi are ortho and B the change of basis
        # since Pi are real Pi^T.(B^T.B).Pi = B^T.B
        # If both Pi are taken into consideration, this fixes completely B^T.B as the Pi are generators of the group in
        # an irreducible representation
        # With KroneckerProduct and NullSpace, B^T.B can be found, and B can be obtained with the CholeskyTypeDecomposition
        if orthogonalize:
            Id = sMat.id((result[0].shape[0]) ** 2)
            aux = [el.conjugate().kroneckerProduct(el).transpose() for el in result]
            ns = (aux[0] - Id).append(aux[1] - Id, axis=0).nullSpace2(vecForm=True)
            
            if ns != []:
                ns = ns[0]
            else:
                exit("Impossible to find null space in SnGenerator.")
            BcB = self.math._inverseFlatten(ns, [result[0].shape[0], result[0].shape[0]])
            B = sMat(self.math._decompositionTypeCholesky(np.array(BcB))).transpose()
            
            result = [B * el * B.inv() for el in result]
        return result

    def decomposeSnProduct(self, partitionsList):
        """
        This method decomposes the product of a list of Sn rep into its irreducible parts
        """
        n = sum(partitionsList[0])
        result = [1 / factorial(n) * sum([
                                             self.snClassOrder(i) * reduce(operator.mul, [
                                                 self.snClassCharacter(inputPartition, list(i)) for inputPartition in
                                                 partitionsList])
                                             * self.snClassCharacter(list(j), list(i)) for i in
                                             list(self.math._partitionInteger(n))]) for j in
                  list(self.math._partitionInteger(n))]
        return result

    def snClassOrder(self, partition):
        """
        size of a given conjugacy class of Sn. The formula is easy but see for example
         Enumerative Combinatorics", Richard P.Stanley, http://math.mit.edu/~rstan/ec/ec1.pdf, 1.3.2 Proposition"
        """
        n = sum(partition)
        aux = self.math.tally(partition)
        return factorial(n) / (
            reduce(operator.mul, [aux[i][0] ** aux[i][1] * factorial(aux[i][1]) for i in range(len(aux))]))

    def snClassCharacter(self, partitionL, partitionM):
        """
        (* See arXiv:math/0309225v1[math.CO] for the way to compute SnClassCharacter from the Murnaghan-Nakayama rule  *)
(* \[Lambda] is the representation; \[Mu] is the conjugacy class. This method computes the character of conjugacy class \mu in the irreducible representation \[Lambda]  *
        """
        if len(partitionL) == 0:
            return 1
        n = sum(partitionL)
        if n != sum(partitionM):
            exit("Error in SnClassCharacter method: both partitions must be of the same order.")
            return
        newL = self.rimHooks(partitionL, partitionM[0])
        newM = partitionM[1:]
        result = sum([(-1) ** newL[i][1] * self.snClassCharacter(newL[i][0], newM) for i in range(len(newL))])
        return result

    def rimHooks(self, partition, l):
        """
        (* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
        (* This method finds all the rim hooks \[Xi] with length l and returns a list with all the possibilities {partition\\[Xi], leg length of rim hook \[Xi]} which is writen as {partition\\[Xi],ll(\[Xi])}*)
        """
        sequence = self._partitionSequence(partition)
        result = []
        for i in range(len(sequence) - l):
            if sequence[i] == 1 and sequence[i + l] == 0:
                seqMinusHook = cp.deepcopy(sequence)
                seqMinusHook[i] = 0
                seqMinusHook[i + l] = 1
                length = sequence[i:i + l + 1].count(0) - 1
                result.append((self._rebuildPartitionFromSequence(seqMinusHook), length))
        return result

    def checkStandardTableaux(self, tab):
        """
        Returns True if tab is a standard tableau i.e. it grows on each line and each columns
        """
        transpose = self._transposeTableaux(tab)
        return all([self.math._issorted(el) for el in tab] + [self.math._issorted(el) for el in transpose])

    def _transposeTableaux(self, tab):
        """
        Transpose a tableaux
        """
        tabcp = cp.deepcopy(tab)
        for iel, el in enumerate(tabcp):
            tabcp[iel] = el + [None] * (len(tabcp[0]) - len(el))
        tabcp = np.array(tabcp).T.tolist()
        for iel, el in enumerate(tabcp):
            tabcp[iel] = [ell for ell in el if ell is not None]
        return tabcp

    def generateStandardTableaux(self, Lambda):
        """
        Generates all the standard tableaux given by the partition LAmbda
        """
        result = self._generateStandardTableauxAux([[None] * el for el in Lambda])
        return result

    def _generateStandardTableauxAux(self, tab):
        """
        Aux function for the recursion algo
        """
        if not (self.checkStandardTableaux(tab)):
            return []
        # stop criterion for the recursion
        # flatten tab
        flttab = sum(tab, [])
        # stopping creterion for the recursion
        if not (None in flttab):
            return [tab]
        n = len(flttab)
        # flatten removes Nones
        temp = [el for el in flttab if el is not None]
        missingNumbers = [el for el in range(1, n + 1) if not (el in temp)]
        stop = False
        for idi, i in enumerate(tab):
            if stop:
                idi -= 1
                break
            for idj, j in enumerate(i):
                if j == None:
                    stop = True
                    break
        if stop:
            positionNone = [idi, idj]
        else:
            positionNone = []
        result = []
        for el in missingNumbers:
            newT = cp.deepcopy(tab)
            newT[positionNone[0]][positionNone[1]] = el
            tp = self._generateStandardTableauxAux(newT)
            result += tp
        return result

    def hookContentFormula(self, partition, nMax):
        """
        1) Applies the Hook Content Formula to a semi-standard Young tableau with cells filled with the numbers 0, ...,n (repetitions are allowed) - see reference below
        2) Recall that a partition {Lambda_1, Lambda_2, ...} is associated with a Young tableau where row i contains Lambda_i cells - for example the partition {4,3,1,1} of 9 yields the tableau
        3) In a semi-standard Young tableau, the x_i which fill it must increase from top to bottom and must not decrease from left to right.
        4) The number of semi-standard Young tableau given by the partition \[Lambda], where the cell can have any positive integer value smaller or equal to n is given by hookContentFormula(Lambda, n).
        5)The application in model building of this is the following: consider a parameter M_f1f2, ... where the f_i =1,...,n are flavor indices. If Mu is known to have some symmetry (given by a partition Lambda) under a permutation of these indices, then the number of degrees of freedom in Mu is given by  hookContentFormula(Lambda_n) (see examples below).
        """
        n1 = partition[0]
        n2 = len(partition)
        inverseP = [len([ell for ell in partition if ell >= el]) for el in range(1, n1 + 1)]
        if type(nMax) != Symbol:
            aux = [[Rational((nMax + i - j), partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1)
                    if partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1 > 0 else 1 for j in range(1, n2 + 1)]
                   for
                   i in range(1, n1 + 1)]
        else:
            aux = [[(nMax + i - j) / (partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1)
                    if partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1 > 0 else 1 for j in range(1, n2 + 1)]
                   for
                   i in range(1, n1 + 1)]
        result = reduce(operator.mul, flatten(aux))
        return result

    def _partitionSequence(self, partition):
        sequence = [1] * partition[-1]
        sequence.append(0)
        for i in range(1, len(partition)):
            sequence = sequence + [1] * (partition[-i - 1] - partition[-i])
            sequence.append(0)
        return sequence

    def _rebuildPartitionFromSequence(self, sequence):
        """
        (* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
        (* RebuiltPartitionFromSequence[PartitionSequence[partition]]=partition *)
        """
        counter1s = 0
        result = []
        for i in range(len(sequence)):
            if sequence[i] == 0:
                result.insert(0, counter1s)
            else:
                counter1s += 1
        return [el for el in result if el != 0]

    def snIrrepDim(self, partition):
        n1 = partition[0]
        n2 = len(partition)
        inverseP = [len([x for x in partition if x >= el]) for el in range(1, n1 + 1)]
        result = [max([partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1, 1]) for j in range(1, n2 + 1) for i
                  in range(1, n1 + 1)]
        result = factorial(sum(partition)) / reduce(operator.mul, result)
        return result

class MathGroup:
    def __init__(self):
        self.M = []
        pass

    def sumperso(self, listMat):
        out = listMat[0]
        for el in listMat[1:]:
            out += el
        return out

    def _decompositionTypeCholesky_old(self, matrix):
        """
        falls back to the regular Cholesky for sym matrices
        """
        n = len(matrix)
        shape = matrix.shape
        matrix = np.array([int(el) if int(el) == el else el for el in matrix.ravel()], dtype=object).reshape(shape)
        matD = np.zeros((n, n), dtype=object)
        matL = np.eye(n, dtype=object)
        for i in range(n):
            for j in range(i):
                if matD[j, j] != 0:
                    if type(matD[j, j]) in [Add, Mul]:
                        coeff = 1 / matD[j, j]
                    else:
                        coeff = Rational(1, matD[j, j])
                    matL[i, j] = coeff * (
                        matrix[i, j] - sum([matL[i, k] * np.conjugate(matL[j, k]) * matD[k, k]
                                            for k in range(j)])
                    )
                else:
                    matL[i, j] = 0
            matD[i, i] = matrix[i, i] - sum([matL[i, k] * np.conjugate(matL[i, k]) * matD[k, k] for k in range(i)])
        # get the sqrt of the diagonal matrix:
        if np.all(matD.transpose() != matD):
            exit("Error, the matD is not diagonal cannot take the sqrt.")
        else:
            matDsqr = diag(*[sqrt(el) for el in matD.diagonal()])
            result = (matL * matDsqr).transpose()
            #  Make the resulting matrix as small as possible by eliminating null columns
            result = np.array(
                [np.array(result.row(i))[0] for i in range(result.rows) if result.row(i) != zeros(1, n)]).transpose()
        return result
    
    def _decompositionTypeCholesky(self, M):
        """
        falls back to the regular Cholesky for sym matrices
        Sparse Matrix version
        """
        
        shape = M.shape
        
        if shape[0] != shape[1]:
            print("Error : Matrix must be square.")
            return
        
        n = shape[0]
        matrix = M.copy()
        matD = sMat(n, n)
        matL = sEye(n)
        
        for i in range(n):
            for j in range(i):
                if matD[j, j] != 0:
                    if type(matD[j, j]) in [Add, Mul]:
                        coeff = 1 / matD[j, j]
                    else:
                        coeff = Rational(1, matD[j, j])
                    matL[i, j] = coeff * (
                        matrix[i, j] - sum([matL[i, k] * np.conjugate(matL[j, k]) * matD[k, k]
                                            for k in range(j)])
                    )
                else:
                    matL[i, j] = 0
            matD[i, i] = matrix[i, i] - sum([matL[i, k] * np.conjugate(matL[i, k]) * matD[k, k] for k in range(i)])
        # get the sqrt of the diagonal matrix:
        if not all([k[0] == k[1] for k in matD._smat]):
            exit("Error, the matD is not diagonal cannot take the sqrt.")
        else:
            matDsqr = sMat(n, n, {k:sqrt(v) for k,v in matD._smat.items()})
            result = matL * matDsqr
            
        #  Make the resulting matrix as small as possible by eliminating null columns
        return result.removeNullCols()
    
    def _partition(self, llist, llen):
        # partition llist into sublist of length len
        res = []
        llistcp = cp.deepcopy(llist)
        while len(llistcp) >= llen:
            res.append(llistcp[:llen])
            llistcp = llistcp[llen:]
        return res

    def _inverseFlatten(self, flattenedList, dims):
        lbd = lambda x, y: self._partition(x, y)
        return reduce(lbd, [flattenedList] + dims[::-1][:-1])

    def _position_in_array(self, target, elem):
        # returns the positions (x,y) of elem in target assume only one occurence
        # e.g. {{1,2,3},{3,4}} -> position_in_array(1) -> (0,0)
        pos = []
        for iel, el in enumerate(target):
            if elem in el:
                pos.append([iel, el.index(elem)])
                break
        return pos

    def _rotateleft(self, llist, n, numpy=False):
        if not (numpy):
            return llist[n:] + llist[:n]
        else:
            return np.append(llist[n:], llist[:n])

    def _issorted(self, llist):
        # returns wether a list is sorted
        return all([llist[i + 1] is None or llist[i] <= llist[i + 1] for i in range(len(llist) - 1)])

    def _tuples(self, llist, n):
        """
        returns all the possible tuples of length n from elementes of llist
        """
        return list(itertools.product(llist, repeat=n))

    def _tuplesList(self, llist):
        return itertools.product(*llist)

    def _tuplesWithMultiplicity(self, listoflists):
        aux1 = list(self._tuplesList(listoflists))
        aux2 = [reduce(operator.mul, [ell[1] for ell in el]) for el in aux1]
        aux1 = [[ell[0] for ell in el] for el in aux1]
        res = list(zip(aux1, aux2))
        return res

    def _yieldParts(self, num, lt):
        if not num:
            yield ()
        for i in range(min(num, lt), 0, -1):
            for parts in self._yieldParts(num - i, i):
                yield (i,) + parts

    def _partitionInteger(self, num):
        # returns all the partition of num
        for part in self._yieldParts(num, num):
            yield part

    def tally(self, llist):
        tally, mul = [], []
        for el in llist:
            if not (el in tally):
                mul.append(llist.count(el))
                tally.append(el)
        return list(zip(tally, mul))

    def _tallyWithMultiplicity(self, listoflists):
        aux1 = self._gatherAux(listoflists)
        aux2 = [sum([ell[1] for ell in el]) for el in aux1]
        aux1 = [el[0][0] for el in aux1]
        result = zip(aux1, aux2)
        return result

    def _gatherAux(self, llist):
        gather = []
        gathered = []
        for el in llist:
            if el[0] in gathered:
                iel = gathered.index(el[0])
                gather[iel].append(el)
            else:
                gather.append([el])
                gathered.append(el[0])
        return gather

    
    def _findNullSpace(self, matrixIn, dt):
        try:
            self.prof.disable()
        except:
            pass
        t0 = time.time()
        ret = self._findNullSpaceIn(matrixIn, dt)
        print(f'\t NulSp in {time.time()-t0:.3f} s.')
        
        try:
            self.prof.enable()
        except:
            pass
        
        
        # m = sMat(matrixIn.
        
        # sep = matrixIn.shape[1] // 2
        # a = matrixIn[:,:sep]
        # b = matrixIn[:,sep:]
        
        
        # t0 = time.time()
        # ra = a.nullSpace2()
        # ta = time.time()-t0
        # rb = b.nullSpace2()
        # tb = time.time()-ta
        
        # print(f"\t  2 steps in {ta:.3f}+{tb:.3f} = {ta+tb:.3f}")
        
        # t0 = time.time()
        # sy = matrixIn.nullspace()
        # print("Sympy's Nullspace in ", time.time()-t0, "seconds")
        
        # print("\tEqual ? ", ret == sy)
        
        return ret
        
        
    def _findNullSpaceIn(self, matrixIn, dt):
        """
        This is the aux function to determine the invariants.
        """
        self.M = matrixIn
        
        # return matrixIn.nullspace()
        # return matrixIn.nullSpace()
        return matrixIn.nullSpace2()
    
        #ADD LOHAN 18.04
        def linearIndependence(term, vv, vSolved):
            mVv = sMat([[v.subs( dict([(ell, 1 if el==ell else 0) for ell in vSolved])  ) for el in vSolved] for v in (vv + [term])])
            return mVv.shape[0] - mVv.rank() == 0
        
        # t0 = time.time()
        
        # # TODO for some reason when we split in several bits res we find two vectors instead of one that need to be summed up...
        # sh = matrixIn.shape
        # matrixInlist = matrixIn.row_list()
        # aux1, gather = [], {}
        # for iel, el in enumerate(matrixInlist):
        #     if el[0] in gather:
        #         gather[el[0]].append(el)
        #     else:
        #         gather[el[0]] = [el]
        # aux1 = sorted(gather.values())
        
        # print(aux1)
        
        # if len(aux1) == 0:
        #     # MODIF LOHAN 16.04.2019
        #     return [eye(sh[1])]
        
        # preferredOrder = flatten(
        #     [[iel for iel, el in enumerate(aux1) if len(el) == i] for i in range(1, max(map(len, aux1)) + 1)])
        # matrix = {}
        # for iel, el in enumerate(preferredOrder):
        #     for ell in aux1[el]:
        #         matrix[(iel, ell[1])] = ell[2]
        
        # nElDic = {}
        
        

        # matrix = sMat(iel + 1, sh[1], matrix)  # the number of columns is kept fix
        
        # self.M2 = matrix
        # return
        matrix = matrixIn.sortRows()
        n, n2 = matrix.shape
        
        # tt = time.time()
        
        # print("Original matrix :")
        # print(matrixIn)
        
        # print('Reordered : ')
        # print(matrix)
        
        varnames = symbols('v0:{}'.format(n2))
        varSol = sMat(varnames)
#        t1 = time.time()
        
        # print(matrix)
        # print(matrix._smat)
        # print("\tPreamble : ", time.time()-t0)
        dt=1
        for i in range(0, n, dt):
            #  To determine the replacement rules we need to create a system of linear equations
            # print(i+1, "/", n)
            
            sys = (matrix[i:min(i + dt, n), :]*varSol)[0,0]#[0,0]
            # print(sys)
            # sys = expand(Matrix(matrix[i - 1:min(i + dt - 1, n), :])*varSol)[0,0]
            res = solve(sys, dict=True, simplify=False, check=False)
            
            # print('\tSolve done.')
            # print(res)
            # print(linsolve(sys.append(sMat(dt, 1))))
            
            if res:
                # print()
                # print(sys)
                # print(res)
                # varSol = varSol.subs(res[0])
                
                for k,v in varSol._smat.items():
                    # print('\t', v.free_symbols)
                    
                    intersect = v.free_symbols.intersection(res[0])
                    
                    if intersect != set():
                        varSol._smat[k] = expand(v.subs(res[0]))
                        # newVal = v
                        # for sub in intersect:
                        #     newVal = newVal.subs(sub, res[0][sub])
                        
                        # varSol._smat[k] = expand(newVal)
                    # if v.free_symbols.intersection(res[0]) != set():
                    #     varSol._smat[k] = expand(v.subs(res[0]))
                        
                # print('\n', varSol)
                # print(res)
                
#        t2 = time.time()
#        print("Time to solve system : " + str(t2-t1))
#        print("\n---\n")
        
        # now we need to extract the vector again
        tally = []
        for el in list(varSol):
            tp = [ell for ell in varnames if list(el.find(ell)) != []]
            tally.append(tp)
        
        vSolved = list(set(flatten(tally)))
        
        if vSolved == []:
            return []
        
        vv = []
        
        for el in varSol :
            if el != 0 and (len(vv) == 0 or (el not in vv and linearIndependence(el, vv, vSolved))):
                vv.append(el)
#                print(el)
            if len(vv) >= len(vSolved):
                break
        
        us = sMat(symbols('u0:{}'.format(len(vv))))
        varSol2 = sMat([simplify(el) for el in varSol.subs(solve(us - Matrix(vv), vSolved, dict = True)[0])])

        res = []
        for el in us:
            tp = cp.deepcopy(varSol2)
            for ell in us:
                if ell == el:
                    tp = tp.subs(ell, 1)
                else:
                    tp = tp.subs(ell, 0)
            res.append(tp)
        return res


class sRow():
    def __init__(self, length, dic={}, zeroCols=[]):
        self.len = length
        self.zeroCols = zeroCols
        self.dic = {}
        self.zero = True
        self.minPos = None # Pos of first non-zero element
        self.minVal = None # Val of first non-zero element
        
        for k,v in dic.items():
            if k in zeroCols:
                continue
            if self.minPos is None or k < self.minPos:
                self.minPos = k
                self.minVal = v
            self.dic[k] = v
            
        if self.minPos is not None:
            self.zero = False
            
    def isNull(self):
        return self.zero
    
    def copy(self):
        return cp.copy(self)

    def __mul__(self, x):
        r = self.copy()
        
        if x == 0:
            r.dic = {}
            r.minPos = None
            r.minVal = None
            r.zero = True
        elif x == 1:
            return r
        else:
            r.dic = {k:v*x for k,v in self.dic.items()}
            r.minVal *= x
        
        return r
    
    def __rmul__(self, x):
        return self*x
    
    def __truediv__(self, x):
        r = self.copy()
        
        if x == 1:
            return r
        else:
            r.dic = {k:v/x for k,v in self.dic.items()}
            r.minVal /= x
        
        return r

    def __add__(self, r2):
        newDic = {}
        
        keys = set(self.dic).union(r2.dic)
        
        for k in keys:
            val = 0
            if k in self.dic:
                val += self.dic[k]
            if k in r2.dic:
                val += r2.dic[k]
                
            if val != 0:
                newDic[k] = val
        
        return sRow(self.len, newDic, zeroCols=self.zeroCols)
    
    def __sub__(self, r2):
        newDic = {}
        
        keys = set(self.dic).union(r2.dic)
        
        for k in keys:
            val = 0
            if k in self.dic:
                val += self.dic[k]
            if k in r2.dic:
                val -= r2.dic[k]
                
            if val != 0:
                newDic[k] = val

        return sRow(self.len, newDic, zeroCols=self.zeroCols)
        
    def __repr__(self):
        return str(sMat(1, self.len, {(0, k):v for k,v in self.dic.items()}))









class sMat(SparseMatrix):
    """ This is an improved version of sympy's SparseMatrix """

    def __new__(self, *args, **kwargs):
        """ sMat(i,j) returns an empty i x j matrix (i.e. SparseMatrix(i,j,{})"""
        if len(args) == 2:
            return SparseMatrix.__new__(self, *args, {})
        return SparseMatrix.__new__(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.rowDic = {}
        if 'rowDic' in kwargs:
            self.computeRowDic = kwargs['rowDic']
        else:
            self.computeRowDic = False
            
        if self.computeRowDic:
            self.sh = list(self.shape)
            for k,v in self._smat.items():
                if k[0] not in self.rowDic:
                    self.rowDic[k[0]] = {}
                self.rowDic[k[0]][k[1]] = v
        
    def id(n, v=1, rowDic=False):
        """ Static function returning identity (times v) """
        
        return sMat(n, n, {(i,i):v for i in range(n)}, rowDic=rowDic)
        
    def removeNullRows(self):
        rows = {}
        smat = {}
        
        for k,v in self._smat.items():
            if k[0] not in rows:
                rows[k[0]] = len(rows)
            smat[rows[k[0]], k[1]] = v
            
        return sMat(len(rows), self.shape[1], smat)
    
    def empty(self):
        return self._smat == {}
    
    def symmetric(self):
        for k,v in self._smat.items():
            if k[0] >= k[1]:
                continue
            if self[k] != self[k[::-1]]:
                return False
        return True
    
    def removeNullCols(self):
        cols = {}
        smat = {}
        
        for k,v in self._smat.items():
            if k[1] not in cols:
                cols[k[1]] = len(cols)
            smat[k[0], cols[k[1]]] = v
            
        return sMat(self.shape[0], len(cols), smat)
            
    def append(self, m2, axis=0):
        if axis == 0:
            if self.cols != m2.cols:
                print("Shapes do not match")
                return
            
            m = sMat(self.rows+m2.rows, self.cols, self._smat)
            
            for k,v in m2._smat.items():
                m[k[0]+self.rows, k[1]] = v
            
        elif axis == 1:
            if self.rows != m2.rows:
                print("Shapes do not match")
                return
            
            m = sMat(self.rows, self.cols+m2.cols, self._smat)
            
            for k,v in m2._smat.items():
                m[k[0], k[1]+self.cols] = v
        else:
            print(f"Axis should be 0 or 1, not {axis}")
            return
        
        return m
    
    def prepend(self, m2, axis=0):
        if axis == 0:
            if self.cols != m2.cols:
                print("Shapes do not match")
                return
            
            m = sMat(self.rows+m2.rows, self.cols)
            
            for k,v in self._smat.items():
                m[k[0]+m2.rows, k[1]] = v
            
            for k,v in m2._smat.items():
                m[k[0], k[1]] = v
            
        elif axis == 1:
            if self.rows != m2.rows:
                print("Shapes do not match")
                return
            
            m = sMat(self.rows, self.cols+m2.cols)
            
            for k,v in self._smat.items():
                m[k[0], k[1]+m2.cols] = v
            
            for k,v in m2._smat.items():
                m[k[0], k[1]] = v
        else:
            print(f"Axis should be 0 or 1, not {axis}")
            return
        
        return m
    
        
    def pad(self, pad_width=((0,0), (0,0))):
        w = pad_width
        if not (type(w) == tuple and len(w) == 2 and type(w[0]) == tuple and type(w[1]) == tuple):
            print("'pad_width' should be of the form (tuple1, tuple2)")
            return
        
        m = sMat(self.shape[0] + sum(w[0]), self.shape[1] + sum(w[1]), {})
        
        for k,v in self._smat.items():
            newK = (k[0] + w[0][0], k[1] + w[1][0])
            m[newK[0], newK[1]] = v

        return m
    
    def multiply(self, m2):
        """ Element-wise matrix multiplication """
        
        if self.shape != m2.shape:
            print("Shapes do not match")
            return
        
        m = sMat(*self.shape)
        
        for k,v in self._smat.items():
            if k in m2._smat:
                m[k] = v*m2._smat[k]
        
        return m
    
    def kroneckerProduct(self, m2, simplify=False):
        m = sMat(self.shape[0]*m2.shape[0], self.shape[1]*m2.shape[1])
        
        for k1, v1 in self._smat.items():
            for k2, v2 in m2._smat.items():
                k = (k1[0]*m2.shape[0]+k2[0], k1[1]*m2.shape[1]+k2[1])
                m._smat[k] = v1*v2
                if simplify and isinstance(m._smat[k], Mul) or isinstance(m._smat[k], Add):
                    m._smat[k] = sSimplify(m._smat[k])
        
        return m
    
    # This function will be useful in PyR@TE when computing the repMats
    # of complex scalars
    def complexToReal(self):
        R = sMat([[1, I],[-I, 1]])
        
        m = sMat(self.shape[0]*R.shape[0], self.shape[1]*R.shape[1])
        
        for k1, v1 in R._smat.items():
            for k2, v2 in self._smat.items():
                k = (k1[0]*self.shape[0]+k2[0], k1[1]*self.shape[1]+k2[1])
                m._smat[k] = I*im(v1*v2)
        
        return m
    
    # def sRowList(self):
    #     """ Returns a list of (r, len) where r are the rows
    #     and len the number of non-zero elements in the row.
    #     This is a generator """
        
    #     rowDic = {}
        
        # for k,v in self._smat.items():
        #     if k[0] not in rowDic:
        #         rowDic[k[0]] = {}
        #     rowDic[k[0]][k[1]] = v
        
        # for 
        # row = [[0]*self.shape[1], 0]
        
    def vecRow(self, row):
        return sMat(1, self.shape[1], {(0, k):v for k,v in row.items()})
    
    
    
    
    
    
    def eliminateRow(self, rowDic, padDic, row, zeroCols = set()):
        currentRow = sRow(self.shape[1], row, zeroCols=zeroCols)
        
        if currentRow.isNull():
            return True
        
        # print("Current row : ", currentRow)
        
        # print("Pad dic : ", padDic)
        
        # print("SortedRows : ", sortedRows)
        
        stop = False
        while not stop:
            # print(currentRow.minPos, currentRow.isNull())
            currentPad = currentRow.minPos
            # print('\n\n')
            # print('current : ', currentRow)
            if currentRow.minVal != 1:
                currentRow = currentRow / currentRow.minVal
                
            # print('normalized : ', currentRow)
            
            # print(currentRow.minPos)
                
            # if currentPad not in padDic:
            #     rowDic[r] = currentRow.dic
            #     padDic[currentRow.minPos] = r
            #     return
                # print(currentRow.minPos, currentPad)
            
            # print(rowDic)
            if currentPad in padDic:
                otherRow = sRow(self.shape[1], rowDic[padDic[currentPad]])
                # print('other : ', otherRow)
            else:
                return currentRow
            
            currentRow = currentRow - otherRow
            
            # print('diff : ', currentRow)
            
            if currentRow.isNull():
                return True
            
            if len(currentRow.dic) <= 1:
                return currentRow.minPos
        
        print('\n### STOP HERE ###')

        
        
        
    def sortRows(self, debug=False):
        t0 = time.time()
        rowDic = {}
        
        zeroRows = {}
        
        for k,v in self._smat.items():
            if k[0] not in rowDic:
                rowDic[k[0]] = {}
                zeroRows[k[0]] = k[1]
            elif k[0] in zeroRows:
                del zeroRows[k[0]]
            
            rowDic[k[0]][k[1]] = v
        
        # Identify rows with only 1 non-zero element
        # + keep only one per column
        oneValRows = set()
        zeroCols = set()
        for k,v in zeroRows.items():
            if v not in zeroCols:
                oneValRows.add(k)
                zeroCols.add(v)
            else:
                del rowDic[k]
        
        newZeros = True
        while newZeros:
            padDic = {}
            newZeros = False
            for k in list(rowDic.keys()):
                r = cp.copy(rowDic[k])
                
                if len(r) == 1 and list(rowDic.values())[0] != 1:
                    rowDic[k] = {c:1 for c in r}
                if len(r) > 1:
                    cols = set(r).difference(zeroCols)
                    
                    if len(cols) == 0:
                        continue
                    
                    minCol = min(cols)
                    if len(cols) == 1:
                        # print(r)
                        if minCol not in zeroCols:
                            # print('\tNEW ZERO ROW : ', r, oneCol)
                            newZeros = True
                            rowDic[k] = {minCol: 1}
                            zeroCols.add(minCol)
                            oneValRows.add(k)
                        else:
                            del rowDic[k]
                    else:
                        rowDic[k] = {c:r[c]/r[minCol] for c in cols}
                        
                        # print(rowDic[k])
                        if minCol not in padDic:
                            padDic[minCol] = []
                        
                        padDic[minCol].append(k)
        
        
        orderedRows = {}
        # missingPads = True
        # while missingPads:
        #     print("Enter missingWhile")
        #     if orderedRows != {}:
        #         save = cp.copy(orderedRows)
        #         orderedRows = {}
        #     missingPads = False
            
        for col in reversed(range(self.shape[1]-1)):
            if (self.shape[1] - col)%50 == 0:
                print('\r' + f"{(self.shape[1] - col)/self.shape[1]*100:2.2f}%", end='\r')
                # time.sleep(.2)
            if col in padDic:
                if not isinstance(padDic[col], list):
                    padDic[col] = [padDic[col]]
                # print(f"Pad {col} : ok :", padDic[col])
                if len(padDic[col]) == 1:
                    padDic[col] = padDic[col][0]
                elif len(padDic[col]) > 1:
                    # First row is the one with the least non-zero cols
                    
                    padList = sorted(padDic[col], key=lambda r: len(rowDic[r]))
                    
                    padDic[col] = padList[0]
                    
                    # Rows need to be reduced
                    for i, r in enumerate(padList[1:]):
                        row = {c:v for c,v in rowDic[r].items() if c not in zeroCols}
                        elim = self.eliminateRow(rowDic, padDic, row)
                        
                        if elim is True:
                            # print('\n Deleteing row ', r, '\n')
                            del rowDic[r]
                        else:
                            # print("## #ELIM# ? ##")
                            # print('\t', r, elim)
                            
                            if isinstance(elim, sRow):
                                rowDic[r] = elim.dic
                                padDic[elim.minPos] = r
                                # missingPads = True
                                orderedRows[elim.minPos] = r
                            else:
                                print("WHAT IS ELIM HERE ? ")
                                
                                # Getting here with invariants([[1,2],[2,1],[1,1],[1,1]])
                                # of SU3
                    
                    
                # orderedRows.append(padDic[col])
                orderedRows[col] = padDic[col]
            elif col in zeroCols:
                pass
            else:
                pass
                    # print(f"Pad {col} is missing")
        print('\r', end='')
        # print(len(orderedRows), len(save))
        
        # print(orderedRows)
        # print(save)
        
        m = sMat(len(zeroCols)+len(orderedRows), self.shape[1])
        
        sortedRows = {}
        for i,c in enumerate(sorted(zeroCols)):
            sortedRows[len(sortedRows)] = [c]
            m[i, c] = 1
        
        for i, k in enumerate(sorted(orderedRows.keys(), reverse=True)):
            r = orderedRows[k]
            # print(len(sortedRows), ':', r, ' -> ', rowDic[r])
            for c,v in rowDic[r].items():
                m[len(zeroCols)+i, c] = v
            sortedRows[len(sortedRows)] = sorted(rowDic[r].keys())
        
        m.rowDic = sortedRows
        
        
        # print(f"Sorted {self.shape[0]}x{self.shape[1]} matrix in {time.time()-t0:.3f} s.")
        return m
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        # print(sorted(padDic.keys()))
        # print(sorted(padDic[-1]))
        # exit()
        previousPad = 1
        dPad = 1
        missingPad = {}
        for l in sorted(padDic.keys()):
            dPad = l-previousPad
            # print(sortedPos, l, previousPad, padDic[l], 'dPad :', dPad)
            # print('\t', missingPad)
            if dPad > 1 and previousPad != -1:
                for d in range(dPad-1):
                    missingPad[previousPad+d+1] = sortedPos
                    sortedPos += 1
                # print(missingPad)
            previousPad = l
            if l == -1:
                for i in padDic[l]:
                    oneValRows.add(sortedPos)
                    sortedRows[sortedPos] = {k:1 for k in rowDic[i]}
                    sortedPos += 1
            else:
                if len(padDic[l]) == 1:
                    factor = rowDic[padDic[l][0]][self.shape[1]-l]
                    sortedRows[sortedPos] = {k:v/factor for k,v in rowDic[padDic[l][0]].items() if k not in zeroCols}
                    padDic[l] = sortedPos
                    sortedPos += 1
                else:
                    sortedPadList = sorted(padDic[l], key=lambda x: len(rowDic[x]))
                    for pos, i in enumerate(sortedPadList):
                        factor = rowDic[i][self.shape[1]-l]
                        newRowDic =  {k:v/factor for k,v in rowDic[i].items() if k not in zeroCols}
                        if pos == 0:
                            sortedRows[sortedPos] = newRowDic
                            padDic[l] = sortedPos
                            # newPadList.append(sortedPos)
                            sortedPos += 1
                        else:
                            print(newRowDic)
                            elim = self.eliminateRow(sortedRows, padDic, newRowDic, l, debug=debug)
                            
                            if elim is True:
                                # print('\t', padDic[l])
                                # sortedPos += 1
                                pass
                            else:
                                if isinstance(elim, int):
                                    # After linear combinations this row contains only 1 term
                                    oneValRows.add(sortedPos)
                                    zeroCols.add(elim)
                                        
                                    sortedRows[sortedPos] = {elim:1}
                                    sortedPos += 1
                                else:
                                # print("What to do here ???")
                                    print(elim.minPos, '->', self.shape[1]-elim.minPos)
                                    # print(sortedRows)
                                    
                                    mPad = self.shape[1] - elim.minPos
                                    # A row must be inserted
                                    
                                    if mPad in missingPad:
                                        sortedRows[missingPad[mPad]] = elim.dic
                                        padDic[mPad] = missingPad[mPad]
                                        del missingPad[mPad]
                                    
                                # padDic[elim.minPos] = elim.
                                # return
                            
            # # for i in padDic[l]:
            #     if l == -1:
            #         oneValRows.append(len(sortedRows))
            #     sortedRows[len(sortedRows)] = rowDic[i]
                
                
                # self.eliminateRow(r, pad, padDic, debug=debug)
            # for i in sorted(lens[l], key=lambda t: t[1]):
            #     sortedRows[len(sortedRows)] = rowDic[i[0]]
            
            
            
            
        # Test with lengths
            
        # print(rowDic)
        
        # lens = {}
        # for i, r in rowDic.items():
        #     l = len(r)
        #     pad = self.shape[1] - min(r.keys())
        #     if l not in lens:
        #         lens[l] = []
            
        #     lens[l].append((i, pad))
        
        # # print()
        # # print(lens)
        
        # sortedRows = {}
        
        # for l in sorted(lens.keys()):
        # # for l in lens.keys():
        #     for i in sorted(lens[l], key=lambda t: t[1]):
        #         sortedRows[len(sortedRows)] = rowDic[i[0]]
                
        # # print()
        # # print(sortedRows)
            
            
        # print("oneValRows : ", oneValRows)
        # print("zeroCols : ", zeroCols)
        # print("Len Sortedrows :", len(sortedRows))
        
        m = sMat(len(sortedRows), self.shape[1])
        for r, dic in sortedRows.items():
            # print(r, dic)
            colList = []
            # minCol = min(dic)
            for c, v in dic.items():
                if c not in zeroCols or r in oneValRows:
                    m[r, c] = v #/ dic[minCol]
                    colList.append(c)
            m.rowDic[r] = sorted(colList)
            # print('\n', r, m.rowDic[r])
        
        
            
        
        
        print(f"Sorted {self.shape[0]}x{self.shape[1]} matrix in {time.time()-t0:.3f} s.")
        return m
    
    def nullSpace(self):
        t0 = time.time()
        sMatrix = self.sortRows()
        
        # print(sMatrix.rowDic)
        # print('\n\n')
        
        # pprint(sMatrix)
        
        # print(sMatrix.nullspace())
        # print('\n\n')
        
        sh = sMatrix.shape
        nsDim = sh[1] - sh[0]  # NullSpace dimension / matrix rank
        
        # print("nsDim : ", nsDim)
        # if nsDim <= 0:
        #     # pprint(sMatrix)
        #     # print(sMatrix.nullspace())
        #     # print('\n\n')
        #     t = time.time()
        #     print("Rank ! ", sMatrix.rank())
        #     print(".. in ", time.time()-t, "seconds.")
        #     return []
        
        solDic = {k:{} for k in range(sh[1])}
        solVecs = [sMat(sh[1], 1, {(sh[0]+i, 0): 1}) for i in range(nsDim)]
        # solMat = sMat(sh[1], min(sh[0], sh[1]))
        # pprint(solVecs)
        
        # print("SOLVECS : ", solVecs)
        
        # print(solDic)
        
        for r, colsList in sMatrix.rowDic.items():
            # print(r, colsList)
            if len(colsList) == 1 or all([solDic[c]==0 for c in colsList[1:]]):
                solDic[colsList[0]] = 0
                # print("\tsolDic[" + str(colsList[0]) + " = 0")
                continue
            for c in colsList[1:]:
                # print('\t', c)
                if solDic[c] == {}:
                    solDic[colsList[0]][c] = -sMatrix[r,c]
                elif solDic[c] == 0:
                    # print('\t\t', solDic[colsList[0]])
                    continue
                else:
                    # if len(solDic[c]) == nsDim:
                    # print('\t', solDic[c])
                    for k,v in solDic[c].items():
                        if k not in solDic[colsList[0]]:
                            solDic[colsList[0]][k] = -sMatrix[r,c]*v
                        else:
                            solDic[colsList[0]][k] -= sMatrix[r,c]*v
                    # else:
                    #     print('ERROR HERE')
                    #     print(f'row {r} col {c}')
                    #     print(solDic[c])
                    #     return
                # print('\t\t', solDic[colsList[0]])
                # print(solDic)
                # pprint(sMatrix)
                # print()
        
        # print(solDic)
        
        # pprint(solMat)
        # print(solDic)
        for row, dic in solDic.items():
            if dic == 0:
                continue
            for col, v in dic.items():
                # print(col)
                solVecs[col-sh[0]][row] = v
                
        # pprint(solVecs)
                
        print(f"Old NS in {time.time()-t0:.3f} s.")
        
        
        print(solVecs)
        return solVecs
        # v = 
        
    def __getitem__(self, key):
        """ Mimic numpy's ndarray.__getitem__ behavior """
        if isinstance(key, slice):
            return self[key, :]
        return super().__getitem__(key)
    
    
    
    def nsSplit(self):
        # t0 = time.time()
        rowDic = {}
        minValDic = {}
        
        # print(self)
        for k,v in self._smat.items():
            if k[0] not in rowDic:
                rowDic[k[0]] = {}
                minValDic[k[0]] = self.shape[1]
                
            rowDic[k[0]][k[1]] = v
            if k[1] < minValDic[k[0]]:
                minValDic[k[0]] = k[1]
        
        # ascending nb of elements
        preferedOrder = sorted(rowDic.keys(), key=lambda k: minValDic[k])
        
        upperTri = True
        newMat = sMat(*self.shape)
        for i, r in enumerate(preferedOrder):
            for c,v in rowDic[r].items():
                newMat[i, c] = v
                
                if i > c:
                    print(i,c)
                    upperTri = False
        
        # if upperTri == False:
        #     exit()
                
        return newMat
    
    def nullSpace2(self, u=0.01, vecForm=False):
        # t0 = time.time()
        rowDic = {}
        
        def result(H):
            if not vecForm:
                return H.rowDic
            
            # counts = {}
            # ret = sMat(len(H.rowDic), self.shape[1])
            # for i,dic in enumerate(H.rowDic.values()):
            #     counts[i] = {}
            #     for k,v in dic.items():
            #         ret[i,k] = v
            #         if v not in counts[i]:
            #             counts[i][v] = (1
                    
            # ret = ret.rref()[0]
            # pprint(ret)
            
            # return [sMat(self.shape[1], 1, {(k[1],0):v for k,v in ret.row(i)._smat.items()}) for i in range(ret.rows)]
            return [sMat(self.shape[1], 1, {(k,0):v for k,v in dic.items()}) for r, dic in H.rowDic.items()]
        
            # return [sMat(self.shape[1], 1, {(self.shape[1]-1-k,0):v for k,v in dic.items()}) for r, dic in H.rowDic.items()]
            
        # print(self)
        
        ## TEST FOR MATHEMATICA "COMPATIBILITY"
        ## mirror of matrix along axis x
        # for k,v in self._smat.items():
        #     if k[0] not in rowDic:
        #         rowDic[k[0]] = {}
        #     rowDic[k[0]][self.shape[1]-1 - k[1]] = v
            
        for k,v in self._smat.items():
            if k[0] not in rowDic:
                rowDic[k[0]] = {}
            rowDic[k[0]][k[1]] = v
        
        # ascending nb of elements
        preferedOrder = sorted(rowDic.keys(), key=lambda k: len(rowDic[k]))
        
        m,n = max(len(preferedOrder), 1), self.shape[1]
        
        # print(n)
        H = sMat.id(n, rowDic=True)
        
        # print(H)
        
        # pprint(self)
        s = 0
        
        r = n
        i = 0
        
        # print("SHAPE : ", self.shape)
        
        stop = False
        while not stop:
            # Progress of the computation 
            if (n-r)%100 == 0 and (n-r) != 0:
                print(n-r, '/', n)

            empty = True
            while empty:
                s = sMat(H.shape[0], 1, {})
                if i < len(preferedOrder):
                    row = rowDic[preferedOrder[i]]
                    a = sMat(n, 1, {(k,0):v  for k,v in row.items()})
                    ##### s = H*a
                    
                    for k,v in H.rowDic.items():
                        # for rA, vA in row.items():
                        keySet = set(v.keys()).intersection(row.keys())
                        if keySet != set():
                            s[(k,0)] = 0
                            
                        for key in keySet:
                            s[(k,0)] += v[key]*row[key]
                    
                empty = s.empty()
                
                if empty:
                    i += 1
                # print('\t :::', i)
                if i >= m:#-1:
                    # print(f"New NS in {time.time()-t0:.3f} s.")
                    return result(H)
            
            if True:#len(s._smat) == 1:
                key = list(s._smat.keys())[0]
                j, sj = key[0], s._smat[key]
            else:
                maxAbs = 0
                for k,v in s._smat.items():
                    # print(k)
                    if k[0] > n-i:
                        continue
                    a = abs(v)
                    if a > maxAbs:
                        maxAbs = a
                
                # print(maxAbs)
                for k,v in s._smat.items():
                    if abs(v) > u*maxAbs:
                        j, sj = k[0], v
                        break
            
            
            H = self.Gmul(H, n, i, j, s, sj)
            
            if i >= m-1:
                # print(f"New NS in {time.time()-t0:.3f} s.")
                return result(H)
            else:
                r -= 1
                i += 1
        
    def Gmul(self, H, n, i, j, s, sj):
        retH = sMat(H.shape[0]-1, H.shape[1])#, {(r,c):v for r,d in H.rowDic.items() for c,v in d.items()})
        retH.computeRowDic = True
        retH.rowDic = {(k if k<j else k-1):d for k,d in H.rowDic.items() if k!=j}
        
        # print(H.rowDic)
        # print(retH.rowDic)
        
        # print(j)
        
        if j in H.rowDic and H.rowDic[j] != {}:
            for k,v in s._smat.items():
                if k[0] == j:
                    continue
                
                val = -v/sj
                if k[0] < j:
                    # sDic[k[0]] = -v/sj
                    row = k[0]
                elif k[0] > j:
                    # sDic[k[0]-1] = -v/sj
                    row = k[0]-1
            
                # print(val)
                for col, hv in H.rowDic[j].items():
                    if row not in retH.rowDic:
                        retH.rowDic[row] = {}
                    if col not in retH.rowDic[row]:
                        retH.rowDic[row][col] = 0
                    
                    retH.rowDic[row][col] += val*hv
                    # print(retH.rowDic[row][col])
                    # if retH.rowDic[row][col] == 0:
                    #     del retH.rowDic[row][col]
                    # if retH.rowDic[row] == {}:
                    #     del retH.rowDic[row]
        
        
        # retH = sMat(*H.sh, {(r,c):v for r,d in H.rowDic.items() for c,v in d.items()})
        # retH.computeRowDic = True
        # retH.rowDic = H.rowDic
        
        # print("\n\n\n\n\n\n\n\n old method : \n")
        # pprint(tmpG*H)
        
        
        # retH._smat = {(r,c):v for r,d in retH.rowDic.items() for c,v in d.items()}
        
        # print("\n\n new method : \n")
        
        # pprint(retH)
        
        return retH
    
    # def Gmat(self, n, i, j, s, sj, r=None):
    #     if r is None:
    #         G = sMat(n-i-1, n-i)
            
    #         for k in range(n-i):
    #             # print(k)
    #             if k < j:
    #                 G[k,k] = 1
    #             elif k > j:
    #                 G[k-1,k] = 1
    #             elif k == j:
    #                 # print(s[:n-i-1])
    #                 # pprint(G)
    #                 # for r, v in s[:n-i-1]._smat.items():
    #                 #     G[r[0], j] = -v/sj
    #                 # print(s[:j]._smat)
    #                 # exit()
    #                 for r, v in s[:j]._smat.items():
    #                     G[r[0], j] = -v/sj
    #                 for r, v in s[j+1:n-i]._smat.items():
    #                     G[j+r[0], j] = -v/sj
    #                 # G[:, j] = s[:n-i]/sj
    #     # else:
    #     #     G = sMat(r-1, r)
            
    #     #     for k in range(r):
    #     #         # print(k)
    #     #         if k < j:
    #     #             G[k,k] = 1
    #     #         elif k > j:
    #     #             G[k-1,k] = 1
    #     #         elif k == j:
    #     #             # print(s[:n-i-1])
    #     #             # pprint(G)
    #     #             for r, v in s[:r-1]._smat.items():
    #     #                 G[r[0], j] = -v/sj
    #     #             # G[:, j] = s[:n-i]/sj
            
    #     #     pprint(G)
    #     #     # exit()
                        
    #     print("OLD S:", G[:, j])
    #     return G
        
    def sqrt(self):
        return sMat(*self.shape, {k:sqrt(v) for k,v in self._smat.items()})
    
    def takagi(self):
        matrix = self
        n = matrix.shape[0]
        
        if n == 1:
            return matrix.sqrt()
        
        vTotal = sMat.id(n)
        diagList = []
        eigenS = (matrix*matrix.conjugate()).eigenvects()
        
        # pprint(matrix*matrix.conjugate())
        
        eigenVal = []
        for el in eigenS:
            if el[1] == 1:
                eigenVal.append(el[0])
            else:
                eigenVal += el[1]*[el[0]]
                
        # eigenVal = [el[0] for el in eigenS]
        auxVec = flatten([el[2] for el in eigenS], cls=list)
        
        eigenVec = sMat(n, n)
        for c, v in enumerate(auxVec):
            for k,val in v._smat.items():
                eigenVec[c, k[0]] = val
            
        # print("Eigenval :", eigenVal)
        # print("Eigenvec :")
        # pprint(eigenVec)
        
        for i in range(n-1):
            # print("\n\nIter ", i)
            matrix2 = matrix*matrix.conjugate()
            mu = sqrt(eigenVal[i])
            v = eigenVec[i:n, i]
            
            # print("v : ", v)
            aux = ( matrix*v.conjugate() ).transpose()
            aux = v.transpose().append(aux, axis=0).rank()
            # print("aux : ", aux)
            
            if aux == 2:
                v = matrix*v.conjugate() + mu*v
            
            vo = self.orthogonalizeFast(v).transpose()
            
            # print(vo)
            
            vCorrected = vo.pad(pad_width=((n-vo.shape[0], 0), (n-vo.shape[1], 0)))
            if i > 0:
                diag = sMat(n,n)
                for p in range(i):
                    diag[p, p] = 1
            
                # print("Diag :")
                # pprint(diag)
                
                vCorrected += diag
            
            # print("VO : ")
            # pprint(vo)
            # aux = 
                
            # print("V CORRECTED : ")
            # pprint(vCorrected)

            eigenVec = vCorrected.adjoint()*eigenVec
            
            vTotal = vTotal * vCorrected
            
            aux = vo.adjoint()*matrix*vo.conjugate()
            
            diagList.append(aux[0,0])
            matrix = aux[1:, 1:]
        
        diagList.append(aux[1,1])
        
        ret = sMat(len(diagList), len(diagList))
        for i,v in enumerate(diagList):
            ret[i,i] = sqrt(v)
            
        ret = vTotal*ret
        
        return ret
        
    def orthogonalizeFast(self, vec):
        # print('\t Vec : ', vec)
        n = vec.shape[0]
        result = sMat(n,n)
        
        pos1 = set(range(n))
        pos2 = set()
        
        for k in vec._smat:
            pos1.discard(k[0])
            pos2.add(k[0])
            
        pos1, pos2 = sorted(pos1), sorted(pos2)
        # print(pos1, pos2)
        
        if pos2 != []:
            vecList = []
            for i in range(len(pos2)):
                vecList.append(sMat(1, len(pos2), {(0,i):1}))
            
            vecList = [vec[pos2, :].transpose()] + vecList
            orth = sMat.orthogonalize(*vecList, normalize=True)
            
            # print(orth)
            for i,v in enumerate(orth):
                for j,p in enumerate(pos2):
                    result[i,p] = v[0,j]
        
        # pprint(result)
        
        if pos1 != []:
            for i, p in enumerate(pos1):
                result[i+len(pos2), p] = 1
    
        # pprint(result)
        return result
        
def sEye(n, v=1):
    return sMat.id(n, v)














from sympy import IndexedBase

class sTensor():
    symbs = [IndexedBase(name) for name in 'abcd']
    
    def __init__(self, *dims, dic=None, subTensor=False, forceSub=False):
        self.rank = len([el for el in dims if el is not None])
        self.dim = (dims if len(dims) == 4 else self.pad(dims, right=True))
        self.trueDim = tuple([d for d in self.dim if d is not None])
        self.subTensor = subTensor
        self.forceSub = forceSub
        # print("dim : ", self.dim)
        
        self.nMax = 1
        for d in self.dim:
            if d is not None:
                self.nMax *= d
        
        self.dic = dict()
        self.subDics = [dict() for _ in range(self.rank)]
        
        self.subTensors = dict()
        
        if dic is not None:
            self.dic = dic
            
            if self.rank > 1 and (self.rank < 4 or self.forceSub):
                for k,v in dic.items():
                    self.fillSubDics(self.pad(k),v)
            
        # for i, d in enumerate(self.subDics):
        #     if d != {}:
        #         for k,v in d.keys():
        #             key = i*(None,) + (k,) + (3-i)*(None,)
        #             subDim = (self.dim[:i] + self.dim[i+1:])
        #             self.subTensors[key] = sTensor(*subDim, dic=v)
                #     self.subTensor[i*(None,) + (i,) + (3-i)*(None,)] = v
        
        # for i in range(self.rank):
        #     self.subTensors[i].dic = self.subDics[i]

    def pad(self, key, right=False):
        # Test: keys are always 4-dim tuples, with None values if rank < 4
        if len(key) == 4:
            return key
        if right:
            return key + (None,) * (4 - self.rank)
        
        k = []
        count = 0
        for d in self.dim:
            if d is not None:
                k.append(key[count])
                count += 1
            else:
                k.append(None)
                
        return tuple(k)
    
    def dimPos(self, i):
        count = 0
        for k, d in enumerate(self.dim):
            if d is not None:
                if count == i:
                    return k
                count += 1
                
    def isEmpty(self):
        return len(self.dic) == 0
    
    def fillSubDics(self, k, v):
        for i, subDic in enumerate(self.subDics):
            j = self.dimPos(i)
            
            if k[j] not in subDic:
                subDic[ k[j] ] = dict()
            
            subDic[ k[j] ][ k[:j] + (None,) + k[j+1:] ] = v
            
            # Update subTensors
            key = j*(None,) + (k[j],) + (3-j)*(None,)
            if self.dim[j] is not None:
                if key not in self.subTensors:
                    subDim = list(self.dim)
                    subDim[j] = None
                    self.subTensors[key] = sTensor(*subDim, dic=subDic[k[j]])
                # else:
                    # if self.subDics[i][k[j]] != self.subTensors[key].dic:
                        # exit()
                        
    def keyToSymbol(self, key):
        tmp = 1
        for pos,ind in enumerate(key):
            if ind is not None:
                tmp *= self.symbs[pos][ind+1]
        
        return tmp
    
    def expr(self):
        expr = 0
        
        for k,v in self.dic.items():
            tmp = v
            for pos,ind in enumerate(k):
                if ind is not None:
                    tmp *= self.symbs[pos][ind+1]
            expr += tmp
        
        return expr
        
            
    def __repr__(self):
        return str(self.expr())
        return (f"Tensor of rank {self.rank} with dimensions {self.dim}.\n" +
                f"Density : {len(self.dic)/self.nMax*100:.2f}%")
    
    def __getitem__(self, key):
        # for i, el in enumerate(key):
        #     if el >= self.dim[i]:
        #         raise ValueError(f"Value {el} in {key} is larger than tensor dimensions : {self.dim}.")

        if key in self.dic:
            return self.dic[self.pad(key)]
        return 0
    
    def __setitem__(self, key, val):
        # for i, el in enumerate(key):
        #     if el is not None and el >= self.dim[i]:
        #         raise ValueError(f"Value {el} in {key} is larger than tensor dimensions : {self.dim}.")
        
        if val == 0:
            if key in self.dic:
                del self.dic[key]
            return

        self.dic[self.pad(key)] = val
        
        if self.rank < 4 or self.forceSub:
            self.fillSubDics(self.pad(key), val)
        
    def items(self):
        return self.dic.items()
    
    def sumKey(self, k1, k2):
        if len(k1) != len(k2):
            exit(f"The two keys {k1} and {k2} must have the same length")

        k = []
        for i,j in zip(k1,k2):
            if i is not None and j is not None:
                exit(f"Cannot sum the two keys {k1} and {k2}")
            
            if i is None:
                k.append(j)
            elif j is None:
                k.append(i)
            else:
                k.append(None)
                
        return tuple(k)
    
    def permute(self, perm):
        perm = tuple(perm)
        
        if len([el for el in perm if el is not None]) != self.rank:
            exit("Permutations list should equal tensor's rank")
            return
        if not all([el in perm for el in range(self.rank)]):
            exit("Information would be lost with this permutation.")
            return
        
        newDim = [None]*4
        
        for pos, i in enumerate(perm):
            if i is not None:
                newDim[pos] = self.dim[i]
        
        # print(self.dim, perm, newDim)

        ret = sTensor(*newDim)
        
        for k,v in self.dic.items():
            key = tuple([(k[el] if el is not None else None) for el in perm])
            # print(f"{k} goes to {key}")
            ret[key] = v
            
        return ret
    
    def keyContains(self, key, subKey):
        for i1, i2 in zip(key, subKey):
            if i1==i2:
                return True

        return False
    
    def sub(self, subDic, expandResult=False, vb=False):
        if subDic == {}:
            return self
        
        # Go through subDic once to see if the substitution is consistent
        # + determine the dimension of the resulting tensor
        dims = set()
        
        for subKey, subTensor in subDic.items():
            complement = [el for el in subTensor.dim]
            for pos, i in enumerate(subKey):
                if i is not None and complement[pos] is not None:
                    complement[pos] = None
                elif i is not None and complement[pos] is None:
                    exit(f"Cannot substitute {subKey} from {subTensor}")
            
            # print("Complement : ", tuple(complement))
            tmpDim = self.sumKey(self.dim, tuple(complement))
            
            # print("TMP DIM : ", tmpDim)
            
            dims.add(tmpDim)
        
        # print("All dims : ", dims)
            
        if len(dims) == 1:
            dims = list(dims)[0]
        else:
            exit(f"Substitution {subDic} in tensor {self} is inconsistent.")
        
        # ex = False
        # # if len(self.dic) == 11:
        # #     print("HEYO\\\\\\\\\\\\\\\\n\n\n")
            
        # #     # print("Original subDic : ", subDic)
        # #     # self.dic = {list(self.dic.keys())[0]: self.dic[list(self.dic.keys())[0]]}
        # #     subDic = {k : subDic[k] for k in list(subDic.keys())[:5]}
            
        # #     ex = True
            
        # print("\n\nSubstituting ", subDic)
        
        # print("\n in tensor :", self)
        
        # print("Subtensors : ", self.subTensors)
        
        
        if vb:
            # n = 100
            # c = 0
            # tmp = {}
            # for i, (k,v) in enumerate(self.dic.items()):
            #     if i >= n:
            #         break
            #     tmp[k] = v
            
            # self = sTensor(*self.dim, dic=tmp, forceSub=True)
            # print("\n\n# TENSOR BEFORE :")
            # print(self.toMathematica())
            # print('\n', self.dic)
            # print("\t CONSISTENCY CHECK of TENSOR")
            # self.checkSubdicConsistency()
            
            
            newDic = {}
            
            # print("\n SUBTENSORS :", self.subTensors)
            # subDic = {k:v for k,v in subDic.items() if k[0] == 3}
            # print("SubDic : ", subDic)
            # print("\n SUBTENSORS :", self.subTensors)
            # print()
            # print('\n')
            for subKey, subTensor in subDic.items():
                # print("Subkey , sub value :", subKey, subTensor)
                if subKey in self.subTensors.keys():
                    # print(" Subkey IS in subTensors.")
                    for k,v in subTensor.items():
                        # print("\tTaking key", k, " in subs.")
                        for K, V in self.subTensors[subKey].items():
                            # print("\t\tTaking key", K, " in subTensors.")
                            key = self.sumKey(k, K)
                            val = v*V
                            
                            # print(key, ":", val)
                            if key not in newDic:
                                # print('newKey :', key, ':', val)
                                newDic[key] = val
                            else:
                                newDic[key] += val
                            # if newTensor[key] == 0:
                            #     newTensor[key] = val
                            # else:
                            # #     # print("\n\t Already here : ", key, '=', self.keyToSymbol(key))
                            # #     # print("With value", newTensor[key])
                            # #     # print("To add is", val)
                            #     newTensor[key] = newTensor[key]+val
            
            # if expandResult:
            #     for k,v in newTensor.dic.items():
            #         newTensor[k] = expand(v)
            
            newTensor = sTensor(*dims, forceSub=self.forceSub, dic={k:v for k,v in newDic.items() if v != 0})
            # print("\t CONSISTENCY CHECK of NEW TENSOR")
            # newTensor.checkSubdicConsistency()
            
            # print("#######\n\n")
            # print("\t\t NEWTENSOR Dic :")
            # print(newTensor.dic)
            # nnewTensor = sTensor(*dims, forceSub=self.forceSub)
            # for k,v in newTensor.items():
            #     # print(k, v)
            #     nnewTensor[k] = v
                # nnewTensor.checkSubdicConsistency()
            
            # print("\t CONSISTENCY CHECK of NNNEW TENSOR")
            # nnewTensor.checkSubdicConsistency()
            
            # print("  Diferrence between two :")
            # # print(newTensor.expr() - nnewTensor.expr())
            # print("# TENSOR After :")
            # print(newTensor.toMathematica())
        else:
            
            newTensor = sTensor(*dims, forceSub=self.forceSub)
    
            for subKey, subTensor in subDic.items():
                if subKey in self.subTensors:
                    for k,v in subTensor.items():
                        for K, V in self.subTensors[subKey].items():
                            key = self.sumKey(k, K)
                            val = v*V
                            
                            if newTensor[key] == 0:
                                newTensor[key] = val
                            else:
                                # print("\n\t Already here : ", key, '=', self.keyToSymbol(key))
                                # print("With value", newTensor[key])
                                # print("To add is", val)
                                newTensor[key] += val
            
            if expand:
                for k,v in newTensor.dic.items():
                    newTensor[k] = expand(v)
        # print('\n\n')
        # print("Result is ")
        # print(newTensor)
        # print('\n\n\n')
        
        
        # print(" Now testing against sympy. Result is : \n")
        
        # ori = self.expr()
        # su = {self.keyToSymbol(k) : v.expr() for k,v in subDic.items()}
        
        # sysu = expand(ori.subs(su, simultaneous=True))
        
        # print(sysu, '\n\n')
        
        # print("Difference :\n")
        # print(sysu-newTensor.expr())
        
        
        # if ex:
        #     exit()
        return newTensor
    
    def toMathematica(self):
        # def copy2clip(txt):
        #     cmd='echo '+txt.strip()+'|clip'
        #     return subprocess.check_call(cmd, shell=True)
        s = str(self).replace('**','^')
        
        while s.find('sqrt') != -1:
            i = s.find('sqrt')
            j = s.find(')', i)
            
            sqrtS = s[i:j+1]
            
            s = s.replace(sqrtS, sqrtS.replace('sqrt', 'Sqrt').replace('(','[').replace(')',']'))
                          
        if s[0] == '[' and s[-1] == ']':
            s = '{' + s[1:-1] + '}'
            
        s = s +';\n\n'
        return s
           
    def __add__(self, other):
        """ Sum of two tensors """
        
        if other.dim != self.dim:
            print("Incompatible tensor dimensions")
            return
        
        retTensor = sTensor(*self.dim)
        retTensor.dic = {k:v for k,v in self.dic.items()}
        
        for k,v in other.items():
            if k in self.dic:
                retTensor.dic[k] += other.dic[k]
                if retTensor.dic[k] == 0:
                    del retTensor.dic[k]
            else:
                retTensor.dic[k] = other.dic[k]
                
        return retTensor
        
    
    def checkSubdicConsistency(self, keyChoose=0):
        subDic = {k:v for k,v in self.subTensors.items() if k[keyChoose] is not None}
        
        newTensor = sTensor(*self.dim)
        
        for k,v in subDic.items():
            for K,V in v.items():
                key = self.sumKey(k, K)
                
                newTensor[key] = V
                
        print("Difference ?", self.expr() - newTensor.expr())
        return
        
    def __mul__(self, other, cplx=False):
        """ Element-wise multiplication """
        
        if other.dim != self.dim:
            print("Incompatible tensor dimensions")
            return
        
        retTensor = sTensor(*self.dim)
        
        order = True
        if len(self.dic) <= len(other.dic):
            smallestDic = self.dic
            otherDic = other.dic
        else:
            smallestDic = other.dic
            otherDic = self.dic
            order = False
        
        if not cplx:
            for k,v in smallestDic.items():
                if k in otherDic:
                    retTensor.dic[k] = v*otherDic[k]
        else:
            for k,v in smallestDic.items():
                if k in otherDic:
                    if order:
                        retTensor.dic[k] = v*conjugate(otherDic[k])
                    else:
                        retTensor.dic[k] = otherDic[k]*conjugate(v)
                    
        return retTensor
        
    def __rmul__(self, x):
        """ Right multiplication by a scalar """
        
        ret = sTensor(*self.dim)
        
        if x != 0:
            for k,v in self.dic.items():
                ret.dic[k] = x*v
            
        return ret
    
    def sum(self, pow=1):
        """ Returns the sum of all elements of the tensor """
        
        s = 0
        
        for v in self.dic.values():
            if pow == 1:
                s += v
            else:
                s += v**pow
            
        return s
    