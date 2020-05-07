from sys import exit

import copy as cp
import numpy as np

import operator
import itertools
from functools import reduce

from sympy.combinatorics import Permutation
from sympy import (Add, I, Mul, SparseMatrix, Symbol, Rational,
                   conjugate, factorial, flatten, im, sqrt)

from sympy import simplify as sSimplify

class Sn:
    def __init__(self):
        # Declare a MathGroup object to access the standard methods
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
            ns = (aux[0] - Id).append(aux[1] - Id, axis=0).nullSpace(vecForm=True)

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
            exit("Error, the matD is not diagonal : cannot take the sqrt.")
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
                if simplify :
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

    def __getitem__(self, key):
        """ Mimic numpy's ndarray.__getitem__ behavior """
        if isinstance(key, slice):
            return self[key, :]
        return super().__getitem__(key)

    def nullSpace(self, u=0.01, vecForm=False, progress=False):
        rowDic = {}

        def result(H):
            if not vecForm:
                return H.rowDic
            return [sMat(self.shape[1], 1, {(k,0):v for k,v in dic.items()}) for r, dic in H.rowDic.items()]

        for k,v in self._smat.items():
            if k[0] not in rowDic:
                rowDic[k[0]] = {}
            rowDic[k[0]][k[1]] = v

        # ascending nb of elements
        preferedOrder = sorted(rowDic.keys(), key=lambda k: len(rowDic[k]))
        m,n = max(len(preferedOrder), 1), self.shape[1]

        H = sMat.id(n, rowDic=True)

        s = 0
        r = n
        i = 0

        stop = False
        while not stop:
            # Progress of the computation
            if progress and (n-r)%100 == 0 and (n-r) != 0:
                print("NullSpace : ", n-r, '/', n)

            empty = True
            while empty:
                s = sMat(H.shape[0], 1, {})
                if i < len(preferedOrder):
                    row = rowDic[preferedOrder[i]]

                    for k,v in H.rowDic.items():
                        keySet = set(v.keys()).intersection(row.keys())
                        if keySet != set():
                            s[(k,0)] = 0

                        for key in keySet:
                            s[(k,0)] += v[key]*row[key]

                empty = s.empty()
                if empty:
                    i += 1
                if i >= m:#-1:
                    return result(H)

            if True:#len(s._smat) == 1:
                key = list(s._smat.keys())[0]
                j, sj = key[0], s._smat[key]
            else:
                maxAbs = 0
                for k,v in s._smat.items():
                    if k[0] > n-i:
                        continue
                    a = abs(v)
                    if a > maxAbs:
                        maxAbs = a

                for k,v in s._smat.items():
                    if abs(v) > u*maxAbs:
                        j, sj = k[0], v
                        break

            H = self.Gmul(H, n, i, j, s, sj)

            if i >= m-1:
                return result(H)
            else:
                r -= 1
                i += 1

    def Gmul(self, H, n, i, j, s, sj):
        retH = sMat(H.shape[0]-1, H.shape[1])
        retH.computeRowDic = True
        retH.rowDic = {(k if k<j else k-1):d for k,d in H.rowDic.items() if k!=j}

        if j in H.rowDic and H.rowDic[j] != {}:
            for k,v in s._smat.items():
                if k[0] == j:
                    continue

                val = -v/sj
                if k[0] < j:
                    row = k[0]
                elif k[0] > j:
                    row = k[0]-1

                for col, hv in H.rowDic[j].items():
                    if row not in retH.rowDic:
                        retH.rowDic[row] = {}
                    if col not in retH.rowDic[row]:
                        retH.rowDic[row][col] = 0

                    retH.rowDic[row][col] += val*hv

        return retH


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

        for i in range(n-1):
            mu = sqrt(eigenVal[i])
            v = eigenVec[i:n, i]

            aux = ( matrix*v.conjugate() ).transpose()
            aux = v.transpose().append(aux, axis=0).rank()

            if aux == 2:
                v = matrix*v.conjugate() + mu*v

            vo = self.orthogonalizeFast(v).transpose()

            vCorrected = vo.pad(pad_width=((n-vo.shape[0], 0), (n-vo.shape[1], 0)))
            if i > 0:
                diag = sMat(n,n)
                for p in range(i):
                    diag[p, p] = 1

                vCorrected += diag


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
    baseSymbs = [IndexedBase(name) for name in 'abcd']

    def __init__(self, *dims, dic=None, subTensor=False, forceSub=False):
        self.rank = len([el for el in dims if el is not None])
        self.dim = (dims if len(dims) == 4 else self.pad(dims, right=True))
        self.trueDim = tuple([d for d in self.dim if d is not None])
        self.subTensor = subTensor
        self.forceSub = forceSub

        self.symbs = self.baseSymbs
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
                tmp *= self.baseSymbs[pos][ind+1]

        return tmp

    def setFields(self, fields):
        if len(fields) != self.rank:
            print(f"Error : please provide this function with exactly {self.rank} fields.")
            return

        self.symbs = [IndexedBase(el) for el in fields]

    def expr(self):
        expr = 0

        for k,v in self.dic.items():
            coeff = v
            tmp = []
            for pos,ind in enumerate(k):
                if ind is not None:
                    tmp.append(self.symbs[pos][ind+1])
            expr += Mul(coeff, *tmp)


        return expr

    def _repr_latex_(self):
        return self.expr()._repr_latex_()

    def __repr__(self):
        return str(self.expr())

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

    def sub(self, subDic, expandResult=False):
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

            tmpDim = self.sumKey(self.dim, tuple(complement))

            dims.add(tmpDim)

        if len(dims) == 1:
            dims = list(dims)[0]
        else:
            exit(f"Substitution {subDic} in tensor {self} is inconsistent.")

        newDic = {}
        for subKey, subTensor in subDic.items():
            if subKey in self.subTensors.keys():
                for k,v in subTensor.items():
                    for K, V in self.subTensors[subKey].items():
                        key = self.sumKey(k, K)
                        val = v*V

                        if key not in newDic:
                            newDic[key] = val
                        else:
                            newDic[key] += val

        return sTensor(*dims, forceSub=self.forceSub, dic={k:v for k,v in newDic.items() if v != 0})

    def toMathematica(self):
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
