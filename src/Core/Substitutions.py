# -*- coding: utf-8 -*-

from Logging import loggingCritical, loggingInfo

from sympy import (Mul, Pow, Symbol, adjoint, conjugate, diag, Matrix, sqrt, root, solveset,
                   Abs, DiagonalMatrix, diff, flatten, MatMul, Integer, transpose)

from Definitions import mSymbol, expand, Trace, trace, replaceKey, insertKey, Identity, splitPow


def getSubstitutions(self, substitutions):
    """ Several possible types of substitutions :
            - Simply rename a quantity
                e.g. : g_SU3C: g_3
            - Explicit definition of a Yukawa coupling matrix :
                e.g. : Yu: [[0, 0,   0  ],
                               [0, 0,   0  ],
                               [0, 0, 'y_t']]
            - Set a quantity to 0:
                e.g. : Ye : 0
            - Definition of new quantities based on a 1-to-1 relation:
                e.g. : alpha_1 : g_1**2 / (4 pi)

        Note that in the right-hand side, previously undefined quantities
        must appear inside ' '. Since the rhs is a string, the user must use
        double quotes to wrap the full expression in this case.
    """

    substitutionDic = {}

    #################
    # Substitutions #
    #################

    substitutionDic['zero'] = {}

    def case(k, v, case):
        if case == 'rename':
            return (' ' not in v and
                    '+' not in v and
                    ('*' not in v or '^*' in v or '^{*}' in v) and
                    '[' not in v and ']' not in v)
        elif case == 'yukMat':
            try:
                return (type(self.parseMathExpr(v)) == list)
            except BaseException as e :
                print(e)
                return False
        elif case == 'sub':
            return ((lambda x: (isinstance(x, Mul) or
                               isinstance(x, Pow) or
                               isinstance(x, Symbol)))(self.parseMathExpr(v))
                    and (' ' not in k and
                         '+' not in k and
                         ('*' not in k or '^*' in v or '^{*}' in v)))
        elif case == 'zero':
            return (v == '0')
        elif case == 'diag':
            return v.lower()[:5] == 'diag(' and v[-1] == ')'

    if substitutions != {}:
        tmpCouplings = []

        for k,v in substitutions.items():
            v = v.strip()
            v = v.replace('Sqrt', 'sqrt')
            selfSub = False

            if k in self.allCouplings:
                cType = self.allCouplings[k]
                if type(cType) == tuple:
                    cType = cType[0]

                if case(k, v, 'rename') and not case(k, v, 'zero') and not case(k, v, 'diag'):
                    if 'rename' not in substitutionDic:
                        substitutionDic['rename'] = {}
                    substitutionDic['rename'][k] = (cType, v)

                    self.allCouplings[v] = cType
                    tmpCouplings.append(v)

                    if k in self.assumptions:
                        self.assumptions[v] = self.assumptions[k]
                    continue
                elif (case(k, v, 'yukMat') or case(k, v, 'diag')) and not case(k, v, 'zero'):
                    if 'yukMat' not in substitutionDic:
                        substitutionDic['yukMat'] = {}

                    diagKW = False
                    if case(k, v, 'diag'):
                        diagKW = True
                        v = v.replace('Diag', 'diag').replace('diag', 'DiagonalMatrix')

                    if 'real' in self.assumptions[k] and self.assumptions[k]['real'] == True:
                        matList = self.parseMathExpr(v, real=True)
                    else:
                        matList = self.parseMathExpr(v)

                    if diagKW:
                        mat = matList
                    elif type(matList[0]) != list:
                        # Diagonal matrix
                        for i, el in enumerate(matList):
                            if type(el) != Symbol and type(el) != conjugate and el != 0:
                                loggingCritical(f"Error in substitution matrix {k} : entries must be either a new symbol or 0.")
                                exit()
                            if el == 0:
                                continue
                            if 'hermitian' in self.assumptions[k] and self.assumptions[k]['hermitian']:
                                matList[i] = Symbol(str(el), real=True)
                        mat = diag(*matList)
                    else:
                        for i, row in enumerate(matList):
                            for j, el in enumerate(row):
                                if type(el) != Symbol and type(el) != conjugate  and el != 0:
                                    loggingCritical(f"Error in substitution matrix {k} : entries must be either a new symbol or 0.")
                                    exit()
                                if 'hermitian' in self.assumptions[k] and self.assumptions[k]['hermitian']:
                                    if i != j and el != conjugate(matList[j][i]):
                                        loggingCritical(f"Error in Substitutions: matrix {k} is supposed to be hermitian.")
                                        exit()
                                    if i == j and el != 0:
                                        matList[i][j] = Symbol(str(el), real=True)
                                if 'symmetric' in self.assumptions[k] and self.assumptions[k]['symmetric']:
                                    if el != matList[j][i]:
                                        loggingCritical(f"Error in Substitutions: matrix {k} is supposed to be symmetric.")
                                        exit()

                        mat = Matrix(matList)

                    # Detect diagonal matrices declared without 'diag' keyword
                    if not diagKW:
                        if mat.shape[0] == mat.shape[1]:
                            if all([ mat[i,j] == 0 for i in range(mat.shape[0]) for j in range(mat.shape[1]) if j!=i]):
                                if all([ mat[i,j] == mat[0,0] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if j==i]):
                                    mat = DiagonalMatrix(mat[0,0])

                    substitutionDic['yukMat'][k] = (cType, mat)

                    for newSymb in mat.free_symbols:
                        self.allCouplings[str(newSymb)] = (cType, newSymb)

                    continue
                elif case(k, v, 'sub') and not case(k, v, 'zero'):
                    selfSub = True
                elif case(k, v, 'zero'):
                    substitutionDic['zero'][k] = cType
                    continue
                else:
                    loggingCritical(f"Warning : substitution < {k} : {v} > not understood. Skipping it.")

            if (k not in self.allCouplings and case(k, v, 'sub')) or selfSub:
                expr = self.parseMathExpr(v)

                #Identify the couplings appearing in the rhs
                coupling = [el for el in expr.atoms() if str(el) in self.allCouplings]

                if len(coupling) == 0:
                    loggingCritical(f"Warning : in substitution < {k} : {v} >, both sides involve unknown couplings. Skipping.")
                    continue
                elif len(coupling) != 1:
                    loggingCritical(f"Warning : in substitution < {k} : {v} >, only one coupling of the model should appear in the rhs (not {len(coupling)}). Skipping.")
                    continue

                coupling = coupling[0]
                symb = Symbol(k)

                cType = self.allCouplings[str(coupling)] if type(self.allCouplings[str(coupling)]) != tuple else self.allCouplings[str(coupling)][0]

                # GUT Normalization
                if coupling == symb and cType == 'GaugeCouplings':
                    if not expr.args.count(coupling) == 1:
                        loggingCritical(f"Warning : in Substitutions, GUT Normalization < {k} : {v} > is invalid. Skipping.")
                        continue

                    aux = expr/coupling
                    if aux.find(Pow) != set():
                        p = min([el.exp for el in aux.find(Pow)])
                        if p < 1:
                            aux = root(aux**(1/p), (1/p), evaluate=False)

                    self.gutNorm[coupling] = (symb, Mul(aux, coupling, evaluate=False), coupling/aux)

                    continue

                # Here we handle mainly 3 cases
                #   - A simple expression involving 'coupling'; e.g. coupling**n
                #   - A combination of 'coupling' and 'conjugate(coupling)'
                #   - 'Abs(coupling)**2'

                # If sub = x*conjugate(x) = abs(x)**2, it must be assumed real
                if expr.find(Abs) != set() or expr.subs(coupling*conjugate(coupling), 1).is_number:
                    symb = Symbol(str(symb), real=True)

                expr = expr.subs(Abs(coupling)**2, coupling*conjugate(coupling))

                if expr.find(conjugate) != set():
                    conj = Symbol('conj' + str(coupling))
                    aux = expr.subs(conjugate(coupling), conj)
                    couplings = [coupling, conj]
                else:
                    conj = Symbol('_None')
                    aux = expr
                    couplings = [coupling]

                derivatives = [(c.subs(conj, conjugate(coupling)), diff(aux, c).subs(conj, conjugate(coupling))) for c in couplings]

                if 'sub' not in substitutionDic:
                    substitutionDic['sub'] = {}
                substitutionDic['sub'][coupling] = (symb, expr, derivatives)

                self.allCouplings[k] = (cType, symb)
                tmpCouplings.append(k)

            else:
                loggingCritical(f"Warning : substitution < {k} : {v} > not understood. Skipping it.")

        for c in tmpCouplings:
            del self.allCouplings[c]

    return substitutionDic


def doSubstitutions(self, substitutionDic, inconsistentRGEerror=False):

    # Replace 'Xstar' / 'X^*' / X^{*} by 'conjugate(X)'
    for k,v in self.allCouplings.items():
        # The conjugated couplings are removed, and must be replaced with Conjugate(...)
        if k[-2:] == '^*' or k[-4:] == '^{*}' or k[-4:] == 'star':
            noStar = k.replace('^*', '').replace('^{*}', '').replace('star', '')
            if noStar in self.allCouplings:
                sub = {v[1]: conjugate(self.allCouplings[noStar][1])}
                for cType, loopDic in self.couplingRGEs.items():
                    for nLoop, RGEdic in loopDic.items():
                        for c, bFunc in RGEdic.items():
                            self.couplingRGEs[cType][nLoop][c] = bFunc.subs(sub)

    # For squared scalar mass parameters, replace mu -> mu^2 everywhere
    muSubDic = {}
    muDic = {}
    for k,v in self.allCouplings.items():
         if v[0] == 'ScalarMasses':
             if k in self.assumptions and self.assumptions[k]['squared'] is True:
                 muSubDic[v[1]] = v[1]**2
                 muDic[k] = v[1]

    if muSubDic != {}:
        for cType, loopDic in self.couplingRGEs.items():
            for nLoop, RGEdic in loopDic.items():
                for c, bFunc in RGEdic.items():
                    self.couplingRGEs[cType][nLoop][c] = bFunc.subs(muSubDic)

                    if cType == 'ScalarMasses' and c in muDic:
                        self.couplingRGEs[cType][nLoop][c] = expand(self.couplingRGEs[cType][nLoop][c]/(2*muDic[c]))

    if substitutionDic == {}:
        for k,v in list(self.NonZeroCouplingRGEs.items()):
            if all([dic == {} for dic in v.values()]):
                del self.NonZeroCouplingRGEs[k]
        return

    if 'rename' in substitutionDic and substitutionDic['rename'] != {}:
        def subRule(c, newC):
            if newC[0] == 'GaugeCouplings':
                return (Symbol(c, real=True), Symbol(newC[1], real=True))
            elif newC[0] == 'Yukawas':
                struc = self.couplingStructure[c]
                if struc[-1] != True:
                    return (mSymbol(c, *struc), mSymbol(newC[1], *struc))
                else:
                    return (mSymbol(c, *(struc[:-1]), symmetric=True), mSymbol(newC[1], *(struc[:-1]), symmetric=True))
            else:
                return (Symbol(c, complex=True), Symbol(newC[1]))


        for k,v in substitutionDic['rename'].items():
            for cType, loopDic in self.couplingRGEs.items():
                for nLoop, RGEdic in loopDic.items():
                    if cType == v[0] and k in RGEdic:
                        self.couplingRGEs[v[0]][nLoop] = replaceKey(self.couplingRGEs[v[0]][nLoop], k, v[1])
                        RGEdic = self.couplingRGEs[v[0]][nLoop]

                    for c, bFunc in RGEdic.items():
                        self.couplingRGEs[cType][nLoop][c] = bFunc.subs(*subRule(k, v))

            cType = self.allCouplings[k][0]

            newAllCouplingsEntry = list(self.allCouplings[k])
            newAllCouplingsEntry[1] = subRule(k, v)[1]
            newAllCouplingsEntry = tuple(newAllCouplingsEntry)

            self.allCouplings = replaceKey(self.allCouplings, k, v[1], newVal=newAllCouplingsEntry)
            self.couplingsPos[cType] = replaceKey(self.couplingsPos[cType], k, v[1])
            if k in self.couplingStructure:
                self.couplingStructure = replaceKey(self.couplingStructure, k, v[1])

            # If the coupling appears in 'expandedPotential', replace it
            for ct, dic in self.expandedPotential.items():
                if k in dic:
                    self.expandedPotential[ct] = replaceKey(dic, k, v[1])

            # Rename the gauge couplings in the kinetic mixing matrix
            if self.kinMix and cType == 'GaugeCouplings':
                self.kinMat = self.kinMat.subs(*subRule(k,v))

    if 'zero' in substitutionDic and substitutionDic['zero'] != {}:
        for k,v in substitutionDic['zero'].items():
            cType, symb = self.allCouplings[k][:2]

            for cType, loopDic in self.couplingRGEs.items():
                for nLoop, RGEdic in loopDic.items():
                    for c, bFunc in list(RGEdic.items()):
                        self.couplingRGEs[cType][nLoop][c] = bFunc.subs(symb, 0).subs(Trace(0), 0)

                        if self.couplingRGEs[cType][nLoop][c] == 0:
                            self.couplingRGEs[cType][nLoop][c] = Integer(0)

            substitutionDic['zero'][k] = symb

    if self.gutNorm != {}:
        for k,v in self.gutNorm.items():
            cType, k = self.allCouplings[str(k)]

            v = list(v)
            v[1] = v[1].subs(v[0], k)
            v[2] = v[2].subs(v[0], k)

            deriv = diff(v[1], k)

            for nLoop, RGEdic in self.couplingRGEs[cType].items():
                newRGE = expand(deriv * RGEdic[str(k)]).subs(k, v[2], simultaneous=True)
                self.couplingRGEs[cType][nLoop][str(k)] = newRGE

            for cType, loopDic in self.couplingRGEs.items():
                for nLoop, RGEdic in loopDic.items():
                    for c, bFunc in list(RGEdic.items()):
                        if c == str(k):
                            continue
                        self.couplingRGEs[cType][nLoop][c] = bFunc.subs(k, v[2])

    if 'yukMat' in substitutionDic and substitutionDic['yukMat'] != {}:

        ##############################################################
        # Definition of local functions used to perform substitution #
        ##############################################################
        storeDic = {}

        def traceSub(tr):
            if tr in storeDic:
                return storeDic[tr]

            subDic = {}
            identitySubs = {}
            toExplicit = {}

            atoms = set([list(el.atoms())[0] for el in splitPow(tr.args[0])])
            for el in atoms:
                if not isinstance(el, mSymbol):
                    continue
                if str(el) in substitutionDic['yukMat']:
                    sub = substitutionDic['yukMat'][str(el)][1]

                    if not hasattr(sub, 'find') or sub.find(Identity) == {}:
                        subDic[el] = sub
                    else:
                        identitySubs[el] = sub.args_cnc()[0][0]
                else:
                    toExplicit[el] = el.as_explicit()
                    if el not in self.ExplicitMatrices:
                        self.ExplicitMatrices.append(el)

            # Identity substitutions
            if identitySubs != {}:
                shape = self.couplingStructure[str(atoms[0])][0]
                comm = 1
                noncomm = []

                for el in splitPow(tr.args[0]):
                    mat = list(el.atoms())[0]
                    if mat in identitySubs:
                        if isinstance(el, adjoint) or isinstance(el, conjugate):
                            comm *= conjugate(identitySubs[mat])
                        else:
                            comm *= identitySubs[mat]
                    else:
                        if mat == el:
                            noncomm.append(el)
                        else:
                            noncomm.append(el.__class__(mat))

                if noncomm == []:
                    ret = shape*comm
                else:
                    ret = comm*traceSub(trace(Mul(*noncomm)))

                storeDic[tr] = ret
                return ret

            if subDic == {}:
                ret = tr
            else:
                subDic = {**subDic, **toExplicit}
                ret = tr.subs(subDic, simultaneous=True).doit()

            storeDic[tr] = ret
            return ret

        def matSub(mats):
            if mats in storeDic:
                return storeDic[mats]

            subDic = {}
            toExplicit = {}
            identitySubs = {}

            atoms = flatten([m.atoms() for m in mats])
            for el in atoms:
                if not isinstance(el, mSymbol):
                    continue
                if str(el) in substitutionDic['yukMat']:
                    sub = substitutionDic['yukMat'][str(el)][1]

                    if not hasattr(sub, 'find') or sub.find(Identity) == {}:
                        subDic[el] = sub
                    else:
                        identitySubs[el] = sub.args_cnc()[0][0]
                else:
                    toExplicit[el] = el.as_explicit()
                    if el not in self.ExplicitMatrices:
                        self.ExplicitMatrices.append(el)

            # Identity substitutions
            if identitySubs != {}:
                shape = self.couplingStructure[str(atoms[0])][0]
                comm = 1
                noncomm = []

                for el in mats:
                    mat = list(el.atoms())[0]
                    if mat in identitySubs:
                        if isinstance(el, adjoint) or isinstance(el, conjugate):
                            comm *= conjugate(identitySubs[mat])
                        else:
                            comm *= identitySubs[mat]
                    else:
                        if mat == el:
                            noncomm.append(el)
                        else:
                            print(el, mat, el.__class__)
                            noncomm.append(el.__class__(mat))

                if noncomm == []:
                    ret = comm*Identity(shape)
                else:
                    ret = comm*matSub(tuple(noncomm))

                storeDic[mats] = ret
                return ret

            if subDic == {}:
                ret = Mul(*mats)
            else:
                subDic = {**subDic, **toExplicit}
                ret = 1

                for el in mats:
                    ret *= el.subs(subDic)

            return ret



        def totalSub(RGE):
            # Returns a list with symbol and explicit matrix parts
            newRGE =  [Integer(0), Integer(0)]

            for term in RGE.as_coeff_add()[1]:
                matFactors = []
                traceFactors = []
                coeff = 1

                for el in splitPow(term.args):
                    if not el.is_commutative:
                        matFactors.append(el)
                    elif isinstance(el, Trace):
                        traceFactors.append(el)
                    else:
                        coeff *= el

                newTerm = coeff
                if not (matFactors == [] and traceFactors == []):

                    #Trace subs
                    for tr in traceFactors:
                        newTerm *= traceSub(tr)

                    #Matrices subs
                    newTerm = expand(newTerm*matSub(tuple(matFactors)))

                if not newTerm.is_Matrix or (isinstance(newTerm, MatMul) and isinstance(newTerm.args[1], Identity)) :
                   newRGE[0] += newTerm
                else:
                    if newRGE[1] == 0:
                        newRGE[1] = newTerm
                    else:
                        newRGE[1] += newTerm

            return newRGE


        def as_explicit(expr):
            terms = expr.as_coeff_add()[1]

            ret = Integer(0)

            for el in terms:
                coeffs = el.args_cnc()

                tmp = Mul(*coeffs[0])
                tmp *= MatMul(*[mat.as_explicit() for mat in coeffs[1]])

                if ret == 0:
                    ret = tmp
                else:
                    ret += tmp

            return ret

        # Actual substitution

        # Identify non-zero entries along with their position
        values = {}
        for k, v in substitutionDic['yukMat'].items():
            mat = v[1]

            if 'unitary' in self.assumptions[k] and self.assumptions[k]['unitary'] is True:
                loggingInfo("Warning : the 'unitary' assumption for matrix '" + k + "' is ignored, since an explicit substitution was given.")
                del self.assumptions[k]['unitary']

            if hasattr(mat, 'find') and mat.find(Identity) != set() and mat.find(Identity) != {}:
                values[k] = (mat.args_cnc()[0][0], mat.args_cnc()[1][0])
                continue
            values[k] = {}
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat[i,j] == 0 or type(mat[i,j]) == conjugate:
                        continue

                    if mat[i,j] not in values[k]:
                        values[k][mat[i,j]] = (i,j)

        for cType, loopDic in self.couplingRGEs.items():
            if 'Anomalous' in cType:
                continue

            for nLoop, RGEdic in loopDic.items():
                for c, bFunc in list(RGEdic.items()):

                    newRGE = totalSub(bFunc)

                    if c in values:
                        if type(values[c]) == tuple:
                            # This means the substitution is y*Identity

                            diagRGE = 0
                            for term in newRGE[0].as_coeff_add()[1]:
                                tmp = 1
                                for el in term.args:
                                    if not (hasattr(el, 'is_Identity') and el.is_Identity):
                                        tmp *= el

                                diagRGE += tmp

                            noIdentity = [el for el in newRGE[0].as_coeff_add()[1] if el.find(Identity) == set()]
                            if newRGE[1] != 0 or noIdentity != []:
                                loggingCritical("Waring : After substitution, the RGE of the diagonal Yukawa matrix '" + c + "' may not be in the form   y*Identity . Keeping its original form.")

                                for el in noIdentity:
                                    loggingCritical("\t" + str(el))

                                if cType not in self.NonZeroDiagRGEs:
                                    self.NonZeroDiagRGEs[cType] = {}
                                if c not in self.NonZeroDiagRGEs[cType]:
                                    self.NonZeroDiagRGEs[cType][c] = {}
                                self.NonZeroDiagRGEs[cType][c][nLoop] = diagRGE

                            else:
                                self.couplingRGEs[cType][nLoop] = replaceKey(self.couplingRGEs[cType][nLoop],
                                                                             c,
                                                                             str(values[c][0]),
                                                                             diagRGE)

                            self.allCouplings[str(values[c][0])] = list(self.allCouplings[c])
                            self.allCouplings[str(values[c][0])][1] = values[c][0]
                            self.allCouplings[str(values[c][0])] = tuple(self.allCouplings[str(values[c][0])])
                            self.couplingsPos[cType][str(values[c][0])] = self.couplingsPos[cType][c]

                            continue

                        elif newRGE[0] != 0:
                            newRGE[1] += as_explicit(newRGE[0])
                            newRGE[0] = Integer(0)

                        self.couplingRGEs[cType][nLoop] = replaceKey(self.couplingRGEs[cType][nLoop],
                                                                     c,
                                                                     tuple([str(el) for el in values[c].keys()]),
                                                                     [newRGE[1][pos] for pos in values[c].values()])


                        for newCoupling in values[c]:
                            self.allCouplings[str(newCoupling)] = list(self.allCouplings[c])
                            self.allCouplings[str(newCoupling)][1] = newCoupling
                            self.allCouplings[str(newCoupling)] = tuple(self.allCouplings[str(newCoupling)])
                            self.couplingsPos[cType][str(newCoupling)] = self.couplingsPos[cType][c]

                        lookedPos = set(values[c].values())

                        for i in range(newRGE[1].shape[0]):
                            for j in range(newRGE[1].shape[1]):
                                if newRGE[1][i,j] == 0 or (i,j) in lookedPos:
                                    continue

                                if inconsistentRGEerror:
                                    raise TypeError

                                self.NonZeroCouplingRGEs[cType][nLoop][Symbol('{'+c+'}_{'+str(i+1)+str(j+1)+'}')] = newRGE[1][i,j]
                                if (self.allCouplings[c][1],) not in self.ExplicitMatrices:
                                    self.ExplicitMatrices.append((self.allCouplings[c][1],))
                    else:
                        # If there are non-zero entries in the matrix part of the RGE,
                        # they are added to the result in the form of a dictionary
                        if newRGE[1] != 0:
                            newRGE[1] = {(i+1,j+1): newRGE[1][i,j] for i in range(newRGE[1].shape[0])
                                                                   for j in range(newRGE[1].shape[1])
                                                                   if newRGE[1][i,j] != 0}
                        else:
                            newRGE = newRGE[0]
                        self.couplingRGEs[cType][nLoop][c] = newRGE


    for k,v in list(self.NonZeroCouplingRGEs.items()):
        if all([dic == {} for dic in v.values()]):
            del self.NonZeroCouplingRGEs[k]


    if 'sub' in substitutionDic and substitutionDic['sub'] != {}:
        def subSet(list1, list2):
            # Checks whether list2 is a subset of list1
            clist1 = list(list1)
            for el2 in list2:
                if el2 in clist1:
                    clist1.remove(el2)
                else:
                    return False

            return True

        def properSub(expr, old, new):
            # Substitution function working for a*b -> c in expressions like a*x*b -> c*x
            # or a*x*b*y*b*a -> c**2 * x * y
            res = Integer(0)

            # Extract the symbolic part from 'old'
            oldTerms = flatten(old.as_coeff_mul())
            oldCoeff = Mul(*[el for el in oldTerms if el.is_number])
            oldSymb = splitPow([el for el in oldTerms if not el.is_number])

            addTerms = expr.as_coeff_add()[1]
            for term in addTerms:
                terms = flatten(term.as_coeff_mul())
                coeff = Mul(*[el for el in terms if el.is_number])
                terms = [el for el in terms if not el.is_number]
                symbs = splitPow(terms)
                traces = [el for el in terms if isinstance(el, Trace)]

                if traces != []:
                    noTraces = [el for el in terms if not isinstance(el, Trace)]
                    tmp = coeff * properSub(Mul(*noTraces), old, new)
                    for t in traces:
                        tmp *= trace(properSub(t.args[0], old, new))
                    res += tmp
                    continue

                while subSet(symbs, oldSymb):
                    for el in oldSymb:
                        symbs.remove(el)

                    symbs.append(new)
                    coeff /= oldCoeff

                res += coeff * Mul(*symbs)

            return res

        for k,v in substitutionDic['sub'].items():
            v = list(v)

            cType, c = self.allCouplings[str(k)][:2]

            if c.is_real:
                v[0] = Symbol(str(v[0]), real=True)

            # Replace the symbols by the true couplings
            v[1] = v[1].subs(k, c)
            v[2] = [(el[0].subs(k, c), el[1].subs(k, c)) for el in v[2]]

            self.allCouplings = insertKey(self.allCouplings, str(k), str(v[0]), (cType, v[0]))
            self.couplingsPos[cType] = insertKey(self.couplingsPos[cType], str(k), str(v[0]), self.couplingsPos[cType][str(k)]+.5)

            for nLoop, RGEdic in self.couplingRGEs[cType].items():
                newRGE = Integer(0)
                for el in v[2]:
                    if not isinstance(el[0], conjugate):
                        beta = RGEdic[str(el[0])]
                    else:
                        beta = conjugate(RGEdic[str(el[0].args[0])])

                    if newRGE == 0:
                        newRGE = beta * el[1]
                    else:
                        newRGE += beta * el[1]

                newRGE = properSub(expand(newRGE), v[1], v[0])

                self.couplingRGEs[cType][nLoop] = replaceKey(self.couplingRGEs[cType][nLoop],
                                                             str(c),
                                                             str(v[0]),
                                                             newRGE)

            for cType, loopDic in self.couplingRGEs.items():
                for nLoop, RGEdic in loopDic.items():
                    for c, bFunc in list(RGEdic.items()):
                        self.couplingRGEs[cType][nLoop][c] = properSub(bFunc, v[1], v[0])


    # Now handle Yukawa matrix unitarity
    unitaryMatrices = {}
    unitarySubs = {}
    for k,v in self.allCouplings.items():
        if k in self.assumptions and 'unitary' in self.assumptions[k] and self.assumptions[k]['unitary'] is True:
            mat = v[1]
            unitarySubs.update({el:Identity(mat.shape[0]) for el in [adjoint(mat)*mat, mat*adjoint(mat), conjugate(mat)*transpose(mat), transpose(mat)*conjugate(mat)]})
            unitaryMatrices[k] = v

    for cType, loopDic in self.couplingRGEs.items():
        for nLoop, RGEdic in loopDic.items():
            for c, bFunc in list(RGEdic.items()):
                newRGE = bFunc.subs(unitarySubs)
                newRGE = newRGE.replace(lambda x: x.is_Pow and isinstance(x.base, Identity), lambda x: x.base).doit()
                self.couplingRGEs[cType][nLoop][c] = newRGE
