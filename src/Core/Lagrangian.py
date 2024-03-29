# -*- coding: utf-8 -*-
from sys import exit

import copy

from Logging import loggingCritical, print_progress
from Definitions import GaugeGroup, Tensor, mSymbol, splitPow, tensorContract, insertKey
from Particles import Particle

import itertools

from sympy import (Adjoint, Function, I, Indexed, IndexedBase, Mul, Pow, Symbol, Wild,
                   conjugate, expand, flatten)

from sympy.functions.special.tensor_functions import Eijk
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication


class TensorObject(Tensor):
    """ This class represent the various quantities that one can find
    in the Lagrangian.
    -> Particles with indices, repMatrices, CGCs, Eps, kronecker delta. """

    epsDic = {}
    kdDic = {}

    def __init__(self, *args, copy=None, fromDef='', expr=None, fields=[]):
        # Initialization from an existing tensor-dic : (symbol, ranges, dic)
        self.fromDef = fromDef
        self.expr = expr
        self.fields = fields
        self.conj = False

        if copy is not None:
            self.range = copy[1]
            self.dim = len(self.range)
            self.dic = copy[2]
            self.sym = False

            name = copy[0]
            if '[' in name:
                name = name[:name.find('[')]
            if self.dim > 0:
                self.symbol = IndexedBase(name)
            else:
                self.symbol = Symbol(name)

            return

        if len(args) == 1:
            arg = args[0]

            if isinstance(arg, Particle):
                self.initFromParticle(arg)
            elif isinstance(arg, str):
                self.initFromString(arg)

    def initFromParticle(self, p):
        Tensor.__init__(self, p.indexStructure)

        if self.dim == 0:
            self.symbol = p._name
            if not p.cplx:
                self.dic[()] = self.symbol
            else:
                self.dic[()] = p.norm*sum([x*field._name
                                       for x,field in zip(p.realComponents, p.realFields)])
        else:
            self.symbol = IndexedBase(str(p._name))
            inds = itertools.product(*[range(i) for i in self.range])

            if p.pseudoRealReps == []:
                for i in inds:
                    if not p.cplx:
                        self.dic[i] = self.symbol[i]
                    else:
                        self.dic[i] = p.norm*sum([x*IndexedBase(str(field._name))[i]
                                              for x,field in zip(p.realComponents, p.realFields)])
            else:
                for i in inds:
                    self.dic[i] = p.computeComponents(i, self.symbol)


    def initFromString(self, s):
        if s.lower() == 'kd':
            # Kronecker delta
            self.symbol = IndexedBase('KD')
            self.dic = None
            self.range = None
            self.dim = None
            self.sym = False
        elif s.lower() == 'eps':
            # Kronecker delta
            self.symbol = IndexedBase('Eps')
            self.dic = None
            self.range = None
            self.dim = None
            self.sym = False
        else:
            loggingCritical("Error : Unkown tensor object '{s}'.")
            return

    def update(self, N):
        """ Updates the range/dim/dic of Eps and KD objects """

        if self.dic is not None:
            return

        if str(self.symbol) == 'Eps':
            self.range = tuple([N]*N)
            self.dim = N

            if N in self.epsDic:
                self.dic = self.epsDic[N]
                return

            self.dic = {}

            indCombinations = itertools.product(*([range(N)]*N))
            for inds in indCombinations:
                val = Eijk(*inds)
                if val != 0:
                    self.dic[inds] = val

            self.epsDic[N] = self.dic

    def getConjugate(self):
        conj = TensorObject()
        conj.conj = True
        conj.fromDef = self.fromDef
        if conj.fromDef != '':
            if '[' not in conj.fromDef:
                conj.fromDef += 'bar'
            else:
                conj.fromDef = conj.fromDef.replace('[', 'bar[')
        conj.expr = self.expr
        conj.fields = self.fields
        conj.range, conj.dim, conj.sym = self.range, self.dim, self.sym

        conj.dic = {k:self.dic[k].subs(I, -I) for k,v in self.dic.items()}

        if isinstance(self.symbol, Symbol):
            conj.symbol = Symbol(str(self.symbol) + 'bar')
        if isinstance(self.symbol, IndexedBase):
            conj.symbol = IndexedBase(str(self.symbol) + 'bar')

        return conj


    def __repr__(self):
        s = ''
        if hasattr(self, 'symbol'):
            s += "Tensor object : " + str(self.symbol) + '\n'
        return s + Tensor.__repr__(self)


class Lagrangian():
    class IndNumberError(Exception):
        pass
    class IndRangeError(Exception):
        pass
    class IndContractionError(Exception):
        pass

    def __init__(self, settings, model, RGmodule):
        self.model = model
        self.RGmodule = RGmodule
        self.idb = model.idb

        # Copy some useful attributes from the Model class
        self.potential = model.potential
        self.translateContent = model.translateContent
        self.translateDic = model.translateDic
        self.assumptions = model.assumptions

        self.dicToFill = {}
        self.currentPotentialDic = {}

        self.expandedPotential = {key:{} for key in self.potential}
        self.fullyExpandedPotential = {key:{} for key in self.potential}

        # First, read the user-defined objects
        self.definitions = {}
        self.cgcs = {}
        self.getDefinitions(settings)

        self.allFermionsValues = list(model.allFermions.values())
        self.allScalarValues = list(model.allFermions.values())

    def getDefinitions(self, settings):
        #######################
        # Pre-Defined objects #
        #######################

        preDefinedObjects = {}

        # #Kronecker delta
        preDefinedObjects['kd'] = TensorObject("kd")

        # #Eps tensors
        preDefinedObjects['Eps'] = TensorObject("Eps")

        self.definitions.update(preDefinedObjects)

        #############
        # Particles #
        #############

        particleTensors = {}
        for name, p in self.model.Particles.items():
            particleTensors[name] = TensorObject(p)

            # Self-conjugate scalars have a complex conjugate counterpart
            if p.pseudoRealReps != []:
                particleTensors[name + 'bar'] = TensorObject(p).getConjugate()

        self.definitions.update(particleTensors)

        ########################
        # User-defined objects #
        ########################

        if 'Definitions' in settings['Potential'] and settings['Potential']['Definitions'] != {}:
            for k,v in settings['Potential']['Definitions'].items():
                obj = self.parseExpression(v, name=k)

                if obj is not None:
                    self.definitions[str(obj.symbol)] = obj

                    if str(type(obj.expr)) == 'cgc':
                        self.cgcs[str(obj.symbol)] = obj

                    if str(type(obj.expr)) == 't':
                        self.definitions[str(obj.symbol)+'bar'] = obj.getConjugate()
                else:
                    loggingCritical(f"Warning : unable to read the definition '{k}: {v}'. Skipping.")


    def parseExpression(self, expr, name=None, expandedTerm=None, coupling=None):
        """ This function handles the convertion from str to TensorObject of
        lagrangian expressions written by the user in the model file.
        As much as possible, the user input is validated and error messages
        are printed if needed."""

        originalExpr = expr
        errorExpr = (name + ' : ' if name is not None else '') + str(originalExpr)

        ##########
        # Case 1 : expr is a representation matrix
        ##########

        if expr[:2].lower() == 't(':
            args = expr[2:-1]
            gp = args.split(',')[0]

            if gp + ',' not in args:
                loggingCritical(f"\nError : representation matrix {expr} should have exactly two arguments : group and rep")
                exit()
            rep = eval(args.replace(gp + ',', ''))

            if gp in self.model.gaugeGroups:
                gp = self.model.gaugeGroups[gp]
            else:
                for gName, g in self.model.gaugeGroups.items():
                    if g.type == gp:
                        gp = g
                        break
                if type(gp) == str:
                    loggingCritical(f"\nError in 'Definitions': gauge group '{gp}' is unknown.")
                    exit()

            # DimR -> Dynkin labels
            if isinstance(rep, int):
                rep = self.idb.get(gp.type, 'dynkinLabels', rep)

            repMats = gp.repMat(tuple(rep))

            shape = tuple([len(repMats), *repMats[0].shape])
            dic = {}
            for i, mat in enumerate(repMats):
                for k,v in mat.todok().items():
                    dic[(i,*k)] = v

            # This is for latex output
            expr = Function('t')(Symbol(gp.type), Symbol(str(rep)))

            return TensorObject(copy=(name, shape, dic), fromDef=name, expr=expr)

        ##########
        # Case 2 : expr is a CGC
        ##########

        if expr[:4].lower() == 'cgc(':
            args = expr[4:-1].split(',')

            # Read the gauge group
            gp = args[0]
            args = args[1:]

            # Collect lists together
            i = 0
            while i < len(args):
                o = args[i].count('[') + args[i].count('(') + args[i].count('{')
                c = args[i].count(']') + args[i].count(')') + args[i].count('}')
                if o > c:
                    args[i] = args[i] + ', ' + args[i+1]
                    args.pop(i+1)
                else:
                    i += 1

            # Read the fields
            fields = []
            for i, el in enumerate(args):
                if el.isnumeric() or ('(' in el and ')' in el):
                    # Stop after encountering an int or a tuple
                    i -= 1
                    break
                fields.append(el)
            args = args[i+1:]

            # Determine which syntax was used
            if gp in self.model.gaugeGroups and all([el in self.model.Particles for el in fields]):
                fieldNames = fields
            elif gp in [gp.type for gp in self.model.gaugeGroupsList] and all([el not in self.model.Particles for el in fields]) :
                fieldNames = []
            else:
                loggingCritical("\nError : CGC syntax is 'cgc(groupName, field1, field2 [, field3 [, field4, [CGC number]]])' or "
                                + "cgc(group, dynkins1, dynkins2 [, dynkins3 [, dynkins4, [CGC number]]]). The group and particles must be defined above.")
                loggingCritical(f"Please rewrite the term '{name}: {expr}' accordingly.")
                exit()

            N = 0
            # The CGC call contains a pos
            if args != []:
                if len(args) == 1:
                    N = int(args[0])-1
                else:
                    loggingCritical(f"\nError in {name}: {expr} ; too much arguments to cgc() function.")
                    exit()
                if N < 0:
                    loggingCritical(f"\nError in {name}: {expr} ; the position argument must be a non-zero positive integer.")
                    exit()

            if not isinstance(N, int):
                loggingCritical(f"\nError in CGC '{name}: {expr}' : position argument must be an integer.")
                exit()

            if fieldNames != []:
                gpName, gType = gp, self.model.gaugeGroups[gp].type
                reps = [self.model.Particles[p].Qnb[gpName] for p in fields]
            else:
                gType, reps = gp, [eval(labels) for labels in fields]

            cgc = self.idb.get(gType, 'invariants', reps,
                               pyrateNormalization=True,
                               realBasis=GaugeGroup.realBasis)

            if len(cgc) == 0:
                loggingCritical(f"Error: no invariant can be formed from the reps provided in '{name}'.")
                exit()
            if N > len(cgc)-1:
                loggingCritical(f"\nError in {name}: {expr} ; the position argument cannot be larger than {len(cgc)} here.")
                exit()

            result = cgc[N]

            shape = result.dim[:result.rank]
            dic = {}
            for k,v in result.dic.items():
                dic[k[:result.rank]] = v

            # This is for latex output
            expr = Function('cgc')(Symbol(gType), *([Symbol(el) for el in fields]+[N]))

            return TensorObject(copy=(name, shape, dic), fromDef=name, expr=expr, fields=fieldNames)

        ##########
        # Case 3 : an expression involving the already defined quantities
        ##########
        localDict = {}

        count = 0
        expr = expr.replace('sqrt', '#').replace('Sqrt', '#')
        for k,v in sorted(self.definitions.items(), key=lambda x:-len(x[0])):
            expr = expr.replace(k, f'@_{count}_')
            localDict[f'symb_{count}_'] = v.symbol
            count += 1
        expr = expr.replace('@', 'symb')
        expr = expr.replace('#', 'sqrt')

        def sympyParse(expr):
            if '^' in expr:
                loggingCritical(f"\nError in expression '{errorExpr}' : powers must be written using the '**' operator")
                exit()
            return parse_expr(expr, local_dict = localDict,
                              transformations=standard_transformations[1:] + (implicit_multiplication,),
                              evaluate=False)

        # A) Replacements to format the expr string
        expr = expr.replace(']', '] ').strip()
        expr = expr.replace(' +', '+').replace(' -', '-').replace(' *', '*').replace(' /', '/')
        expr = expr.replace(' )', ')')
        expr = expr.replace('] ', ']*')

        for k,v in localDict.items():
            if isinstance(v, Symbol):
                expr = expr.replace(k, k + ' ')

        # B) Parse the string
        try:
            expr = sympyParse(expr)
        except:
            loggingCritical(f"\nError while parsing the term '{errorExpr}'.")
            exit()

        rep = {}
        if expr.find(Pow) != set():
            # Now we want to expand the term, keeping any form (a*b)**2 unexpanded
            a,b,c = [Wild(x, exclude=(1,)) for x in 'abc']
            rep = expr.replace((a*b)**c, lambda a,b,c : (a*b)**Symbol('n_' + str(c)), map=True)

        if rep == {} or rep[1] == {}:
            termList = expand(expr).as_coeff_add()[1]
        else:
            termList = expand(rep[0], power_base=False).as_coeff_add()[1]

        # C) Parse the left hand side of the definition (if given)
        Linds = []
        if name is not None:
            if '[' in name and ']' in name:
                Lbase = name[:name.find('[')]
                Linds = name[name.find('[')+1:name.find(']')].split(',')
                Linds = [Symbol(el) for el in Linds]

        # D) Validate and compute the expression
        rhsResult = 0
        commonFreeInds = None

        for term in termList:
            split = splitPow(term)

            rationalFactors = [el for el in split if el.is_number]
            terms = tuple([el for el in split if el not in rationalFactors])
            coeff = Mul(*rationalFactors)

            # Handle expr**N now
            newTerms = []
            for subTerm in terms:
                if isinstance(subTerm, Pow):
                    base, exp = subTerm.base, subTerm.exp
                    if isinstance(exp, Symbol):
                        exp = int(exp.name[2:])
                    indexed = base.find(Indexed)

                    if indexed != set():
                        indices = flatten([el.indices for el in indexed])

                        indCopies = {}
                        for i in indices:
                            if i not in indCopies:
                                indCopies[i] = [Symbol(str(i)+f'_{p}') for p in range(1,exp)]

                    else:
                        indCopies = {}
                    newTerms.append(base)
                    for p in range(0,exp-1):
                        sub = {i:copy[p] for i,copy in indCopies.items()}
                        newTerms.append(base.subs(sub))
                else:
                    newTerms.append(subTerm)

            terms = []
            for subTerm in newTerms:
                if isinstance(subTerm, Mul) or isinstance(subTerm, Pow):
                    tmp = splitPow(subTerm)
                    for el in tmp:
                        if not el.is_number:
                            terms.append(el)
                        else:
                            coeff *= el
                else:
                    if not subTerm.is_number:
                        terms.append(subTerm)
                    else:
                        coeff *= subTerm

            if expandedTerm is not None:
                if expandedTerm == []:
                    expandedTerm.append(Mul(coeff, *terms, evaluate=False))
                else:
                    expandedTerm[0] += Mul(coeff, *terms, evaluate=False)

            inds = []
            indRanges = {}
            for i, field in enumerate(terms):
                if isinstance(field, Symbol):
                    continue
                try:
                    fieldInds = field.indices
                except AttributeError:
                    loggingCritical(f"\nError (in term '{expr}') while reading the quantity '{field}'. It seems that indices are missing.")
                    exit()

                fieldDef = self.definitions[str(field.base)]
                if fieldDef.dim is not None and len(fieldInds) != fieldDef.dim:
                    loggingCritical(f"\nError (in term '{expr}'): the quantity {field.base} should carry exactly {fieldDef.dim} indices")
                    exit()

                inds += list(fieldInds)
                for p, ind in enumerate(field.indices):
                    indRanges[ind] = (fieldDef, p)

            freeInds = []
            for ind in set(inds):
                count = inds.count(ind)
                if count == 1:
                    freeInds.append(ind)
                if count > 2:
                    loggingCritical(f"\nError: in term '{term}', the index '{ind}' appears more than twice.")
                    exit()

            if commonFreeInds is None:
                commonFreeInds = freeInds
            elif freeInds != commonFreeInds:
                loggingCritical(f"\nError : each term of the sum '{expr}' must contain the same free indices.")
                exit()
            if name is not None and set(freeInds) != set(Linds):
                loggingCritical(f"\nError in term {term}: there should be {len(set(Linds))} free indices" + (' -> ' + str(tuple(set(Linds))) if set(Linds) != set() else ''))
                exit()

            # Now that the term is validated, construct the resulting tensor object
            contractArgs = []
            fermions = []
            for field in terms:
                if not isinstance(field, Symbol):
                    base, inds = field.base, field.indices
                else:
                    base, inds = field, []

                # Store fermions to detect (anti-)symmetric Yukawas
                if str(base) in self.model.Fermions:
                    fermions.append(base)

                tens = self.definitions[str(base)]
                tens.update(len(inds))

                inds = [Wild(str(el)) for el in inds]
                contractArgs.append(tens(*inds))

            freeDummies = [Wild(str(el)) for el in Linds]

            # Handle (anti-)symmetric Yukawas
            if len(set(fermions)) != len(fermions) and coupling is not None:
                tmp = self.handleSymmetricYukawa(fermions, contractArgs, coeff, freeDummies, coupling)
            else:
                tmp = tensorContract(*contractArgs, value=coeff, freeDummies=freeDummies, doit=True)

                if not isinstance(tmp, dict):
                    tmp = expand(tmp)

            if rhsResult == 0:
                rhsResult = tmp
            else:
                if not isinstance(rhsResult, dict):
                    rhsResult += tmp
                else:
                    for k,v in tmp.items():
                        v = expand(v)
                        if k in rhsResult:
                            rhsResult[k] += v
                        else:
                            rhsResult = k

                        if rhsResult[k] == 0:
                            del rhsResult[k]

        if not isinstance(rhsResult, dict):
            return TensorObject(copy=('' if name is None else name, (), {(): rhsResult}), fromDef=name, expr=expr)

        ranges = []

        for freeInd in Linds:
            iRange = indRanges[freeInd]
            iRange = iRange[0].range[iRange[1]]
            ranges.append(iRange)

        try:
            return TensorObject(copy=(Lbase, ranges, rhsResult), fromDef=name, expr=expr)
        except:
            loggingCritical(f"\nError while parsing the term '{errorExpr}': please check the consistency of contracted indices.")
            exit()


    def handleSymmetricYukawa(self, fermions, contractArgs, coeff, freeDummies, coupling):
        duplicateFermions = [el for el in fermions if fermions.count(el) == 2]

        if len(duplicateFermions) != 2:
            return expand(tensorContract(*contractArgs, value=coeff, freeDummies=freeDummies, doit=True))

        expanded = []

        duplicateFermion = duplicateFermions[0]

        symbs = [el[0].symbol for el in contractArgs]
        duplicatePos = [i for i, el in enumerate(symbs) if el == duplicateFermion]

        for pos in duplicatePos:
            duplicateTensor = copy.deepcopy(contractArgs[pos][0])

            newDic = {}
            for k, v in duplicateTensor.dic.items():
                if isinstance(v, Symbol):
                    newDic[k] = Symbol('df$' + str(v))
                elif isinstance(v, Indexed):
                    newDic[k] = Indexed('df$' + str(v.base), *v.indices)

            duplicateTensor.dic = newDic
            newcArgs = []
            for i, el in enumerate(contractArgs):
                if i != pos:
                    newcArgs.append(el)
                else:
                    if len(contractArgs[pos]) > 1:
                        newcArgs.append((duplicateTensor, *contractArgs[pos][1:]))
                    else:
                        newcArgs.append((duplicateTensor, ))

            expanded.append(expand(tensorContract(*newcArgs, value=coeff, freeDummies=freeDummies, doit=True)))

        if expand(expanded[0] + expanded[1]) == 0:
            # Antisymmetric Yukawa matrix

            # For one generation, the Yukawa operator simply vanishes.
            # In this case an error should be raised eventually
            if self.model.Fermions[str(duplicateFermion)].gen == 1:
                return expand(tensorContract(*contractArgs, value=coeff, freeDummies=freeDummies, doit=True))


            self.model.assumptions[coupling]['antisymmetric'] = True
        else:
            # Symmetric Yukawa matrix
            self.model.assumptions[coupling]['symmetric'] = True

        return expanded[0]


    def expand(self):
        """ Performs a first level of expansion of the Lagrangian. More precisely, replaces
        all the occurences of user-defined quantities with their expression."""

        def isComplex(cType, c, expTerm):
            return (cType in ('QuarticTerms', 'TrilinearTerms', 'ScalarMasses')
                    and c + 'star' not in self.potential[cType]
                    and not(c[-4:] == 'star' and c[:-4] in self.potential[cType])
                    and expTerm.find(I) != set())

        count = 0
        content = ()
        for couplingType, terms in self.potential.items():
            if couplingType in self.translateContent:
                self.dicToFill = self.translateDic(self.RGmodule)[couplingType]
                content = self.translateContent[couplingType]
                self.currentPotentialDic = self.potential[couplingType]
            else:
                continue

            for coupling, term in list(terms.items()):
                TensorObject.globalTensorCount = 1
                parsedTerm = []

                try:
                    expTerm = self.parseExpression(term, expandedTerm=parsedTerm, coupling=coupling).dic[()]
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except BaseException as e:
                    if str(e) != '':
                        loggingCritical(f"\nError while expanding the term '{coupling}':")
                        loggingCritical(f" -> {e}")
                    exit()

                self.expandedPotential[couplingType][coupling] = parsedTerm[0]
                self.fullyExpandedPotential[couplingType][coupling] = expTerm

                if isComplex(couplingType, coupling, expTerm):
                    if not ('real' in self.assumptions[coupling] and self.assumptions[coupling]['real'] is True):
                        self.conjugateScalarTerm(couplingType, coupling, content)
                        count += 1
                    else:
                        self.expandedPotential[couplingType][coupling] += Symbol('_hc')
                        self.fullyExpandedPotential[couplingType][coupling] += expTerm.subs(I, -I)

                self.fillTensorDic(coupling, expTerm, content)
                count += 1
                print_progress(count, self.model.nCouplings, prefix=' '*4, bar_length=20, printTime=self.model.times, logProgress=True)

            # Finally, remove all vanishing elements from dict
            for k,v in list(self.dicToFill.items()):
                if v == 0:
                    self.dicToFill.pop(k)

    def conjugateScalarTerm(self, cType, c, content):
        cStar = c + 'star'
        expTerm = self.fullyExpandedPotential[cType][c].subs(I, -I)

        self.potential[cType] = insertKey(self.potential[cType], c, cStar, self.potential[cType][c])
        self.expandedPotential[cType][cStar] = Symbol('_hc')
        self.fullyExpandedPotential[cType][cStar] = expTerm
        self.model.nCouplings += 1

        self.model.allCouplings = insertKey(self.model.allCouplings, c, cStar, cType)
        self.model.couplingsPos[cType][cStar] = self.model.couplingsPos[cType][c] + .5
        self.fillTensorDic(cStar, expTerm, content)

        self.model.assumptions[cStar] = self.model.assumptions[c]

    def fillTensorDic(self, coupling, expTerm, content):
        subTerms = expTerm.as_coeff_add()[1]
        tensorInds = ()
        coeff = 0

        fermions = ()
        scalars = ()

        coupling = Symbol(coupling, complex=True)

        cType = self.model.allCouplings[str(coupling)]
        if type(cType) == tuple:
            cType = cType[0]

        for subTerm in subTerms:
            subTerm = list(subTerm.as_coeff_mul())
            rationalFactors = [el for el in subTerm[1] if el.is_number]
            subTerm[1] = tuple([el for el in subTerm[1] if el not in rationalFactors])
            subTerm[0] *= Mul(*rationalFactors)

            subTerm[1] = splitPow(subTerm[1])

            #For fermions, we have to be careful that the order in which the user wrote the terms
            # is preserved here. Inverting them would transpose the Yukawa / mass matrix

            fermions = [self.model.allFermions[str(el).replace('df$', '')] for el in subTerm[1] if str(el).replace('df$', '') in self.model.allFermions]

            if fermions != []:
                # Sort fermions according to their pos in the potential term
                fermionSortKey = {}
                fermionNames = sorted([str(el[1]) for el in fermions], key=lambda x: len(x), reverse=True)
                potStr = self.currentPotentialDic[str(coupling)]
                for el in fermionNames:
                    fermionSortKey[el] = potStr.find(el)
                    potStr = potStr.replace(el, ' '*len(el))
                fermions = sorted(fermions, key=lambda x: fermionSortKey[str(x[1])])

            fGen = [f[1].gen for f in fermions]
            fermions = [f[0] for f in fermions]

            scalars = [self.model.allScalars[str(el)][0] for el in subTerm[1] if str(el) in self.model.allScalars ]
            # sGen = [self.model.allScalars[str(el)][1].gen for el in subTerm[1] if str(el) in self.model.allScalars]

            if content == (2,1): #Yukawa
                if len(fermions) != 2 or len(scalars) != 1:
                    loggingCritical(f"Error in term {str(coupling)} : \n\tYukawa terms must contain exactly 2 fermions and 1 scalar.")
                    exit()
                tensorInds = tuple(scalars + fermions)
                coeff = subTerm[0] * 2/len(set(itertools.permutations(fermions, 2)))

                assumptionDic = self.model.assumptions[str(coupling)]

                coupling = mSymbol(str(coupling), fGen[0], fGen[1], **assumptionDic)
                if coupling not in self.model.couplingStructure:
                    self.model.couplingStructure[str(coupling)] = (fGen[0], fGen[1])

            if content == (0,4): #Quartic
                if len(fermions) != 0 or len(scalars) != 4:
                    loggingCritical(f"\nError in term {str(coupling)} : \n\tQuartic terms must contain exactly 0 fermion and 4 scalars.")
                    exit()
                tensorInds = tuple(sorted(scalars))
                coeff = subTerm[0] * 24/len(set(itertools.permutations(tensorInds, 4)))#/len(set(itertools.permutations(tensorInds, 4)))

            if content == (0,3): #Trilinear
                if len(fermions) != 0 or len(scalars) != 3:
                    loggingCritical(f"\nError in term {str(coupling)} : \n\tTrinilear terms must contain exactly 0 fermion and 3 scalars.")
                    exit()
                tensorInds = tuple(sorted(scalars))
                coeff = subTerm[0] * 6/len(set(itertools.permutations(tensorInds, 3)))

            if content == (0,2): #Scalar Mass
                if len(fermions) != 0 or len(scalars) != 2:
                    loggingCritical(f"\nError in term {str(coupling)} : \n\tScalar mass terms must contain exactly 0 fermion and 2 scalars.")
                    exit()
                tensorInds = tuple(sorted(scalars))
                coeff = subTerm[0] * 2/len(set(itertools.permutations(tensorInds, 2)))

            if content == (2,0): #Fermion Mass
                if len(fermions) != 2 or len(scalars) != 0:
                    loggingCritical(f"\nError in term {str(coupling)} : \n\tFermion mass terms must contain exactly 2 fermions and 0 scalar.")
                    exit()
                tensorInds = tuple(fermions)
                coeff = subTerm[0] * 2/len(set(itertools.permutations(tensorInds, 2)))

                assumptionDic = self.model.assumptions[str(coupling)]

                coupling = mSymbol(str(coupling), fGen[0], fGen[1], **assumptionDic)
                if coupling not in self.model.couplingStructure:
                    self.model.couplingStructure[str(coupling)] = (fGen[0], fGen[1])

            if tensorInds not in self.dicToFill:
                self.dicToFill[tensorInds] = coupling*coeff
            else:
                self.dicToFill[tensorInds] += coupling*coeff

            #Update the 'AllCouplings' dictionary
            if type(self.model.allCouplings[str(coupling)]) != tuple:
                tmp = [cType, coupling]

                if isinstance(coupling, mSymbol):
                    orderedFermions = [str(list(self.model.allFermions.values())[f][1]) for f in fermions]
                    tmp.append(tuple(orderedFermions))

                self.model.allCouplings[str(coupling)] = tuple(tmp)

            # If Yukawa / Fermion mass, properly fill the dictionary, taking into
            # account possible sym/antisym couplings.
            # Then, fill the conjugate part of y(a,i,j).
            if content == (2,1) or content == (2,0):
                antiFermions = [self.antiFermionPos(f) for f in fermions]
                antiFermions.reverse()
                adjTensorInds = tuple(scalars + antiFermions + [True])
                adjCoeff = conjugate(coeff)
                adjCoupling = Adjoint(coupling).doit()

                if adjTensorInds not in self.dicToFill:
                    self.dicToFill[adjTensorInds] = adjCoupling*adjCoeff
                else:
                    self.dicToFill[adjTensorInds] += adjCoupling*adjCoeff

                if tensorInds[-2] != tensorInds[-1]:
                    rev = self.reverseFermionInds(tensorInds)
                    revAdj = tuple(list(self.reverseFermionInds(adjTensorInds[:-1])) + [True])

                    if rev not in self.dicToFill:
                        self.dicToFill[rev] = (coupling*coeff).transpose()
                        self.dicToFill[revAdj] = (adjCoupling*adjCoeff).transpose()
                    else:
                        self.dicToFill[rev] += (coupling*coeff).transpose()
                        self.dicToFill[revAdj] += (adjCoupling*adjCoeff).transpose()


    def reverseFermionInds(self, tensorInds):
        reversedInds = list(tensorInds)
        (reversedInds[-1], reversedInds[-2]) = (reversedInds[-2], reversedInds[-1])
        return tuple(reversedInds)

    def getFermionShape(self, i, j):
        return (lambda l: (l[i][1].gen, l[j][1].gen))(self.allFermionsValues)

    def antiFermionPos(self, pos):
        aux = self.allFermionsValues[pos][0:2]
        if aux[1].conj:
            return aux[0] - len(self.model.allFermions)//2
        return aux[0] + len(self.model.allFermions)//2
