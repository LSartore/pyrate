# -*- coding: utf-8 -*-
from sys import exit

from Logging import loggingInfo, loggingCritical, print_progress
from Definitions import GaugeGroup, Tensor, mSymbol, splitPow, tensorContract
from Particles import Particle

import itertools

from sympy import (Adjoint, Function, Indexed, IndexedBase, Mul, Pow, Symbol, Wild,
                   conjugate, expand, flatten, sqrt, sympify)

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
            
            for i in inds:
                if not p.cplx:
                    self.dic[i] = self.symbol[i]
                else:
                    self.dic[i] = p.norm*sum([x*IndexedBase(str(field._name))[i]
                                          for x,field in zip(p.realComponents, p.realFields)])
    
    
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
            
    # def __getitem__(self, inds):
    #     if self.dic is not None:
    #         return Tensor.__getitem__(self, inds)
    #     else:
    #         print(" ## GETITEM, NONE ! ##")


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
        
        self.dicToFill = {}
        self.currentPotentialDic = {}
        
        self.expandedPotential = {key:{} for key in self.potential}
        self.fullyExpandedPotential = {key:{} for key in self.potential}
        
        # First, read the user-defined objects
        self.definitions = {}
        self.cgcs = {}
        self.getDefinitions(settings)
        
        self.allFermionsValues = list(model.allFermions.values())
        
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
        
        particleTensors = {name: TensorObject(p) for name, p in self.model.Particles.items()}
        
        # self.ParticleDefinitions = particleTensors
        self.definitions.update(particleTensors)
        
        ########################
        # User-defined objects #
        ########################
        
        if 'Definitions' in settings['Potential'] and settings['Potential']['Definitions'] != {}:
            for k,v in settings['Potential']['Definitions'].items():
                # print('\n\n->', k, ':', v)
                obj = self.parseExpression(v, name=k)
                if obj is not None:
                    self.definitions[str(obj.symbol)] = obj
                    
                    if str(type(obj.expr)) == 'cgc':
                        self.cgcs[str(obj.symbol)] = obj
                else:
                    loggingCritical(f"Warning : unable to read the definition '{k}: {v}'. Skipping.")

    def parseExpression(self, expr, name=None, expandedTerm=None):
        """ This function handles the convertion from str to TensorObject of
        lagrangian expressions written by the user in the model file.
        As much as possible, the user input is validated and error messages 
        are printed if needed."""
        
        ##########
        # Case 1 : expr is a representation matrix
        ##########
        
        if expr[:2].lower() == 't(':
            args = expr[2:-1]
            gp = args.split(',')[0]
            # print(gp, args.replace(gp + ',', ''))
            
            if gp + ',' not in args:
                loggingCritical(f"Error : representation matrix {expr} should have exactly two arguments : group and rep")
                return
            rep = eval(args.replace(gp + ',', ''))
            
            if gp in self.model.gaugeGroups:
                gp = self.model.gaugeGroups[gp].type
            
            # DimR -> Dynkin labels
            if isinstance(rep, int):
                rep = self.idb.get(gp, 'dynkinLabels', rep)
            
            repMats = self.idb.get(gp, 'repMatrices', rep)
            shape = tuple([len(repMats), *repMats[0].shape])
            dic = {}
            for i, mat in enumerate(repMats):
                for k,v in mat._smat.items():
                    dic[(i,*k)] = v
            
            # This is for latex output
            expr = Function('t')(Symbol(gp), Symbol(str(rep)))
            
            return TensorObject(copy=(name, shape, dic), fromDef=name, expr=expr)
        
        ##########
        # Case 2 : expr is a CGC
        ##########
        
        # Several ways to call cgc():
        # -> 
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
            # lastpos = 0
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
                loggingCritical("Error : CGC syntax is 'cgc(groupName, field1, field2 [, field3 [, field4, [CGC number]]])' or "
                                + "cgc(group, dynkins1, dynkins2 [, dynkins3 [, dynkins4, [CGC number]]]). The group and particles must be defined above.")
                loggingCritical(f"Please rewrite the term '{name}: {expr}' accordingly.")
                return
            
            N = 0
            # The CGC call contains a pos
            if args != []:
                # print(args)
                if len(args) == 1:
                    N = int(args[0])
                else:
                    loggingCritical(f"Error in {name}: {expr} ; too much arguments to cgc() function.")
                    return
                
            if not isinstance(N, int):
                loggingCritical(f"Error in CGC '{name}: {expr}' : position argument must be an integer.")
                
            if fieldNames != []:
                gpName, gType = gp, self.model.gaugeGroups[gp].type
                reps = [self.model.Particles[p].Qnb[gpName] for p in fields]
            else:
                gType, reps = gp, [eval(labels) for labels in fields]
                
            cgc = self.idb.get(gType, 'invariants', reps,
                               pyrateNormalization=True,
                               realBasis=GaugeGroup.realBasis)
            
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
        
        barCount = 0
        for k,v in self.definitions.items():
            if k[-3:] != 'bar':
                localDict[k] = v.symbol
            else:
                expr = expr.replace(k, '_bar' + str(barCount))
                localDict['_bar' + str(barCount)] = v.symbol
                barCount += 1

        def sympyParse(expr):
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
        
        # print('Str expr :', expr)
        # B) Parse the string
        expr = sympyParse(expr)
        
        # print("Expr : ", expr)
        
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
            # Lbase = name
            # LnInds = 0
            if '[' in name and ']' in name:
                Lbase = name[:name.find('[')]
                Linds = name[name.find('[')+1:name.find(']')].split(',')
                Linds = [Symbol(el) for el in Linds]
                # LnInds = len(Linds)
        
        # D) Validate and compute the expression
        
        rhsResult = 0
        commonFreeInds = None
        for term in termList:
            coeff, terms = term.as_coeff_mul()
            
            rationalFactors = [el for el in terms if el.is_number]
            terms = tuple([el for el in terms if el not in rationalFactors])
            coeff *= Mul(*rationalFactors)
            
            # print("Coeff, terms : ", coeff, terms)
            
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
                            if base.count(i) != 2:
                                loggingCritical(f"Error in expression {subTerm} : all indices must be contracted.")
                                return
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
            
            # print("TERMS :", terms)
            if expandedTerm is not None:
                if expandedTerm == []:
                    expandedTerm.append(Mul(coeff, *terms))
                else:
                    expandedTerm[0] += Mul(coeff, *terms)
            
            inds = []
            indRanges = {}
            for i, field in enumerate(terms):
                if isinstance(field, Symbol):
                    continue
                inds += list(field.indices)
                for p, ind in enumerate(field.indices):
                    indRanges[ind] = (self.definitions[str(field.base)], p)
            
            freeInds = []
            for ind in set(inds):
                count = inds.count(ind)
                if count == 1:
                    freeInds.append(ind)
                if count > 2:
                    loggingCritical(f"Error: in term '{term}', the index '{ind}' appears more than twice.")
                    return
            
            if commonFreeInds is None:
                commonFreeInds = freeInds
            elif freeInds != commonFreeInds:
                loggingCritical(f"Error : each term of the sum '{expr}' must contain the same free indices.")
                return
            if name is None and set(freeInds) != set(Linds):
                # print(freeInds)
                # print(inds)
                loggingCritical(f"Error in term {term}: the free indices must be identical to those in the definition '{name}'.")
                return

            # Now that the term is validated, construct the resulting tensor object
            contractArgs = []
            
            for field in terms:
                # print("Field : ", field)
                if not isinstance(field, Symbol):
                    base, inds = field.base, field.indices
                else:
                    base, inds = field, []
                    
                tens = self.definitions[str(base)]
                tens.update(len(inds))
                
                inds = [Wild(str(el)) for el in inds]
                contractArgs.append(tens(*inds))
            
            freeDummies = [Wild(str(el)) for el in Linds]
            tmp = tensorContract(*contractArgs, value=coeff, freeDummies=freeDummies)
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
        
        # retTensor = 
        
        # print("Ind positions :", indRanges)
        ranges = []
        
        for freeInd in Linds:
            iRange = indRanges[freeInd]
            iRange = iRange[0].range[iRange[1]]
            ranges.append(iRange)
            
        return TensorObject(copy=(Lbase, ranges, rhsResult), fromDef=name, expr=expr)
            
    def expand(self):
        """ Performs a first level of expansion of the Lagrangian. More precisely, replaces 
        all the occurences of user-defined quantities with their expression."""
        loggingInfo("Expanding the Lagrangian ...")
        
        #Create an instance of the Lagrangian expand class
        # LagExp = ExpandClass(self)
        
        count = 0
        content = ()
        for couplingType, terms in self.potential.items():
            if couplingType in self.translateContent:
                self.dicToFill = self.translateDic(self.RGmodule)[couplingType]
                content = self.translateContent[couplingType]
                self.currentPotentialDic = self.potential[couplingType]
            else:
                continue
            
            for coupling, term in terms.items():
                TensorObject.globalTensorCount = 1
                # print("\n\n\t", coupling, " ; ", term, "\n")
                # expTerm, inds = self.replaceTensorObjects(term)
                
                # expTerm = self.expandTerm(expTerm, inds)
                parsedTerm = []
                expTerm = self.parseExpression(term, expandedTerm=parsedTerm).dic[()]
                # print(expTerm)
                # print(parsedTerm)
                # print('\n')
                
                self.expandedPotential[couplingType][coupling] = parsedTerm[0]
                self.fullyExpandedPotential[couplingType][coupling] = expTerm
                
                self.fillTensorDic(coupling, expTerm, content)
                
                count += 1
                print_progress(count, self.model.nCouplings, prefix=' '*4, bar_length=20, printTime=self.model.times, logProgress=True)
            
            # Finally, remove all vanishing elements from dict
            for k,v in list(self.dicToFill.items()):
                if v == 0:
                    self.dicToFill.pop(k)
        

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
            
            fermions = [self.model.allFermions[str(el)] for el in subTerm[1] if str(el) in self.model.allFermions ]
           
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
            sGen = [self.model.allScalars[str(el)][1].gen for el in subTerm[1] if str(el) in self.model.allScalars]
            
            if content == (2,1): #Yukawa
                if len(fermions) != 2 or len(scalars) != 1:
                    loggingCritical(f"Error in term {str(coupling)} : \n\tYukawa terms must contain exactly 2 fermions and 1 scalar.")
                    exit()
                tensorInds = tuple(scalars + fermions)
                coeff = subTerm[0] * 2/len(set(itertools.permutations(fermions, 2)))
                
                # Fermion1 = Fermion2 : the matrix is symmetric
                if self.allFermionsValues[fermions[0]][1] == self.allFermionsValues[fermions[1]][1]:
                    self.model.assumptions[str(coupling)]['symmetric'] = True
                # Fermion1 = Fermion2bar : the matrix is hermitian
                if self.allFermionsValues[fermions[0]][1] == self.allFermionsValues[self.antiFermionPos(fermions[1])][1]:
                    self.model.assumptions[str(coupling)]['hermitian'] = True
                    
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
                
                # Fermion1 = Fermion2 : the matrix is symmetric
                if fermions[0] == fermions[1]:
                    self.model.assumptions[str(coupling)]['symmetric'] = True
                # Fermion1 = Fermion2bar : the matrix is hermitian
                if fermions[0] == self.antiFermionPos(fermions[1]):
                    self.model.assumptions[str(coupling)]['hermitian'] = True
                    
                assumptionDic = self.model.assumptions[str(coupling)]
                
                coupling = mSymbol(str(coupling), fGen[0], fGen[1], **assumptionDic)
                if coupling not in self.couplingStructure:
                    self.couplingStructure[str(coupling)] = (fGen[0], fGen[1])
                    
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
            
            # If Yukawa / Fermion mass, add the hermitian conjugate to Dic
            if content == (2,1) or content == (2,0):
                antiFermions = [self.antiFermionPos(f) for f in fermions]
                antiFermions.reverse()
                tensorInds = tuple(scalars + antiFermions + [True])
                coeff = conjugate(coeff)
                adjCoupling = Adjoint(coupling).doit()
                
                if tensorInds not in self.dicToFill:
                    self.dicToFill[tensorInds] = adjCoupling*coeff
                else:
                    self.dicToFill[tensorInds] += adjCoupling*coeff
    
            
    def getFermionShape(self, i, j):
        return (lambda l: (l[i][1].gen, l[j][1].gen))(self.allFermionsValues)
    
    def antiFermionPos(self, pos):
        aux = self.allFermionsValues[pos][0:2]
        if aux[1].conj:
            return aux[0] - len(self.model.allFermions)//2
        return aux[0] + len(self.model.allFermions)//2