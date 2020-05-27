# -*- coding: utf-8 -*-
from sys import exit
import copy
import re as reg
import itertools

from sympy import (DiagonalMatrix, Indexed, IndexedBase, Integer, Matrix, Mul, Pow,
                   Rational, SparseMatrix, Symbol, eye, flatten, lambdify, pi,
                   simplify, sqrt, zeros)

from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication


from Logging import loggingInfo, loggingCritical, print_progress

from Particles import Particle, ComplexScalar

from Lagrangian import Lagrangian

from Substitutions import getSubstitutions, doSubstitutions

from BetaFunctions import (GaugeBetaFunction, YukawaBetaFunction, QuarticBetaFunction,
                           TrilinearBetaFunction, FermionMassBetaFunction, ScalarMassBetaFunction,
                           FermionAnomalous, ScalarAnomalous, VevBetaFunction)

from Definitions import GaugeGroup, Identity, expand


class Model(object):
    """Takes the settings read from the YAML file and eventually filtered in RGEs.py and construct the class Model out of it incorporating the groups"""

    def __init__(self, settings, runSettings, idb, realBasis='all'):
        ###############
        # Definitions #
        ###############
        self._Name = settings['Name'].replace(' ', '_').replace('/', '_').replace('\\', '_')
        self._Author = settings['Author']
        self._Date = settings['Date']
        self.times = runSettings['PrintComputationTimes']
        self.saveSettings = copy.deepcopy(settings)
        self.runSettings = runSettings

        # Declare an interactive db access object
        self.idb = idb

        self.loopDic = {}

        self.validateSettings(settings, runSettings)

        loggingInfo("Loading the model ...", end=' ')

        ####################
        # Get gauge groups #
        ####################

        self.gaugeGroups = {}
        self.gaugeGroupsList = []
        self.UgaugeGroups = []

        self.realBasis = runSettings['RealBasis']
        self.getGaugegroups(settings, realBasis=self.realBasis)

        # Number of U1 gauge factors
        self.nU = [g.abelian for g in self.gaugeGroupsList].count(True)
        self.kinMix = (self.nU > 1 and runSettings['NoKinMix'] is False)

        #################
        # Get particles #
        #################

        self.Particles = {}
        self.Fermions = {}
        self.Scalars = {}
        self.ComplexScalars = {}

        #The following dicts contain all the components of the fields
        self.allFermions = {}
        self.allScalars = {}

        self.symbolicGen = False
        self.getParticles(settings)

        ######################
        # Read the potential #
        ######################

        self.potential = settings['Potential']
        self.assumptions = {}

        self.getAssumptions()

        #Read the vevs + possible gauge fixing
        self.vevs = {}
        self.getVevs(settings)

        self.gaugeFixing = None
        if 'GaugeParameter' in settings:
            self.gaugeFixing = self.parseMathExpr(settings['GaugeParameter'], real=True)

        #Read the anomalous dimensions
        self.scalarAnomalous = {}
        self.fermionAnomalous = {}

        self.getAnomalous(settings)

        #Identify the various couplings of the model
        self.allCouplings = {}
        self.couplingsPos = {'GaugeCouplings': {}}
        self.YukPos = {}

        self.ExplicitMatrices = []

        self.couplingStructure = {}

        self.gaugeCouplings = []
        for i, gp in enumerate(self.gaugeGroupsList):
            self.allCouplings[str(gp.g)] = ('GaugeCouplings', gp.g)
            self.couplingsPos['GaugeCouplings'][str(gp.g)] = i
            self.gaugeCouplings.append(str(gp.g))

        self.mixedGaugeCouplings = []

        self.upper = True
        if self.kinMix:
            self.kinMat = zeros(self.nU)
            for i in range(self.nU):
                for j in range(self.nU):
                    if (self.upper and i < j) or (not self.upper and i > j):
                        c = 'g_'+str(i+1)+str(j+1)
                        pos = [self.gaugeGroupsList.index(self.UgaugeGroups[k]) for k in (i,j)]
                        self.allCouplings[c] = ('GaugeCouplings', Symbol(c, real=True))
                        self.couplingsPos['GaugeCouplings'][c] = max(pos) + .5

                        self.kinMat[i,j] = Symbol(c, real=True)

                        self.mixedGaugeCouplings.append(self.kinMat[i,j])
                        self.gaugeCouplings.append(c)
                    elif i == j:
                        self.kinMat[i,j] = self.UgaugeGroups[i].g

            self.kinMat2 = self.kinMat*self.kinMat.transpose()


        # Fill the dics related to the various couplings of the model
        self.nCouplings = 0
        for couplingType, terms in self.potential.items():
            if couplingType == 'Definitions':
                continue
            tKeys = list(terms.keys())
            for coupling in terms:
                self.nCouplings += 1

                # Add the various couplings to the Couplings dictionnary
                if couplingType == 'Yukawas':
                    self.YukPos[coupling] = tKeys.index(coupling)
                if couplingType == 'FermionMasses':
                    self.YukPos[coupling] = 100 + tKeys.index(coupling)

                if couplingType not in self.couplingsPos:
                    self.couplingsPos[couplingType] = {}

                self.couplingsPos[couplingType][coupling] = tKeys.index(coupling)

                self.allCouplings[coupling] = couplingType


        if self.vevs != {}:
            self.couplingsPos['Vevs'] = {}
            for i, (k,v) in enumerate(self.vevs.items()):
                self.allCouplings[k] = ('Vevs', v[1])
                self.couplingsPos['Vevs'][k] = i


        # Read the substitutions
        self.substitutions = {}
        self.gutNorm = {}
        if 'Substitutions' in settings and settings['Substitutions'] != {}:
           self.substitutions = getSubstitutions(self, settings['Substitutions'] if 'Substitutions' in settings else {})


        # Read the beta-factor
        if 'BetaFactor' in settings:
            if type(settings['BetaFactor']) not in (list, tuple):
                self.betaFactor = self.parseMathExpr(settings['BetaFactor'])
                self.betaExponent = lambda n: 2*n
            else:
                self.betaFactor = self.parseMathExpr(settings['BetaFactor'][0])
                self.betaExponent = self.parseMathExpr(settings['BetaFactor'][1])
                if self.betaExponent.find('n') == set():
                    loggingCritical("Error : the beta-exponent must be an integer function of 'n'. Setting it to default (2*n).")
                    self.betaExponent = lambda n: 2*n
                else:
                    lambdaExponent = lambdify(Symbol('n'), self.betaExponent)
                    self.betaExponent = lambda n: lambdaExponent(n)
            if self.betaFactor == 0:
                loggingCritical("Error : beta-factor cannot be 0. Exiting.")
                exit()
        else:
            self.betaFactor = Integer(1)
            self.betaExponent = lambda n: Integer(2)*n

        self.translateContent = {'GaugeCouplings': (0,0),
                                 'Yukawas': (2,1),
                                 'QuarticTerms': (0,4),
                                 'TrilinearTerms' : (0,3),
                                 'ScalarMasses': (0,2),
                                 'FermionMasses': (2,0),
                                 'FermionAnomalous': (2,0),
                                 'ScalarAnomalous': (0,2),
                                 'Vevs': (0,1)}

        self.translateDic = lambda RGmodule: {'Yukawas': RGmodule.YDic,
                                              'QuarticTerms': RGmodule.LambdaDic,
                                              'TrilinearTerms' : RGmodule.Hdic,
                                              'ScalarMasses': RGmodule.MSdic,
                                              'FermionMasses': RGmodule.MFdic,
                                              'FermionAnomalous': RGmodule.gammaFdic,
                                              'ScalarAnomalous': RGmodule.gammaSdic,
                                              'Vevs': RGmodule.Vdic}

        self.translateBetaFunction = {'GaugeCouplings': GaugeBetaFunction,
                                      'Yukawas': YukawaBetaFunction,
                                      'QuarticTerms': QuarticBetaFunction,
                                      'TrilinearTerms' : TrilinearBetaFunction,
                                      'ScalarMasses': ScalarMassBetaFunction,
                                      'FermionMasses': FermionMassBetaFunction,
                                      'FermionAnomalous': FermionAnomalous,
                                      'ScalarAnomalous': ScalarAnomalous,
                                      'Vevs': VevBetaFunction}

        self.lagrangianMapping = {}
        self.toCalculate = {}

        self.RGclasses = {}
        self.allRGEs = {}
        self.couplingRGEs = {}

        self.NonZeroCouplingRGEs = {}
        self.NonZeroDiagRGEs = {}


    def validateSettings(self, settings, runSettings):
        """Implements the different checks carried out on the input provided by the user"""

        ########################################
        # Check the gauge groups and particles #
        ########################################

        if not 'Groups' in settings:
            loggingCritical("Error : No gauge groups specified. Exiting.")
            exit()
        else:
            groups = settings['Groups'].keys()

        allParticles = {}

        if 'Fermions' in settings and settings['Fermions'] != {}:
            for k,v in settings['Fermions'].items():
                if k not in allParticles:
                    allParticles[k] = v
                else:
                    loggingCritical(f"Error : Particle '{k}' cannot be defined twice. Please check the model file.")
                    exit()
        if 'RealScalars' in settings:
            for k,v in settings['RealScalars'].items():
                if k not in allParticles:
                    allParticles[k] = v
                else:
                    loggingCritical(f"Error : Particle '{k}' cannot be defined twice. Please check the model file.")
                    exit()
        if 'ComplexScalars' in settings:
            for k,v in settings['ComplexScalars'].items():
                twice = []

                for f in v['RealFields']:
                    if '*' in f or '+' in f or '-' in f:
                        loggingCritical(f"Error : Invalid field name '{f}' in RealScalars of particle '{k}'. Exiting")
                        exit()
                    if f in allParticles:
                        twice.append(f)
                    else:
                        allParticles[f] = None

                if k in allParticles:
                    twice.append(k)

                if twice != []:
                    for el in twice:
                        loggingCritical(f"Error : Particle '{el}' cannot be defined twice. Please check the model file.")
                    exit()

                allParticles[k] = v

        # Check that all the gauge groups are defined above
        for part, val in allParticles.items():
            if val is None:
                continue
            if 'Qnb' in val:
                tags = val['Qnb'].keys()
            else:
                tags = val.keys()
            if not all([el in groups for el in tags]):
                loggingCritical(f"Error : the particle '{part}' is charged under an unknown gauge group.")
                exit()
        if not 'Potential' in settings:
            settings['Potential'] = {}
            self.saveSettings['Potential'] = {}


        ################
        # RUN settings #
        ################

        if 'Loops' in runSettings:
            maxLoops = {'GaugeCouplings': 3,
                        'Yukawas': 2,
                        'QuarticTerms': 2,
                        'TrilinearTerms' : 2,
                        'ScalarMasses': 2,
                        'FermionMasses': 2,
                        'Vevs': 2}

            if type(runSettings['Loops']) == str and runSettings['Loops'].lower() == 'max':
                loops = 'max'
            else:
                try:
                    loops = eval(str(runSettings['Loops']))
                except:
                    loops = str(runSettings['Loops'])

            if type(loops) == int:
                self.nLoops = 7*[loops]

                self.loopDic['GaugeCouplings'] = self.nLoops[0]
                self.loopDic['Yukawas'] = self.nLoops[1]
                self.loopDic['QuarticTerms'] = self.nLoops[2]

                self.loopDic['TrilinearTerms'] = self.nLoops[3]
                self.loopDic['ScalarMasses'] = self.nLoops[4]
                self.loopDic['FermionMasses'] = self.nLoops[5]

                self.loopDic['Vevs'] = self.nLoops[6]

            elif type(loops) == list and len(loops) == 3:
                self.nLoops = loops

                self.loopDic['GaugeCouplings'] = self.nLoops[0]
                self.loopDic['Yukawas'] = self.nLoops[1]
                self.loopDic['QuarticTerms'] = self.nLoops[2]

                self.loopDic['FermionMasses'] = self.loopDic['Yukawas']
                self.loopDic['TrilinearTerms'] = self.loopDic['QuarticTerms']
                self.loopDic['ScalarMasses'] = self.loopDic['QuarticTerms']

                self.loopDic['Vevs'] = self.loopDic['QuarticTerms']

            elif type(loops) == list and len(loops) == 6:
                self.nLoops = loops

                self.loopDic['GaugeCouplings'] = self.nLoops[0]
                self.loopDic['Yukawas'] = self.nLoops[1]
                self.loopDic['QuarticTerms'] = self.nLoops[2]

                self.loopDic['TrilinearTerms'] = self.nLoops[3]
                self.loopDic['ScalarMasses'] = self.nLoops[4]
                self.loopDic['FermionMasses'] = self.nLoops[5]

                self.loopDic['Vevs'] = self.loopDic['QuarticTerms']

            elif type(loops) == list and len(loops) == 7:
                self.nLoops = loops

                self.loopDic['GaugeCouplings'] = self.nLoops[0]
                self.loopDic['Yukawas'] = self.nLoops[1]
                self.loopDic['QuarticTerms'] = self.nLoops[2]

                self.loopDic['TrilinearTerms'] = self.nLoops[3]
                self.loopDic['ScalarMasses'] = self.nLoops[4]
                self.loopDic['FermionMasses'] = self.nLoops[5]

                self.loopDic['Vevs'] = self.nLoops[6]
            elif type(loops) == str and loops == 'max':
                self.nLoops = []
                for k,v in maxLoops.items():
                    self.nLoops.append(v)
                    self.loopDic[k] = v
            else:
                loggingCritical("Error : Loops should be in one of the following forms :\n" +
                                "\t- A single integer\n" +
                                "\t- A list of three, six or seven integers\n" +
                                "\t- The keyword 'max'")
                exit()

            # Nothing to calculate ?
            if all([el == 0 for el in self.nLoops]):
                loggingCritical("Nothing to calculate ! Exiting.")
                exit()

            # If loop orders are too high, set them to the max allowed value
            for k,v in maxLoops.items():
                if self.loopDic[k] > v:
                    loggingInfo(f"Warning : Loop level for '{k}' is too high ({self.loopDic[k]}). Setting it to {v}")
                    self.loopDic[k] = v

            # Anomalous
            self.loopDic['ScalarAnomalous'] = self.loopDic['QuarticTerms']
            self.loopDic['FermionAnomalous'] = self.loopDic['Yukawas']

    def getGaugegroups(self, settings, realBasis=None):
        """Create the different gauge groups"""

        GaugeGroup.realBasis = realBasis
        for gpName, gp in settings['Groups'].items():
            self.gaugeGroups[gpName] = GaugeGroup(gpName, gp, self.idb)

        self.gaugeGroupsList = list(self.gaugeGroups.values())
        self.UgaugeGroups = [g for g in self.gaugeGroupsList if g.abelian]

    def getParticles(self, settings):
        for key, value in settings.items():
            if key == 'Fermions':
                self.Fermions = value
                antiFermions = {}
                # Create the particle and store it in Fermions
                for part, val in value.items():
                    self.Fermions[part] = Particle(part, val, self.gaugeGroups, self.idb)
                    antiFermions[part+'bar'] = self.Fermions[part].antiParticle()

                self.Fermions.update(antiFermions)
            elif key == 'RealScalars':
                # Copy the particles in the class
                for part, qnb in value.items():
                    if 'Gen' not in qnb:
                        Qnb = {'Gen': 1, 'Qnb': qnb}
                    else :
                        Qnb = qnb
                    self.Scalars[part] = Particle(part, Qnb, self.gaugeGroups, self.idb)
            elif key == 'Potential':
                self.potential = value

        self.Particles.update(self.Fermions)
        self.Particles.update(self.Scalars)

        # Now that the Real Scalars have been created we can create the Cplx one associated
        if 'ComplexScalars' in settings:
            for part, setts in settings['ComplexScalars'].items():
                setts['Norm'] = self.parseMathExpr(setts['Norm'])
                if 'Gen' not in setts:
                    setts['Gen'] = 1

                self.ComplexScalars[part] = ComplexScalar(part, setts, self.gaugeGroups, self.idb)
                self.ComplexScalars[part+'bar'] = self.ComplexScalars[part].antiParticle()

                for r in self.ComplexScalars[part].realFields:
                    self.Scalars[str(r)] = r

        self.Particles.update(self.ComplexScalars)

        nF = 0
        for fName, f in self.Fermions.items():
            ranges = [r for r in f.indicesRange.values()]
            nonNullRanges = [r for r in ranges if r != 0]
            if nonNullRanges == []: #Singlet
                tup = [nF, f, tuple([-1]*len(ranges))]
                nF += 1
                self.allFermions[fName] = tuple(tup)
            else :
                for el in itertools.product(*[(list(range(r)) if r != 0 else [-1]) for r in ranges]):
                    tup = [nF, f, tuple(el), parse_expr(str(fName) + str([n for n in el if n != -1]), local_dict={str(f._name): IndexedBase(str(f._name))})]
                    nF += 1
                    self.allFermions[fName + str([n for n in el if n != -1])] = tuple(tup)

        nS = 0
        for sName, s in self.Scalars.items():
            ranges = [r for r in s.indicesRange.values()]
            nonNullRanges = [r for r in ranges if r != 0]
            if nonNullRanges == []: #Singlet
                self.allScalars[sName] = (nS, s, tuple([-1]*len(ranges)))
                nS += 1
            else :
                for el in itertools.product(*[(list(range(r)) if r != 0 else [-1]) for r in ranges]):
                    tup = [nS, s, tuple(el), parse_expr(str(sName) + str([n for n in el if n != -1]), local_dict={str(s._name): IndexedBase(str(s._name))})]
                    self.allScalars[sName + str([n for n in el if n != -1])] = tuple(tup)
                    nS += 1

        self.symbolicGen = any([isinstance(p.gen, Symbol) for p in self.Particles.values()])

    def getVevs(self, settings):
        if 'Vevs' in settings and settings['Vevs'] != {}:
            for k, v in settings['Vevs'].items():
                if '[' in v and ']' in v:
                    try:
                        field = v[:v.find('[')]
                        inds = eval('('+v[v.find('[')+1:v.find(']')]+',)')
                    except:
                        loggingCritical("Error while reading the vev '" + k + "'. Skipping.")
                        continue
                else:
                    field = v
                    inds = tuple()

                if field not in self.Scalars:
                    loggingCritical("Error while reading the vev '" + k + "' : scalar '" + field + "' is unkown. Skipping")
                    continue

                field = self.Scalars[field]
                ni = len(field.indexStructure)

                if len(inds) != ni or any([i<1 or i>field.indexStructure[pos] for pos, i in enumerate(inds)]):
                    loggingCritical("Error while reading the vev '" + k + "' : " +
                                    "scalar '" + str(field) + "' should have exactly " +
                                    str(ni) + (" indices with ranges " if ni>1 else " index with range ") +
                                    str(field.indexStructure) + ". Skipping")
                    continue

                vSymb = Symbol(k, real=True)
                inds = tuple([i-1 for i in inds])

                scalarStr = str(field)
                if inds != tuple():
                    scalarStr += str(inds).replace('(', '[').replace(')', ']').replace(',]', ']')

                # Now determine whether the scalar is a real/imaginary part of a cplx field
                fromCplx = field.fromCplx

                if fromCplx != False:
                    part = fromCplx.realFields.index(field)
                    self.vevs[k] = (self.allScalars[scalarStr][0], vSymb, scalarStr, fromCplx, part)
                else:
                    self.vevs[k] = (self.allScalars[scalarStr][0], vSymb, scalarStr)


    def getAnomalous(self, settings):
        allAnomalous = lambda dic: (len(dic) == 1 and 'All' in dic and dic['All'] is None)

        if 'ScalarAnomalous' in settings and settings['ScalarAnomalous'] != {} and self.loopDic['ScalarAnomalous'] > 0:
            sa = settings['ScalarAnomalous']

            if not allAnomalous(sa):
                for k in sa:
                    if not (k[0] == '(' and k[-1] == ')'):
                        loggingCritical("Warning in ScalarAnomalous : the correct syntax is '(scalar1, scalar2)'.\n\
                                         Please check the term '" + k.replace(';', ', ') + "'. Skipping.")
                        continue

                    fields = k[1:-1].split(';')

                    if len(fields) != 2:
                        loggingCritical("Warning in ScalarAnomalous : the correct syntax is '(scalar1, scalar2)'.\n\
                                         Please check the term '" + k.replace(';', ', ') + "'. Skipping.")
                        continue

                    skip = False
                    for i, f in enumerate(fields):
                        if '[' not in f:
                            continue

                        base = f[:f.find('[')]
                        inds = f[f.find('['):]

                        try:
                            inds = str([el-1 for el in eval(inds)])
                            fields[i] = base + inds
                        except:
                            skip = "unable to read the term '" + k.replace(';', ', ') + "'"
                            break

                        if fields[i] not in self.allScalars:
                            skip = "in term '" + k.replace(';', ', ') + "', scalar '" + f + "' is unknown"
                            break

                    if skip:
                        loggingCritical("Warning in ScalarAnomalous : " + skip + ". Skipping.")
                        continue

                    key = [self.allScalars[s][(3 if len(self.allScalars[s]) > 3 else 1)] for s in fields]
                    key = tuple([(k if isinstance(k, Indexed) else k._name) for k in key])
                    val = tuple([self.allScalars[s][0] for s in fields])

                    self.scalarAnomalous[key] = val
            else:
                # All anomalous
                self.saveSettings['ScalarAnomalous'] = 'All'

                for i, s1 in enumerate(self.allScalars.values()):
                    for j, s2 in enumerate(self.allScalars.values()):
                        if i>j:
                            continue

                        key = [s[(3 if len(s) > 3 else 1)] for s in (s1, s2)]
                        key = tuple([(k if isinstance(k, Indexed) else k._name) for k in key])
                        val = tuple([s[0] for s in (s1, s2)])

                        self.scalarAnomalous[key] = val


        if 'FermionAnomalous' in settings and settings['FermionAnomalous'] != {} and self.loopDic['FermionAnomalous'] > 0:
            fa = settings['FermionAnomalous']

            if not allAnomalous(fa):
                for k in fa:
                    if not (k[0] == '(' and k[-1] == ')'):
                        loggingCritical("Warning in FermionAnomalous : the correct syntax is '(fermion1, fermion2)'.\n\
                                         Please check the term '" + k.replace(';', ', ') + "'. Skipping.")
                        continue

                    fields = k[1:-1].split(';')

                    if len(fields) != 2:
                        loggingCritical("Warning in FermionAnomalous : the correct syntax is '(fermion1, fermion2)'.\n\
                                         Please check the term '" + k.replace(';', ', ') + "'. Skipping.")
                        continue

                    skip = False
                    for i, f in enumerate(fields):
                        if '[' not in f:
                            continue

                        base = f[:f.find('[')]
                        inds = f[f.find('['):]

                        try:
                            inds = str([el-1 for el in eval(inds)])
                            fields[i] = base + inds
                        except:
                            skip = "unable to read the term '" + k.replace(';', ', ') + "'"
                            break

                        if fields[i] not in self.allFermions:
                            skip = "in term '" + k.replace(';', ', ') + "', fermion '" + f + "' is unknown"
                            break

                    if skip:
                        loggingCritical("Warning in FermionAnomalous : " + skip + ". Skipping.")
                        continue

                    key = [self.allFermions[f][(3 if len(self.allFermions[f]) > 3 else 1)] for f in fields]
                    key = tuple([(k if isinstance(k, Indexed) else k._name) for k in key])
                    val = tuple([self.allFermions[f][0] for f in fields])

                    self.fermionAnomalous[key] = val
            else:
                # All anomalous
                self.saveSettings['FermionAnomalous'] = 'All'

                for i, f1 in enumerate(self.allFermions.values()):
                    if f1[1].conj:
                        continue
                    for j, f2 in enumerate(self.allFermions.values()):
                        if f2[1].conj or i>j:
                            continue

                        key = [f[(3 if len(f) > 3 else 1)] for f in (f1, f2)]
                        key = tuple([(k if isinstance(k, Indexed) else k._name) for k in key])
                        val = tuple([f[0] for f in (f1, f2)])

                        self.fermionAnomalous[key] = val


    def parseMathExpr(self, expr, real=False):
        if type(expr) != str:
            expr = Rational(expr)
            return expr
        else:
            if real:
                assumptions = {'real': True}
            else:
                assumptions = {'complex': True}

            expr = expr.replace('Sqrt', 'sqrt').replace('abs', 'Abs').replace('\t', '\\t').replace('\n', '\\n')

            #'lambda' -> 'Lambda' workaround
            expr = expr.replace('lambda', 'pyLambda')

            # First, try to see if there are newly defined symbols ( 'symb' ) in the string

            # Primes inside { } are replaced by ":
            src = reg.findall(r'{.+?}', expr)
            for el in src:
                expr = expr.replace(el, el.replace("'", '"'))

            #Find all symbols, i.e. expressions inside ' '
            symbs = {}
            symbs['pyLambda'] = Symbol('lambda')
            src = reg.findall(r"'.+?'", expr)
            for i,el in enumerate(src):
                expr =  expr.replace(el, 'PySymb'+str(i))
                symbs['PySymb'+str(i)] = Symbol(el.replace("'", '').replace('"', "\'"), **assumptions)

            return parse_expr(expr, local_dict=symbs,
                              transformations=standard_transformations[1:] + (implicit_multiplication,))

    def getAssumptions(self):
        for cType, terms in self.potential.items():
            if cType == 'Definitions':
                continue

            newKeys = []
            for k,v in terms.items():
                assumptions = {}
                invalidAssumptions = []

                if cType == 'ScalarMasses':
                    assumptions = {'squared': False}
                    if type(v) != tuple:
                        v = (v,)

                if type(v) == tuple:
                    if 'real' in v:
                        assumptions['real'] = True
                    if 'symmetric' in v:
                        assumptions['symmetric'] = True
                    if 'hermitian' in v:
                        assumptions['hermitian'] = True
                    if 'unitary' in v:
                        assumptions['unitary'] = True

                    if cType == 'ScalarMasses':
                        if 'squared' in v:
                            assumptions['squared'] = True

                    invalidAssumptions = [el for el in v[1:] if el not in assumptions]

                if assumptions != {} or invalidAssumptions != []:
                    self.potential[cType][k] = v[0]

                    for el in invalidAssumptions:
                        loggingCritical("Warning : assumption '" +  el + "' is not understood. Ignoring it.")

                self.assumptions[k] = assumptions
                newKeys.append(k)
            self.potential[cType] = {k:v for k,v in zip(newKeys, terms.values())}


    def expandLagrangian(self, RGmodule):
        self.lagrangian = Lagrangian(self.saveSettings, self, RGmodule)
        loggingInfo("Done.")

        loggingInfo("Expanding the Lagrangian ...")
        self.lagrangian.expand()

        self.expandedPotential = self.lagrangian.expandedPotential

        # If any matrix substitution is provided, check now that the shapes correspond.
        # This is to prevent the computation from starting if not.
        # Also, if the matrix satisfies Y = Diag(y), replace it by y*Identity(nG)

        if 'yukMat' in self.substitutions:
            for k, v in self.substitutions['yukMat'].items():
                shape = tuple([el for el in self.couplingStructure[k] if type(el) != bool])

                if not isinstance(v[1], DiagonalMatrix):
                    if v[1].shape != shape:
                        loggingCritical("Error : The shape of the matrix " + k + " given in Substitutions" +
                                        " should be " + str(shape))
                        exit()
                else:
                    if shape[0] != shape[1]:
                        loggingCritical("Error in Substitutions : the 'diag' keyword cannot be used for" +
                                        " the rectangular matrix '" + k + "'")
                        exit()

                    cType, diagMat = self.substitutions['yukMat'][k]
                    self.substitutions['yukMat'][k] = (cType, diagMat.arg*Identity(shape[0]))


        # VeVs
        if self.vevs != {}:
            for k,v in self.vevs.items():
                RGmodule.Vdic[(v[0],)] = v[1]

        # Anomalous dimensions
        if self.fermionAnomalous != {}:
            for k,v in self.fermionAnomalous.items():
                RGmodule.gammaFdic[v] = k

        if self.scalarAnomalous != {}:
            for k,v in self.scalarAnomalous.items():
                RGmodule.gammaSdic[v] = k


    def doSubstitutions(self):
        loggingInfo("Applying substitutions ...")
        doSubstitutions(self, self.substitutions)

    ###################################################
    # Map the model's Lagrangian onto the general one #
    ###################################################

    def constructMapping(self, RGmodule):
        loggingInfo("Mapping the model onto the general Lagrangian ...")

        #Gauge couplings mapping, taking into account possible kinetic mixing
        noMix = {}
        mix = {}
        alreadyTaken = set()

        for el in itertools.combinations_with_replacement(range(RGmodule.nGi), 2):
            A,B = [RGmodule.gi[i] for i in el]
            c = RGmodule.G_(A,B)
            if c != 0 and c not in alreadyTaken:
                dic = noMix if A == B else mix

                if not self.upper:
                    A,B = B,A

                dic[(A,B)] = len(dic)
                alreadyTaken.add(c)

        newInds = {**noMix, **{k:v+len(noMix) for k,v in mix.items()}}
        gaugeMatrix = zeros(len(newInds))

        def delta(A,B):
            if A == B:
                return 1
            return 0

        def G(A,B):
            if RGmodule.G_(A,B) == 0:
                return 0
            if not self.kinMix or A not in RGmodule.Ugauge or B not in RGmodule.Ugauge:
                return sqrt(RGmodule.G_(A,B)).args[0]

            i,j = RGmodule.Ugauge.index(A), RGmodule.Ugauge.index(B)
            return self.kinMat[i,j]

        for (A,B), X in newInds.items():
            for (C,D), Y in newInds.items():
                gaugeMatrix[X,Y] = G(B,D)*delta(A,C) + G(A,D)*delta(B,C)

        gaugeMatrix = simplify(gaugeMatrix.inv())

        couplingType = 'GaugeCouplings'
        self.potential[couplingType] = {}
        for c in self.gaugeCouplings:
            self.potential[couplingType][c] = 0

        self.lagrangianMapping[couplingType] = gaugeMatrix * self.betaFactor
        self.toCalculate[couplingType] = list(newInds.keys())


        count = 0
        translation = self.translateDic(RGmodule)
        for couplingType in self.potential:
            if couplingType == 'Definitions':
                continue
            if ( couplingType in translation and self.potential[couplingType] != {}
             and translation[couplingType] != {}):
                coeffList = []
                dicList = []
                mappingMatrix = []
                sortedList = []

                coeffList = [c for c in self.potential[couplingType].keys()]

                mappingMatrix = SparseMatrix(len(coeffList), len(coeffList), 0)
                sortedList = sorted([(key, val) for key, val in translation[couplingType].items() if not(type(key[-1]) == bool and key[-1] == True)],
                                     key=lambda x: (len(set(x[0])), len(x[1].as_coeff_add()[1]), x[0]))

                trys = 0
                for el in sortedList:
                    trys += 1
                    matTry = self.fillMappingMatrix(mappingMatrix, coeffList, el)
                    if(matTry.rank() > mappingMatrix.rank()):
                        mappingMatrix = matTry
                        dicList.append(el[0])

                        count += 1
                        print_progress(count, self.nCouplings, prefix=' '*4, bar_length=20, printTime=self.times, logProgress=True)

                    if matTry.rank() == len(coeffList):
                        break

                try:
                    self.lagrangianMapping[couplingType] = Matrix(mappingMatrix).inv() * self.betaFactor
                except:
                    # from sympy import pretty
                    loggingCritical("\nError in Lagrangian mapping : matrix of couplings is not invertible.")
                    loggingCritical("\tCoupling type : " + couplingType)
                    # loggingCritical("\t\t" + pretty(mappingMatrix).replace("\n", "\n\t\t"))
                    exit()

                self.toCalculate[couplingType] = dicList


        # Add vevs and anomalous dimensions by hand (not related to the Lagrangian)
        if self.vevs != {}:
            couplingType = 'Vevs'

            self.potential[couplingType] = {}
            for c in self.vevs:
                self.potential[couplingType][c] = 0
            self.lagrangianMapping[couplingType] = eye(len(self.vevs))*self.betaFactor
            self.toCalculate[couplingType] = list(RGmodule.Vdic.keys())

        if self.fermionAnomalous != {}:
            couplingType = 'FermionAnomalous'

            self.potential[couplingType] = {}
            for c in self.fermionAnomalous:
                self.potential[couplingType][c] = 0
            self.lagrangianMapping[couplingType] = eye(len(self.fermionAnomalous))
            self.toCalculate[couplingType] = list(RGmodule.gammaFdic.keys())

        if self.scalarAnomalous != {}:
            couplingType = 'ScalarAnomalous'

            self.potential[couplingType] = {}
            for c in self.scalarAnomalous:
                self.potential[couplingType][c] = 0
            self.lagrangianMapping[couplingType] = eye(len(self.scalarAnomalous))
            self.toCalculate[couplingType] = list(RGmodule.gammaSdic.keys())


    def fillMappingMatrix(self, mappingMatrix, coeffList, newTerm):
        newMat = copy.deepcopy(mappingMatrix)
        j = mappingMatrix.rank()

        pos = []

        for subTerm in newTerm[1].as_coeff_add()[1]:
            splitTerm = flatten(subTerm.as_coeff_mul())
            numeric = [x for x in splitTerm if x.is_number]
            coeff = str([x for x in splitTerm if not x in numeric][0])
            pos.append((coeffList.index(coeff), Mul(*numeric)))

        for el in pos:
            newMat[j, el[0]] = el[1]

        return newMat


    ##################
    # Beta-functions #
    ##################

    def defineBetaFunctions(self, RGmodule):
        for couplingType in self.toCalculate:
            RGclass = self.translateBetaFunction[couplingType]
            content = self.translateContent[couplingType]
            self.RGclasses[couplingType] = RGclass(self, RGmodule, content, self.loopDic[couplingType])

            self.allRGEs[couplingType] = {}
            self.couplingRGEs[couplingType] = {}
            self.NonZeroCouplingRGEs[couplingType] = {}

            for i in range(self.loopDic[couplingType]):
                self.allRGEs[couplingType][i] = []
                self.couplingRGEs[couplingType][i] = {}
                self.NonZeroCouplingRGEs[couplingType][i] = {}

    def computeBetaFunctions(self):
        loggingInfo("Computing the RGES ...")
        for couplingType, terms in self.toCalculate.items():
            if self.loopDic[couplingType] == 0:
                continue

            loggingInfo("     -> " + couplingType)
            for n in range(self.loopDic[couplingType]):
                loggingInfo("         -> " + str(n+1) + "-loop")
                print_progress(0, len(terms), prefix=' '*8, bar_length=10, printTime=self.times)
                for i, term in enumerate(terms):
                    self.allRGEs[couplingType][n].append(self.RGclasses[couplingType].compute(*term, nLoops=n))
                    print_progress(i+1, len(terms), prefix=' '*8, bar_length=10, printTime=self.times, logProgress=True)

        loggingInfo("    ... Done")

    def mapBetaFunctions(self):
        loggingInfo("Re-combining the RGES ...")

        #This is for progress bar
        nTot = 0
        count = 0
        for couplingType, RGlist in self.allRGEs.items():
            nTot += len(self.potential[couplingType])*self.loopDic[couplingType]


        for couplingType, RGloops in self.allRGEs.items():
            mat = self.lagrangianMapping[couplingType]
            for n, RGlist in RGloops.items():
                couplingRGEs = mat*Matrix(RGlist)

                # Take into account the beta-exponent
                expFactor = 1
                if 'Anomalous' not in couplingType:
                    exponent = self.betaExponent(n+1) - 2*(n+1)
                    if exponent != 0:
                        expFactor = Pow(4*pi, exponent)

                for pos, coupling in enumerate(list(self.potential[couplingType])):
                    try:
                        self.couplingRGEs[couplingType][n][coupling] = expand(couplingRGEs[pos]*expFactor)
                    except BaseException as e:
                        loggingCritical(f"Error expanding term at : {couplingType}, {n}, {pos}")
                        loggingCritical(e)
                        exit()
                    count += 1
                    print_progress(count, nTot, prefix='    ', bar_length=20, printTime=self.times)
