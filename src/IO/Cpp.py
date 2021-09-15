# -*- coding: utf-8 -*-
import os

from sympy.printing.cxx import CXX11CodePrinter
from sympy import Abs, I, Mul, Symbol, conjugate

from Definitions import splitPow, trace, Trace, sortYukTrace, mSymbol

from Logging import loggingCritical, loggingInfo

class CppExport():
    def __init__(self, model, latexSubs={}):
        self._Name = model._Name.replace('-', '').replace('+', '')
        if self._Name[0].isdigit():
            self._Name = '_' + self._Name
        self.model = model

        # BetaFunc definition
        self.betaFactor = model.betaFactor
        self.betaExponent = str(model.betaExponent(Symbol('n')))

        self.translation = {'GaugeCouplings': 'Gauge Couplings',
                            'Yukawas': 'Yukawa Couplings',
                            'QuarticTerms': 'Quartic Couplings',
                            'TrilinearTerms' : 'Trilinear Couplings',
                            'ScalarMasses': 'Scalar Mass Couplings',
                            'FermionMasses': 'Fermion Mass Couplings',
                            'Vevs': 'Vacuum-expectation Values'}

        self.nLoopDic = {'GaugeCouplings': 'nGauge',
                         'Yukawas': 'nYuk',
                         'QuarticTerms': 'nQuartic',
                         'TrilinearTerms' : 'nQuartic',
                         'ScalarMasses': 'nQuartic',
                         'FermionMasses': 'nYuk',
                         'Vevs': 'nQuartic'}

        self.complexCouplings = set()
        self.couplingStructure = {ccode(model.allCouplings[k][1]): v for k,v in model.couplingStructure.items()}

        self.conjugatedCouplings = {}

        self.inconsistentRGset = (model.NonZeroCouplingRGEs != {} or model.NonZeroDiagRGEs != {})
        if self.inconsistentRGset:
            raise TypeError("The RGE set is inconsistent. Please refer to the latex output.")

        self.gaugeFixing = False

        # Fix the symbolic gen numbers
        self.symbolicGens = []
        self.genFix = ''
        for p in model.Particles.values():
            if isinstance(p.gen, Symbol):
                if p.gen not in self.symbolicGens:
                    self.symbolicGens.append(p.gen)

        if self.symbolicGens != []:
            self.genFix = ' = '.join([str(el) for el in self.symbolicGens]) + ' = 3'

        self.lightSolver = model.runSettings['LightCppSolver']
        self.generateCppFile(model)


    def write(self, path):
        self.storePath = path

        if not os.path.exists(os.path.join(path, 'PythonOutput')):
            os.makedirs(os.path.join(path, 'PythonOutput'))

        # First : write the C++ solver file
        fileName = os.path.join(path, 'PythonOutput', self._Name + '.cpp')
        try:
            self.file = open(fileName, 'w')
        except:
            loggingCritical('ERROR while creating the C++ output file. Skipping.')
            return

        self.file.write(self.allStr)
        self.file.close()

        # Then, create the makefile
        fileName = os.path.join(path, 'PythonOutput', 'Makefile')
        try:
            self.file = open(fileName, 'w')
            self.file.write(self.makeFileString())
        except:
            loggingCritical('ERROR while creating the Makefile. Skipping.')
            return
        self.file.close()


        # Optional: write the light Cpp output
        if self.lightSolver:
            fileName = os.path.join(self.lightSolver, 'RGEsolver.cpp')
            try:
                self.file = open(fileName, 'w')
            except:
                loggingCritical('ERROR while creating the light C++ output file. Skipping.')
                return

            self.file.write(self.lallStr)
            self.file.close()



        # # If the cpplist argument was given, export the list of couplings
        # if self.model.runSettings['CppCouplingsList'] is True:
        #     fileName = os.path.join(path, 'PythonOutput', 'couplings')
        #     try:
        #         self.file = open(fileName, 'w')
        #         self.file.write(self.couplingListString())
        #     except:
        #         loggingCritical('ERROR while creating the list of couplings. Skipping.')
        #         return
        #     self.file.close()



    def buildCommands(self, commands):
        path = os.path.join(self.storePath, 'PythonOutput')
        commands += ['cd ' + path, 'make']


    def makeFileString(self):
        includeDir = os.path.join(self.model.runSettings['pyrateDir'], 'src', 'IO', 'include')

        soName = 'lib' + self._Name
        self.soName = soName

        s = soName + """\
.so: """ + soName + """.o
	g++ -shared -o $@ $+ -larmadillo -llapack -lblas -std=c++11 -I """ + includeDir + '\n' + soName + """\
.o: """ + self._Name + """.cpp
	g++ -fPIC -c -o $@ $< -std=c++11 -I """ + includeDir + '\n' + soName + """\
clean :
	rm *.o
	rm *.so
"""
        return s

    # def couplingListString(self):
    #     totString = '{'

    #     for i, c in enumerate(self.arrayToCouplings.values()):
    #         cString = c[0]
    #         if len(c) == 3:
    #             cString += str(c[2][0]+1) + 'x' + str(c[2][1]+1)
    #         if c[0] in self.complexCouplings:
    #             if c[1] == 0:
    #                 cString = 'Re[' + cString + ']'
    #             elif c[1] == 1:
    #                 cString = 'Im[' + cString + ']'
    #         if i < len(self.arrayToCouplings) - 1:
    #             cString += ','
    #             if (i+1)%5 != 0:
    #                 cString += ' '
    #             else:
    #                 cString += '\n'
    #         totString += cString

    #     totString += '}\n'

    #     return totString


    def generateCppFile(self, model):
        self.couplingsDefinition(model)
        self.betaFunction(model)
        self.cppSolver()
        self.preamble()

        self.allStr = self.initString + self.betaString + self.solverString

        if self.lightSolver:
            self.lallStr = self.linitString + self.betaString + self.lsolverString

    def preamble(self):
        mats = (self.couplingStructure != {})
        self.initString = '#include <iostream>\n#include <math.h>\n'

        if mats:
            # self.initString += '#define ARMA_NO_DEBUG\n#include <armadillo>\n'
            self.initString += '#include <armadillo>\n'

        self.initString += """\
#include "ascent/Ascent.h"

using namespace asc;
"""

        if mats:
            self.initString += 'using namespace arma;\n'

        self.linitString = self.initString + '\n'

        self.initString += """
extern "C"{
    extern """ + self.solverPrototype + """;
    extern """ + self.psolverPrototype + """;
}

"""





    def couplingsDefinition(self, model):
        def rSymb(s):
            return Symbol(s, real=True)

        self.couplingsToArray = {}
        self.arrayToCouplings = {}
        substitutedCouplings = [str(k) for subDic in model.substitutions.values() for k in subDic]

        count = 0
        for cType in model.toCalculate:
            if 'Anomalous' in cType:
                continue
            if cType not in model.loopDic or model.loopDic[cType] == 0:
                continue

            if cType == 'Vevs' and model.gaugeFixing is None:
                self.xi = 0
                self.gaugeFixing = True

            for k,v in model.allCouplings.items():
                if cType not in ('Yukawas', 'FermionMasses') or k not in model.couplingStructure:
                    if v[0] == cType and k not in substitutedCouplings:
                        if not k[-2:] == '^*' and not k[-4:] == '^{*}' and not k[-4:] == 'star':
                            self.couplingsToArray[k] = (f'x[{count}]',)
                            self.arrayToCouplings[f'x[{count}]'] = (k, 0)

                            findConjugate = [k+el in model.allCouplings for el in ('^*', '^{*}', 'star')]

                            cplx = any(findConjugate) or (cType in ('Yukawas', 'FermionMasses') and v[1].is_real is not True)
                            if cplx:
                                count += 1
                                self.couplingsToArray[k] += (f'x[{count}]',)
                                self.arrayToCouplings[f'x[{count}]'] = (k, 1)
                                self.complexCouplings.add(k)

                            count += 1
                        # else:
                        #     candidates = [el for el in model.allCouplings if el in k and el != k]
                        #     if len(candidates) == 1:
                        #         self.conjugatedCouplings[k] = model.allCouplings[candidates[0]][1]
                        #     else:
                        #         lengths = [len(el) for el in candidates]
                        #         i, maxLen = lengths.index(max(lengths)), max(lengths)
                        #         lengths.remove(maxLen)

                        #         if maxLen not in lengths:
                        #             self.conjugatedCouplings[k] = model.allCouplings[candidates[i]][1]
                        #         else:
                        #             loggingCritical(f"Warning in Cpp export: could not determine the conjugate quantity of {k}.")
                        #             exit()

                else:
                    if v[0] == cType and k not in substitutedCouplings:
                        struc = model.couplingStructure[k]
                        if not v[1].is_realMatrix:
                            self.complexCouplings.add(k)
                            if struc == (1,1):
                                self.couplingsToArray[k] = (f'x[{count}]', f'x[{count+1}]')
                                self.arrayToCouplings[f'x[{count}]'] = (k, 0)
                                self.arrayToCouplings[f'x[{count+1}]'] = (k, 1)
                                count += 2
                            else:
                                self.couplingsToArray[k] = {}
                                for i in range(struc[0]):
                                    for j in range(struc[1]):
                                        self.couplingsToArray[k][(i,j)] = (f'x[{count}]', f'x[{count+1}]')
                                        self.arrayToCouplings[f'x[{count}]'] = (k, 0, (i,j))
                                        self.arrayToCouplings[f'x[{count+1}]'] = (k, 1, (i,j))
                                        count += 2
                        else:
                            if struc == (1,1):
                                self.couplingsToArray[k] = (f'x[{count}]', )
                                self.arrayToCouplings[f'x[{count}]'] = (k, 0)
                                count += 1
                            else:
                                self.couplingsToArray[k] = {}
                                for i in range(struc[0]):
                                    for j in range(struc[1]):
                                        self.couplingsToArray[k][(i,j)] = (f'x[{count}]', )
                                        self.arrayToCouplings[f'x[{count}]'] = (k, 0, (i,j))
                                        count += 1#define ARMA_NO_DEBUG



    def betaFunction(self, model):
        self.types = {}
        self.constQuantities = {}
        self.nPar = len(self.arrayToCouplings)

        self.cTranslations = {}

        funcDefinition = '\tvoid operator()(const state_t& x, state_t& dx, const double){\n'

        # betaInitString = f'betaFunction(const double (&x)[{self.nPar}] , double (&x)[{self.nPar}], double t_) {{\n'
        betaInitString = '\n'
        for c, v in self.couplingsToArray.items():
            if isinstance(v, tuple):
                if len(v) == 1:
                    self.types[c] = 'double'
                    # betaInitString += f"\tdouble {c} = {v[0]};\n"
                    self.cTranslations[c] = v[0]
                elif len(v) == 2:
                    self.types[c] = 'std::complex<double>'
                    betaInitString += f"\t\tstd::complex<double> {c}({v[0]}, {v[1]});\n"
            else:
                if c in self.complexCouplings:
                    self.types[c] = 'Mat<std::complex<double>>'
                    betaInitString += f"\t\tMat<std::complex<double>> {c} = {{\n"
                    for (i,j), t in v.items():
                        if j == 0:
                            if i != 0:
                                betaInitString += '},\n'
                            betaInitString += '\t\t\t{'
                        else:
                            betaInitString += ', '
                        betaInitString += f"std::complex<double>({t[0]}, {t[1]})"
                    betaInitString += '}};\n'
                else:
                    # self.types[c] = 'Mat<double>'
                    # betaInitString += f"\t\tMat<double> {c} = {{\n"
                    # for (i,j), t in v.items():
                    #     if j == 0:
                    #         if i != 0:
                    #             betaInitString += '},\n'
                    #         betaInitString += '\t\t\t{'
                    #     else:
                    #         betaInitString += ', '
                    #     betaInitString += t[0]
                    # betaInitString += '}};\n'
                    self.types[c] = 'Mat<std::complex<double>>'
                    betaInitString += f"\t\tMat<std::complex<double>> {c} = {{\n"
                    for (i,j), t in v.items():
                        if j == 0:
                            if i != 0:
                                betaInitString += '},\n'
                            betaInitString += '\t\t\t{'
                        else:
                            betaInitString += ', '
                        betaInitString += f"std::complex<double>({t[0]}, 0)"
                    betaInitString += '}};\n'

        # The following is a fix, to avoid replacements of symbs whose name
        # contains the name of another coupling
        self.auxTranslations = {}
        otherNames = [el for el in self.couplingsToArray if el not in self.cTranslations]
        for k in list(self.cTranslations):
            for n in otherNames:
                if k in n and n not in self.auxTranslations:
                    self.auxTranslations[n] = f'#{len(self.auxTranslations)}#'


        betaExpressions = ''
        resultInitString = '\n'

        maxLoops = 0
        for cType, loopDic in model.couplingRGEs.items():
            if 'Anomalous' in cType:
                continue

            betaExpressions += '\n'
            for nLoop, RGEdic in loopDic.items():
                if nLoop + 1 > maxLoops:
                    maxLoops = nLoop + 1

                betaExpressions += '\t\tif(' + self.nLoopDic[cType] + f' >= {nLoop+1}){{\n'

                for c, RGE in RGEdic.items():
                    if c not in self.couplingsToArray:
                        continue

                    if RGE.find(Symbol('_xiGauge', real=True)) != set():
                        if 'xiGauge' not in self.constQuantities:
                            self.constQuantities['xiGauge'] = ('double', '0')

                    betaString = ccode(RGE/self.betaFactor, processContent=True, dic=self.cTranslations, auxDic=self.auxTranslations)

                    if c in self.cTranslations:
                        lhs = 'd' + self.cTranslations[c]
                        cTypeString = ''
                    else:
                        lhs = f'b{ccode(Symbol(c))}'
                        cTypeString = self.types[c] + ' '

                    if nLoop == 0:
                        if 'Mat' not in self.types[c] :
                            resultInitString += f'\t\t{cTypeString}{lhs} = 0;\n'
                        else:
                            resultInitString += f'\t\t{cTypeString}{lhs}' + str(self.couplingStructure[c])[:-1] + ', fill::zeros);\n'

                    takeRe = False
                    if c not in self.complexCouplings and 'Mat' not in self.types[c] and self.detectComplexSubterms(RGE):
                        takeRe = True

                    if not takeRe:
                        betaExpressions += '\t\t\t' + lhs + ' += (' + betaString + ')*kappaFactor' + str(nLoop+1) + '*logFactor;\n'
                    else:
                        betaExpressions += '\t\t\t' + lhs + ' += std::real(' + betaString + ')*kappaFactor' + str(nLoop+1) + '*logFactor;\n'

                betaExpressions += '\t\t}\n'


        xConversion = '\n'

        for x, tup in self.arrayToCouplings.items():
            c = tup[0]

            if c in self.cTranslations:
                continue

            if tup[1] == 0:
                cplxToRe = 'real'
            elif tup[1] == 1:
                cplxToRe = 'imag'

            if len(tup) == 2:
                if c not in self.complexCouplings:
                    xConversion += f'\t\td{x} = b{ccode(Symbol(c))};\n'
                else:
                    xConversion += f'\t\td{x} = b{ccode(Symbol(c))}.{cplxToRe}();\n'
            elif len(tup) == 3:
                if True:#c in self.complexCouplings:
                    xConversion += f'\t\td{x} = b{ccode(Symbol(c))}{tup[2]}.{cplxToRe}();\n'
                else:
                    xConversion += f'\t\td{x} = b{ccode(Symbol(c))}{tup[2]};\n'

        # self.betaInitString, self.resultInitString, self.betaExpressions, self.xConversion = betaInitString, resultInitString, betaExpressions, xConversion

        self.constQuantities['logFactor'] = ('double', 'log(10)')
        for i in range(maxLoops):
            self.constQuantities[f'kappaFactor{i+1}'] = ('double', '1.0/pow((4*M_PI), ' + str(eval(self.betaExponent.replace('n', str(i+1)))) + ')')

        for k, v in self.constQuantities.items():
            funcDefinition += f'\t\tstatic constexpr {v[0]} {k} = {v[1]};\n'

        if Printer.defineI:
            funcDefinition += '\t\tstatic constexpr std::complex<double> I(0,1);\n'


        initMatrices = self.initMatrices()

        betaStr = (funcDefinition + betaInitString + initMatrices + resultInitString
                  + betaExpressions + xConversion + '\t}')

        betaStr = """\
struct BetaFunction
{
\tBetaFunction(int nG, int nY, int nQ){
\t\tnGauge = nG;
\t\tnYuk = nY;
\t\tnQuartic = nQ;
\t}

""" + betaStr + """\n
\tprivate:
\t\tint nGauge;
\t\tint nYuk;
\t\tint nQuartic;
};

"""
        self.betaString = betaStr

    def initMatrices(self):
        s = '\n'
        for mat in Printer.preDefined['a']:
            c = ccode(mat.args[0])
            self.types['{c}_adj'] = self.types[c]
            s += f'\t\t{self.types[c]} {c}_adj = trans({c});\n'
        for mat in Printer.preDefined['t']:
            c = ccode(mat.args[0])
            self.types['{c}_trans'] = self.types[c]
            if c in self.complexCouplings:
                s += f'\t\t{self.types[c]} {c}_trans = strans({c});\n'
            else:
                s += f'\t\t{self.types[c]} {c}_trans = trans({c});\n'
        for mat in Printer.preDefined['c']:
            self.types['{c}_conj'] = self.types[c]
            c = ccode(mat.args[0])
            s += f'\t\t{self.types[c]} {c}_conj= conj({c});\n'


        for tr, (pos, count) in Printer.preDefined['tr'].items():
            conj = sortYukTrace(trace(conjugate(tr.args[0])), self.model.YukPos)
            realTrace = (conj == tr)

            if realTrace:
                # allReal = [ccode(el) not in self.complexCouplings for el in tr.args[0].atoms()]
                if False:#all(allReal):
                    s += f'\t\tdouble trace_{pos} = ' + ccode(tr, processContent=False) + ';\n'
                else:
                    s += f'\t\tdouble trace_{pos} = ' + ccode(tr, processContent=False) + '.real();\n'
            else:
                s += f'\t\tstd::complex<double> trace_{pos} = ' + ccode(tr, processContent=False) + ';\n'

        return s + '\n'

    def detectComplexSubterms(self, expr):
        # expr = expr.replace(Trace, lambda x: 1)
        atoms = expr.atoms()
        involvedCouplings = [el for el in atoms if not el.is_number]

        for c in involvedCouplings:
            cName = ccode(c)
            if cName in self.complexCouplings:
                return True

        if I in atoms:
            Printer.defineI = True
            return True

        return False

    def cppSolver(self):
        self.solverPrototype = 'int solver(double t0, double tmin, double tmax, double step, double* initialCouplings, double* tArray, double* resArray, int nG, int nY, int nQ)'
        self.psolverPrototype = 'int psolver(double t0, double t1, double step, double* initialCouplings, double* resArray, int nG, int nY, int nQ)'
        self.ppsolverPrototype = 'int ppsolver(double t0, double t1, double step, double* initialCouplings, double** resArray, int nG, int nY, int nQ)'

        solverStr = """{
    state_t x;

    double t = t0;
    int n = 0;

    bool landauDown = false;
    bool landauUp = false;
    double landauThreshold = 100;

    int nCouplings = """ + str(self.nPar) + """;

    RK4 integrator;
    BetaFunction betaFunction(nG,nY,nQ);


    for(int i=0; i<nCouplings; i++){
        x.push_back(initialCouplings[i]);
        resArray[i] = initialCouplings[i];
    }

    tArray[0] = t0;

    if(t0 > tmin){
        while(!landauDown && t > tmin + step){
            integrator(betaFunction, x, t, -1*step);
            n++;

            for(int i=0; i<nCouplings; i++){
                resArray[nCouplings*n+i] = x.at(i);
                if (x.at(i) > landauThreshold || x.at(i) < -1*landauThreshold)
                    landauDown = true;
            }
            tArray[n] = t;

        }
        if(!landauDown && t > tmin){
            integrator(betaFunction, x, t, tmin-t);
            n++;

            for(int i=0; i<nCouplings; i++){
                resArray[nCouplings*n+i] = x.at(i);
            }
            tArray[n] = t;
        }
    }

    t = t0;
    while(!landauUp && t < tmax - step){
        integrator(betaFunction, x, t, step);
        n++;

        for(int i=0; i<nCouplings; i++){
            resArray[nCouplings*n+i] = x.at(i);
            if (x.at(i) > landauThreshold || x.at(i) < -1*landauThreshold)
                landauUp = true;
        }
        tArray[n] = t;

    }
    if(!landauUp && t < tmax){
        integrator(betaFunction, x, t, tmax-t);
        n++;

        for(int i=0; i<nCouplings; i++){
            resArray[nCouplings*n+i] = x.at(i);
        }
        tArray[n] = t;
    }

    if(!landauDown && !landauUp)
        return n+1;
    return -1*(n+1);
}

"""

        psolverStr = """{
    state_t x;

    double t = t0;

    bool landauDown = false;
    bool landauUp = false;
    double landauThreshold = 100;

    int nCouplings = """ + str(self.nPar) + """;

    RK4 integrator;
    BetaFunction betaFunction(nG,nY,nQ);


    for(int i=0; i<nCouplings; i++){
        x.push_back(initialCouplings[i]);
    }

    if(t0 > t1){
        while(!landauDown && t > t1 + step){
            integrator(betaFunction, x, t, -1*step);

            for(int i=0; i<nCouplings; i++){
                if (x.at(i) > landauThreshold || x.at(i) < -1*landauThreshold)
                    landauDown = true;
            }
        }
        if(!landauDown && t > t1)
            integrator(betaFunction, x, t, t1-t);
    }
    else{
        while(!landauUp && t < t1 - step){
            integrator(betaFunction, x, t, step);

            for(int i=0; i<nCouplings; i++){
                if (x.at(i) > landauThreshold || x.at(i) < -1*landauThreshold)
                    landauUp = true;
            }
        }
        if(!landauUp && t < t1)
            integrator(betaFunction, x, t, t1-t);
    }

    for(int i=0; i<nCouplings; i++){
        resArray[i] = x.at(i);
    }

    if(!landauDown && !landauUp)
        return 1;
    return -1;
}

"""
        ppsolverStr = psolverStr.replace('resArray[i] = x.at(i);', '(*resArray[i]) = x.at(i);')

        self.solverString = self.solverPrototype + solverStr + self.psolverPrototype + psolverStr

        if self.lightSolver:
            self.lsolverString = self.psolverPrototype + psolverStr + self.ppsolverPrototype + ppsolverStr



class Printer(CXX11CodePrinter):
    preDefined = {'a': set(), 'c': set(), 't': set(), 'm': {}, 'tr': {}}
    defineI = False

    def __init__(self, end='', processContent=False, dic=None, auxDic=None):
        CXX11CodePrinter.__init__(self)
        self.end = end
        self.processContent = processContent

        self.dic = dic or {}
        self.auxDic = auxDic or {}

    def _print_Integer(self, expr):
        return super(CXX11CodePrinter, self)._print_Integer(expr) + '.'

    def _print_Symbol(self, expr):
        if expr == Symbol('_xiGauge', real=True):
            # return 'self.xi'
            return 'xiGauge'

        ret = super(CXX11CodePrinter, self)._print_Symbol(expr)
        ret = ret.replace('\\', '')

        return ret

    def _print_Pi(self, expr):
        return 'mathpi'

    def _print_I(self, expr):
        exit()

    def _print_adjoint(self, expr):
        if expr not in Printer.preDefined['a']:
            Printer.preDefined['a'].add(expr)

        return ccode(expr.args[0]) + '_adj'

    def _print_conjugate(self, expr):
        if isinstance(expr.args[0], mSymbol):
            if expr not in Printer.preDefined['c']:
                Printer.preDefined['c'].add(expr)

            return ccode(expr.args[0]) + '_conj'
        else:
            return 'std::conj(' + ccode(expr.args[0]) + ')'

    def _print_transpose(self, expr):
        if expr not in Printer.preDefined['t']:
            Printer.preDefined['t'].add(expr)

        return ccode(expr.args[0]) + '_trans'

    def _print_Trace(self, expr):
        if self.processContent:
            if expr not in Printer.preDefined['tr']:
                Printer.preDefined['tr'][expr] = [len(Printer.preDefined['tr']), 0]
            Printer.preDefined['tr'][expr][1] += 1

            return f"trace_{Printer.preDefined['tr'][expr][0]}"

        return 'trace(' + ccode(expr.args[0], processContent=self.processContent) + ')'


    def _print_Mul(self, expr):
        if expr.find(conjugate) != set():
        # Substitution x * conjugate(x) -> abs(x)^2
            conjArgs = {}
            args = splitPow(expr)
            for el in args:
                if isinstance(el, conjugate) or el.is_commutative == False or el.is_real:
                    continue
                else:
                    count = min(args.count(el), args.count(conjugate(el)))
                    if count != 0:
                        conjArgs[el] = count
            if conjArgs != {}:
                for k,v in conjArgs.items():
                    for _ in range(v):
                        args.remove(k)
                        args.remove(conjugate(k))
                        args.append(Abs(k)**2)
                expr = Mul(*args)


        if self.processContent:
            args = splitPow(expr)
            matrixArgs = []
            for el in args:
                if el.is_commutative is False:
                    matrixArgs.append(el)
            matrixArgs = tuple(matrixArgs)

            if len(matrixArgs) >= 2:
                if matrixArgs not in Printer.preDefined['m']:
                    Printer.preDefined['m'][matrixArgs] = 0
                Printer.preDefined['m'][matrixArgs] += 1

        return super()._print_Mul(expr)


    def subs(self, s):
        if self.dic == {}:
            return s
        for k,v in sorted(self.auxDic.items(), key=lambda x:-len(x[0])):
            s = s.replace(k,v)
        for k,v in sorted(self.dic.items(), key=lambda x:-len(x[0])):
            s = s.replace(k, v)
        for k,v in sorted(self.auxDic.items(), key=lambda x:-len(x[0])):
            s = s.replace(v,k)
        return s


def ccode(expr, **settings):
    p = Printer(**settings)
    if not Printer.defineI and expr.find(I):
        Printer.defineI = True
    return p.subs(p.doprint(expr))

