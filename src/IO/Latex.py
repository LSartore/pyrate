try:
    from sys import exit
    from sympy import (latex, Symbol, Add, adjoint, Mul, Abs, LeviCivita,
                       Indexed, IndexedBase, Rational, sympify,
                       Pow, conjugate, expand, flatten)

    from sympy.core.numbers import NegativeOne

    from sympy.printing.latex import LatexPrinter, modifier_dict, tex_greek_dictionary

    import datetime

    from Logging import loggingCritical
    from Definitions import mSymbol, splitPow, Trace

except ImportError:
    exit('Error while loading modules in Latex.py')


class LatexExport():
    def __init__(self, model, getLatexSubs=False):
        self._Name = model._Name
        self.string = ""

        self.latex = {}
        self.getLatex(model)

        self.latexReverse = {v:k for k,v in self.latex.items()}
        self.greekReverse = {v:k for k,v in tex_greek_dictionary.items()}

        self.cgcs = {}
        self.groupTheoryInfo = ('GroupTheoryInfo' in model.runSettings and model.runSettings['GroupTheoryInfo'] is True)

        # Update sympy's modifier for 'Xbar' -> overline for fermions, dagger for scalars
        if 'bar' in modifier_dict:
            del modifier_dict['bar']

        # Remove sympy's adjoint latex printing function
        if hasattr(adjoint, '_latex'):
            delattr(adjoint, '_latex')

        Printer.model = model
        Printer.absReplacements = model.runSettings['AbsReplacements']

        # BetaFunc definition
        self.betaFactor = model.betaFactor

        # Particles
        self.particleDic = model.Particles

        self.handleExplicit(model)

        self.dynkin = {}

        if getLatexSubs:
            return

        self.preamble(model)
        self.gaugeGroups(model)
        self.particles(model)
        self.lagrangian(model)
        self.RGEs(model)
        self.anomalous(model)

        if self.groupTheoryInfo:
            self.groupTheory(model)

        self.replacements(model)

        self.string += "\n\end{document}"

        model.latex = self.latex

    def write(self, fileName):
        try:
            self.file = open(fileName, 'w')
        except:
            exit('ERROR while opening the latex output file')

        self.file.write(self.string)
        self.file.close()

    def getLatex(self, model):
        if 'Latex' in model.saveSettings and model.saveSettings['Latex'] != {}:
            for k,v in model.saveSettings['Latex'].items():
                if "'" in v:
                    v = '{' + v + '}'
                if k in model.allCouplings:
                    self.latex[model.allCouplings[k][1]] = v
                elif k in model.Particles or k in model.Scalars:
                    self.latex[Symbol(k)] = v
                elif k in model.lagrangian.definitions:
                    self.latex[Symbol(k)] = v
                else:
                    loggingCritical(f"Warning in 'Latex' : object '{k}' is not defined. Skipping.")

        self.latex[Symbol('_xiGauge', real=True)] = '\\xi'

        # Conjugated particles
        for k,v in model.Fermions.items():
            if v.conj:
                continue

            if Symbol(k) not in self.latex:
                self.latex[Symbol(k + 'bar')] = '\\overline{'+k+'}'
            else:
                self.latex[Symbol(k + 'bar')] = '\\overline{'+self.latex[Symbol(k)]+'}'


        for k,v in model.ComplexScalars.items():
            if v.conj:
                continue

            if Symbol(k) not in self.latex:
                self.latex[Symbol(k + 'bar')] = k + '^{\\dagger}'
            else:
                self.latex[Symbol(k + 'bar')] = self.latex[Symbol(k)]+'^{\\dagger}'

        self.latex[Symbol('Eps')] = '\\epsilon'

        # Replacement Xstar -> X^*
        for k,v in model.allCouplings.items():
            if k[-4:] == 'star' and k not in self.latex:
                noStar = k[:-4]
                if noStar in model.allCouplings:
                    noStarSymb = model.allCouplings[noStar][1]
                    self.latex[v[1]] = self.totex(conjugate(noStarSymb))

            if v[0] == 'ScalarMasses' and k in model.assumptions and model.assumptions[k]['squared'] is True:
                if v[1] in self.latex:
                    sm = self.latex[v[1]]
                else:
                    sm = self.totex[v[1]]

                self.latex[Symbol(str(v[1]), commutative=False)] = '{'+sm+'}^2'

    def preamble(self, model):
        date = datetime.datetime.now()
        month = {1: 'January', 2: 'Febuary', 3: 'March', 4: 'April',
                 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        self.string += r"""\documentclass[12pt]{article}
\pagestyle{empty}

\usepackage[top=2cm,bottom=2cm,left=3cm,right=2cm,nohead,nofoot]{geometry}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{bbm}
\usepackage{booktabs}
\usepackage[]{hyperref}
\usepackage{autobreak}"""

        if self.groupTheoryInfo:
            self.string += r"""
\usepackage{booktabs}
\usepackage{multirow}"""

        self.string += (r"""
\setlength{\parindent}{0pt}

\newcommand{\tr}{\mathrm{Tr}}
\newcommand{\trans}{\mathrm{T}}

\title{""" + model._Name.replace('_', '\\_') + '}\n' +
r"\author{PyR@TE 3.0}" + '\n'
r"\date{" + str(date.day) + " " + month[date.month] + " " + date.strftime("%Y, %H:%M") +r"""}

\begin{document}
    \maketitle
    \tableofcontents
    \clearpage""")

    def gaugeGroups(self, model):
        def printCoupling(coupling):
            gutNorm = {str(k):v for k,v in model.gutNorm.items()}
            if str(coupling) not in gutNorm:
                return self.totex(coupling)
            else:
                return self.totex(coupling) + r'\rightarrow' + latex(gutNorm[str(coupling)][1], fold_short_frac=None).replace(self.totex(coupling), r'\,' + self.totex(coupling))
        # Gauge groups
        gaugeGroups = []

        for gName, g in model.gaugeGroups.items():
            coupling = str(g.g)
            if 'rename' in model.substitutions and coupling in model.substitutions['rename']:
                coupling = model.substitutions['rename'][coupling][1]
            coupling = model.allCouplings[coupling][1]
            gaugeGroups.append( (gName, '$'+g.latex+'$', str(g.abelian),  '$'+printCoupling(coupling)+'$') )

        self.string += (r"""

\section{Model}{

\subsection{Gauge groups}

\begin{table}[h]
\renewcommand{\arraystretch}{1.3}
\centering
\begin{tabular}{c@{\hskip .66cm}c@{\hskip .66cm}c@{\hskip .5cm}c}
\hline
Name & Type & Abelian & Coupling constant \\ \hline
""" + " \\\\\n".join([r' & '.join(g) for g in gaugeGroups]) +
r""" \\ \hline
\end{tabular}
\end{table}""")

        if model.kinMix:
            self.string += '\n\n' + r'\textbf{Kinetic mixing in the abelian sector :}\\' + '\n'
            self.string += '\\begin{equation*}\n'
            self.string += r'G = ' + self.totex(model.kinMat) + '\n'
            self.string += '\\end{equation*}\n'
        elif model.nU > 1:
            self.string += '\n\n' + r'\textbf{Note : kinetic mixing in the abelian sector is neglected}\\'


    def particles(self, model):
        totalGroup = list(model.gaugeGroups)
        totalGroup = r' $\times$ '.join(totalGroup)
        # Fermions
        fermions = []

        for name, f in model.Fermions.items():
            if f.conj:
                continue
            gen = str(f.gen) if type(f.gen) == int else '$'+self.totex(f.gen)+'$'

            rep = [0]*len(model.gaugeGroups)
            for g, qnb in f.Qnb.items():
                gPos = list(model.gaugeGroups).index(g)

                if model.gaugeGroups[g].abelian:
                    repName = ('+' if qnb > 0 else '') + self.totex(Rational(qnb))
                else:
                    repName = model.gaugeGroups[g].repName(qnb)

                    # Fill Dynkin dic
                    for r,v in model.gaugeGroups[g].repDic.items():
                        self.dynkin[str(list(r))] = v[4]
                        self.dynkin[str(tuple(r))] = v[4]

                rep[gPos] = '$' + repName + '$'


            repStr = '('+', '.join(rep)+')' if len(rep) > 1 else rep[0]
            fermions.append(('$'+self.totex(Symbol(name))+'$', gen, repStr))

        self.string += ('\n' + r"""
\subsection{Fermions}

\begin{table}[h]
\renewcommand{\arraystretch}{1.15}
\centering
\begin{tabular}{c@{\hskip .66cm}c@{\hskip .66cm}c}
\hline
Name & Generations & """ + totalGroup + r"""\\ \hline \\ [-2ex]
""" +
" \\\\[.2cm]\n".join([r' & '.join(f) for f in fermions]) +
r""" \\[.1cm] \hline
\end{tabular}
\end{table}""")

        # Scalars
        scalars = []

        for name, s in model.Particles.items():
            if name in model.Fermions or s.conj:
                continue
            gen = str(s.gen) if type(s.gen) == int else '$'+self.totex(s.gen)+'$'

            rep = [0]*len(model.gaugeGroups)
            for g, qnb in s.Qnb.items():
                gPos = list(model.gaugeGroups).index(g)

                if model.gaugeGroups[g].abelian:
                    repName = ('+' if qnb > 0 else '') + self.totex(Rational(qnb))
                else:
                    repName = model.gaugeGroups[g].repName(qnb)
                rep[gPos] = '$' + repName + '$'

            # Norm*(re + im) formatting
            if s.cplx:
                if isinstance(s.norm, Mul) and len(s.norm.args) == 2 and s.norm.args[1]**(-2) == s.norm.args[0]:
                    expr = '\\frac{1}{'+self.totex(s.norm.args[1])+'}'
                elif s.norm != 1:
                    expr = self.totex(s.norm)
                else:
                    expr = ''

                real = ' + i\\, '.join([self.totex(r._name) for r in s.realFields])
                if s.norm != 1:
                    real = ' \\left(' + real + '\\right)'

                expr = '$' + expr + real + '$'
            else:
                expr = '/'
            repStr = '('+', '.join(rep)+')' if len(rep) > 1 else rep[0]
            scalars.append(('$'+self.totex(Symbol(name))+'$', str(s.cplx), expr, gen, repStr))

        self.string += ('\n' + r"""
\subsection{Scalars}

\begin{table}[h]
\renewcommand{\arraystretch}{1.15}
\centering
\begin{tabular}{c@{\hskip .66cm}c@{\hskip .66cm}ccc}
\hline
Name & Complex & Expression & Generations & """ + totalGroup + r"""\\ \hline \\ [-2ex]
""" +
" \\\\[.2cm]\n".join([r' & '.join(s) for s in scalars]) +
r""" \\[.1cm] \hline
\end{tabular}
\end{table}""")

    def handleExplicit(self, model):
        for coupling in model.ExplicitMatrices:
            strForm = False
            if type(coupling) == tuple:
                coupling = coupling[0]
                strForm = True
            mat = coupling.as_explicit()
            for i,c in enumerate(mat):
                if c not in model.allCouplings:
                    model.allCouplings[str(c)] = list(model.allCouplings[str(coupling)])
                    model.allCouplings[str(c)][1] = c
                    model.allCouplings[str(c)] = tuple(model.allCouplings[str(c)])

                    cType = model.allCouplings[str(c)][0]
                    model.couplingsPos[cType][str(c)] = model.couplingsPos[cType][str(coupling)] + (i+1)/1000

                    if coupling in self.latex:
                        if strForm:
                            c = str(c)
                        self.latex[c] = str(c).replace(str(coupling), r'\left(' + self.latex[coupling] + r'\right)')


    def lagrangian(self, model):
        self.string += "\n\n\\section{Lagrangian}\n"

        translation = {'Yukawas': 'Yukawa couplings',
                       'QuarticTerms': 'Quartic couplings',
                       'TrilinearTerms' : 'Trilinear couplings',
                       'ScalarMasses': 'Scalar mass couplings',
                       'FermionMasses': 'Fermion mass couplings'}

        lagIndex = {'Yukawas': 'Y',
                    'QuarticTerms': 'Q',
                    'TrilinearTerms' : 'T',
                    'ScalarMasses': '{sm}',
                    'FermionMasses': '{fm}'}


        def parseTerm(cType, cSymb, expr):
            if isinstance(expr, Add):
                return Mul(cSymb, Add(*[parseTerm(cType, 1, el) for el in expr.args], evaluate=False), evaluate=False)

            def split(term):
                noInds = str(term)[:str(term).find('[')] if str(term).find('[') != -1 else str(term)
                inds = str(term).replace(noInds, '')

                if inds == '':
                    inds = []
                else:
                    inds = eval(inds.replace('[', "['").replace(',', "','").replace(']',"']"))
                return (noInds, inds)

            def sortKey(x):
                sp = split(x)
                return (sp[0] not in model.lagrangian.cgcs, sp[1], (0 if 'bar' in str(x) else 1), sp[0])


            args = []
            levi = []
            coeff = 1

            for el in splitPow(expr, deep=True):
                if el.is_number:
                    coeff *= el
                elif isinstance(el, LeviCivita):
                    levi.append(el)
                else:
                    args.append(el)

            base1, base2 = None, None

            # Workaround for (x^\dagger)**2 x**2 -> (x^\dagger x)**2
            # + Workaround for  x^\dagger x^\dagger x x -> (x^\dagger x)**2
            if (len(args) == 2 and isinstance(args[0], Pow) and args[0].args[1] == 2
                               and isinstance(args[1], Pow) and args[1].args[1] == 2):
                base1, base2 = sorted([str(args[0].args[0]), str(args[1].args[0])])
            if (lambda a: len(a) == 4 and a[0] == a[1] and a[2] == a[3])(sorted(args, key=lambda x:str(x))):
                sArgs = sorted(args, key=lambda x: str(x))
                base1, base2 = str(sArgs[1]), str(sArgs[2])

            if base1 is not None and base2 is not None:
                if base2 == base1+'bar':
                    if cSymb != 1:
                        new = Symbol(str(cSymb), commutative=False)
                    if cSymb in self.latex and new not in self.latex:
                        self.latex[new] = self.latex[cSymb]
                        cSymb = new
                    else:
                        cSymb = new

                    ret = (Symbol(base2, commutative=False)*Symbol(base1, commutative=False))**2

                    if Symbol(base1) in self.latex and Symbol(base1, commutative=False) not in self.latex:
                        self.latex[Symbol(base1, commutative=False)] = self.latex[Symbol(base1)]
                    if Symbol(base2) in self.latex and Symbol(base2, commutative=False) not in self.latex:
                        self.latex[Symbol(base2, commutative=False)] = self.latex[Symbol(base2)]

                    if coeff*cSymb != 1:
                        return coeff*cSymb*ret

                    return ret


            if isinstance(cSymb, mSymbol):
                newArgs = [0 for _ in range(len(args))]
                fermions = list(model.allCouplings[str(cSymb)][2])

                # Possibly replace f^2 by [f,f] in the list of args (e.g. majorana fermions)
                args = splitPow(args)
                newArgs = [0,0]
                scalar = 0

                remain = []
                # Add generation indices to fermions and sort as [fermion1, scalar, fermion2]
                for el in args:
                    splitEl = split(el)

                    if splitEl[0] in fermions:
                        fCount = fermions.index(splitEl[0]) + 1
                        pos = fermions.index(splitEl[0])

                        el = IndexedBase(splitEl[0]).__getitem__(('f'+str(fCount), *splitEl[1]))

                        newArgs[pos] = el
                        fermions[pos] = 0
                    else:
                        if isinstance(el, Indexed):
                            base = el.base
                        else:
                            base = el

                        if str(base) in model.Particles:
                            scalar = el
                        else:
                            remain.append(el)

                if scalar != 0:
                    newArgs.insert(1, scalar)

                newArgs = remain + newArgs

                if cSymb in self.latex:
                    cSymb = Symbol(str(self.latex[cSymb]) + '{}_{f_1,f_2}', commutative=False)
                else:
                    cSymb = Symbol(str(cSymb) + '{}_{f_1,f_2}', commutative=False)
            else:
                newArgs = sorted(args, key=sortKey)

                if cSymb != 1:
                    new = Symbol(str(cSymb), commutative=False)
                    if cSymb in self.latex and new not in self.latex:
                        self.latex[new] = self.latex[cSymb]
                        cSymb = new
                    else:
                        cSymb = new

            for el in levi:
                for p, obj in enumerate(newArgs):
                    if el.args[1] in obj.args:
                        pos = p
                        break
                newArgs.insert(pos, el)

            if coeff*cSymb != 1:
                return Mul(coeff*cSymb, *newArgs, evaluate=False)
            else:
                return Mul(*newArgs, evaluate=False)

        # Print definitions
        userDefinitions = {k:v for k,v in model.lagrangian.definitions.items() if k not in model.Particles and k not in ('kd', 'Eps')}
        if userDefinitions != {}:
            self.string += "\n\\subsection{Definitions}\n{\\allowdisplaybreaks\n\\begin{align*}\n"

            for dName, d in userDefinitions.items():
                sDef = d.fromDef.replace('[', '_{').replace(']', '}')
                for k,v in self.latex.items():
                    if str(k) in model.lagrangian.definitions:
                    # if str(k) == sDef:
                        sDef = sDef.replace(str(k), v)

                self.string += sDef + ' &= '

                if d.expr is not None:
                    expr = self.totex(d.expr)
                else:
                    obj = model.saveSettings['Potential']['Definitions'][d.fromDef]
                    expr = self.totex(obj)

                self.string += expr

                if dName in model.lagrangian.cgcs:
                    self.cgcs[dName] = (sDef, expr)

                self.string += r'\\' + '\n'
            self.string += '\\end{align*}'

        for cType, dic in model.expandedPotential.items():
            if cType == 'Definitions' or dic == {}:
                continue
            lag = []

            self.string += "\\subsection{" + translation[cType] + "}\n{\\allowdisplaybreaks\n\\begin{align*}\n"

            for c, expr in dic.items():
                cSymb = model.allCouplings[c][1]

                term = parseTerm(cType, cSymb, expr)
                if type(term) != list:
                    lag.append(term)
                else:
                    for el in term:
                        lag.append(el)

            lagExpr = self.totex(lag).replace(r'\left', r'\big').replace(r'\right', r'\big')
            self.string += r"\begin{autobreak}" + "\n"
            self.string += r"-\mathcal{L}_" + lagIndex[cType] + " = " + lagExpr

            if cType == 'Yukawas' or cType == 'FermionMasses':
                self.string += '\n' + r' + \text{h.c.}'
            self.string += '\n' + r"\end{autobreak}"
            self.string += "\n\\end{align*}\n}"


    def RGEs(self, model):
        Printer.RGE = True

        self.string += "\n\n\\section{Renormalization Group Equations}\n"

        translation = {'GaugeCouplings': 'Gauge couplings',
                       'Yukawas': 'Yukawa couplings',
                       'QuarticTerms': 'Quartic couplings',
                       'TrilinearTerms' : 'Trilinear couplings',
                       'ScalarMasses': 'Scalar mass couplings',
                       'FermionMasses': 'Fermion mass couplings',
                       'Vevs': 'Vacuum-expectation values'}

        self.string += '\\subsection{Convention}\n\\begin{equation*}\n'

        if self.betaFactor == 1:
            beta = r'\mu \frac{d X}{d \mu}'
        else:
            X = Symbol('X')
            if self.betaFactor == Rational(1,2):
                beta = r'\mu^2 \frac{d X}{d \mu^2}'
            elif isinstance(self.betaFactor, Mul) and Rational(1,2) in self.betaFactor.args:
                    self.betaFactor *= Rational(2)
                    beta = r'\mu^2 \frac{d}{d \mu^2}\left(' + latex(self.betaFactor*X) + r'\right)'
            else:
                beta = r'\mu \frac{d}{d \mu}\left(' + latex(self.betaFactor*X) + r'\right)'


        self.string += r'\beta\left(X\right) \equiv ' + beta
        self.string += r'\equiv' + '+'.join([latex(sympify(f'1/(4*pi)**({model.betaExponent(n)})', evaluate=False))+'\\beta^{('+str(n)+')}(X)' for n in range(1, 1+max(model.nLoops))])

        self.string += '\n\\end{equation*}\n'

        if ('sub' in model.substitutions and model.substitutions['sub'] != {}) or ('yukMat' in model.substitutions and model.substitutions['yukMat'] != {}):
            self.string += '\\subsection{Definitions and substitutions}\n'

            if 'zero' in model.substitutions and model.substitutions['zero'] != {}:
                nPerLine = 5
                i = 0
                n0 = len(model.substitutions['zero'])
                n = n0
                while n > 0:
                    if n > nPerLine:
                        group = list(model.substitutions['zero'].items())[i:i+nPerLine]
                        n -= nPerLine
                        i += nPerLine
                    else:
                        group = list(model.substitutions['zero'].items())[i:]
                        n = 0

                    self.string += '\\begin{equation*}\n'
                    self.string += ' \\quad,\\quad '.join([self.totex(v) + '=' + self.totex(0) for k,v in group]) + '\n'
                    self.string += '\\end{equation*}\n'


            if 'yukMat' in model.substitutions and model.substitutions['yukMat'] != {}:
                nPerLine = 3
                i = 0
                n0 = len(model.substitutions['yukMat'])
                n = n0
                while n > 0:
                    if n > nPerLine:
                        group = list(model.substitutions['yukMat'].items())[i:i+nPerLine]
                        n -= nPerLine
                        i += nPerLine
                    else:
                        group = list(model.substitutions['yukMat'].items())[i:]
                        n = 0

                    self.string += '\\begin{equation*}\n'
                    self.string += ' \\quad,\\quad '.join([self.totex(model.allCouplings[k][1]) + '=' + self.totex(v[1]) for k,v in group]) + '\n'
                    self.string += '\\end{equation*}\n'


            if 'sub' in model.substitutions and model.substitutions['sub'] != {}:
                nPerLine = 3
                i = 0
                n0 = len(model.substitutions['sub'])
                n = n0
                while n > 0:
                    if n > nPerLine:
                        group = list(model.substitutions['sub'].items())[i:i+nPerLine]
                        n -= nPerLine
                        i += nPerLine
                    else:
                        group = list(model.substitutions['sub'].items())[i:]
                        n = 0

                    self.string += '\\begin{equation*}\n'
                    self.string += ' \\quad,\\quad '.join([self.totex(v[0]) + '\equiv' + self.totex(v[1], baseMul=True) for k,v in group]) + '\n'
                    self.string += '\\end{equation*}\n'

        for cType, dic in model.couplingRGEs.items():
            if dic == {}:
                continue
            if 'Anomalous' in cType:
                continue

            self.string += "\n\n\\subsection{" + translation[cType] + "}\n{\\allowdisplaybreaks\n"

            if cType in model.NonZeroCouplingRGEs:
                self.string += r'\emph{\textbf{Warning:} The following couplings were set to 0 in the model file, but have a non-zero \mbox{$\beta$-function}.'
                self.string += '\n' + r'This may lead to an inconsistent RG flow, except if these couplings can be safely approximated to 0 at all considered energy scales : ' + '\n}\n'
                self.string += r'\begin{center}' + '\n'
                self.string += '$' + '$, $'.join([(str(c) if str(c) not in self.latex else self.latex[str(c)]) for c in model.NonZeroCouplingRGEs[cType][0]]) + '$ .\n'
                self.string += r'\end{center}' + '\n'

            if cType == 'Vevs':
                self.string += self.vevs(model)

            for c in dic[0]:
                for n in range(model.loopDic[cType]):
                    RGE = dic[n][c]
                    cSymb = model.allCouplings[c][1]

                    self.string += "\n\\begin{align*}\n\\begin{autobreak}\n"
                    if type(RGE) != list:
                        self.string += '\\beta^{('+str(n+1)+')}(' + self.totex(cSymb) + ') ='
                        self.string += self.totex(RGE, sort=True, cType=cType)
                    else:
                        self.string += '\\left.\\beta^{('+str(n+1)+')}(' + self.totex(cSymb) + ')\\right\\rvert_{i j} ='
                        totalRGE = RGE[0]
                        for k,v in RGE[1].items():
                            if not isMinus(v.args[0] if isinstance(v, Add) else v):
                                totalRGE += Delta(Symbol('i'), k[0], Symbol('j'), k[1])*v
                            else:
                                totalRGE -= Delta(Symbol('i'), k[0], Symbol('j'), k[1])*(-1*v)

                        self.string += self.totex(expand(totalRGE), sort=True, cType=cType)

                    self.string += "\n\\end{autobreak}\n\\end{align*}"

            if cType not in model.NonZeroCouplingRGEs:
                self.string += "\n}"
                continue

            dic = model.NonZeroCouplingRGEs[cType]

            for c in dic[0]:
                for n in range(model.loopDic[cType]):
                    RGE = dic[n][c]

                    self.string += "\n\\begin{align*}\n\\begin{autobreak}\n"
                    self.string += '\\beta^{('+str(n+1)+')}(' + self.latex[str(c)] + ') ='
                    self.string += self.totex(RGE, sort=True, cType=cType).replace('\\left(', '\\big(').replace('\\right)', '\\big)')

                    self.string += "\n\\end{autobreak}\n\\end{align*}"

            self.string += "\n}"

    def vevs(self, model):
        if model.vevs == {}:
            return ''

        s = "\n\\textbf{Definitions:}\n\\begin{align*}\n\t"
        allStrings = []
        for k,v in model.vevs.items():
            fieldComponent = list(model.allScalars.values())[v[0]]

            indices = [i+1 for i in fieldComponent[2] if i != -1]
            multiplet = ( indices != [] )
            cplx = ( len(v) > 3 )

            sField = self.totex(Symbol(str(fieldComponent[1])))

            if multiplet:
                sField = '{'+sField+'}_{'+','.join([str(i) for i in indices])+'}'

            norm = ''
            if cplx:
                if v[3].norm != 1:
                    if isinstance(v[3].norm, Mul) and len(v[3].norm.args) == 2 and v[3].norm.args[1]**(-2) == v[3].norm.args[0]:
                        norm = '\\frac{1}{'+self.totex(v[3].norm.args[1])+'}'
                    else:
                        norm = self.totex(v[3].norm)

            if norm != '':
                vevStr = (norm + sField + ' &\\rightarrow ' + norm+ '\\left(' +
                          sField + ' + ' + self.totex(v[1]) + '\\right)')
            else:
                vevStr = sField + ' &\\rightarrow ' + sField + ' + ' + self.totex(v[1])

            if cplx:
                vevStr = self.totex(Symbol(str(v[3]))) + ' \\; : \\; ' + vevStr
            else:
                vevStr = self.totex(Symbol(str(fieldComponent[1]))) + ' \\; : \\; ' + vevStr

            allStrings.append(vevStr)

        s += '\\\\\n\t'.join(allStrings)
        s += '\n\\end{align*}\n\n'

        # Display and apply the user-defined gauge fixing
        if model.gaugeFixing is not None:
            xi = Symbol('_xiGauge', real=True)
            s += r'\textbf{Gauge fixing:}\\' + '\n\\begin{equation*}\n'
            s += self.totex(xi) + ' \\rightarrow ' + self.totex(model.gaugeFixing)
            s += '\n\\end{equation*}\n'


            for nLoop, RGEdic in model.couplingRGEs['Vevs'].items():
                for vev, RGE in RGEdic.items():
                    model.couplingRGEs['Vevs'][nLoop][vev] = RGE.subs(xi, model.gaugeFixing)

        s += '\\textbf{RGEs:}\n'
        return s


    def anomalous(self, model):
        if model.fermionAnomalous == {} and model.scalarAnomalous == {}:
            return

        s = '\n\\section{Anomalous dimensions}\n\n'

        s += '\\subsection{Convention}\n\\begin{equation*}\n'

        maxLoops = max(model.loopDic['FermionAnomalous'], model.loopDic['ScalarAnomalous'])
        s += r'\gamma\left(X,\\ Y\right) '
        s += r'\equiv' + '+'.join([latex(sympify(f'1/(4*pi)**({2*n})', evaluate=False))+'\\gamma^{('+str(n)+')}\\left(X, Y\\right)' for n in range(1, 1+maxLoops)])

        s += '\n\\end{equation*}\n'

        if model.gaugeFixing is not None:
            xi = Symbol('_xiGauge', real=True)
            s += r'\textbf{Gauge fixing:}\\' + '\n\\begin{equation*}\n'
            s += self.totex(xi) + ' \\rightarrow ' + self.totex(model.gaugeFixing)
            s += '\n\\end{equation*}\n'


            for nLoop, RGEdic in model.couplingRGEs['Vevs'].items():
                for vev, RGE in RGEdic.items():
                    model.couplingRGEs['Vevs'][nLoop][vev] = RGE.subs(xi, model.gaugeFixing)

        if model.fermionAnomalous != {}:
            s += "\\subsection{Fermion anomalous dimensions}\n{\\allowdisplaybreaks\n"

            if model.saveSettings['FermionAnomalous'] == 'All':
                s += '\n\\emph{All fermion anomalous dimensions were computed. Those equal to 0 do not appear below.}\n\n'

            for k,v in model.fermionAnomalous.items():
                if model.saveSettings['FermionAnomalous'] == 'All':
                    if all([el[k] == 0 for el in model.couplingRGEs['FermionAnomalous'].values()]):
                        continue

                gamma = lambda n: '\\gamma_F^{('+str(n+1)+')}' + self.totex(k, incrementInds=True)

                for n, RGEdic in model.couplingRGEs['FermionAnomalous'].items():
                    RGE = RGEdic[k]

                    if model.gaugeFixing is not None:
                        RGE = RGE.subs(xi, model.gaugeFixing)

                    s += "\n\\begin{align*}\n\\begin{autobreak}\n"
                    s += gamma(n) + ' = '
                    s += self.totex(RGE, sort=True, cType='Vevs')
                    s += "\n\\end{autobreak}\n\\end{align*}"

            s += "\n}"

        if model.scalarAnomalous != {}:
            s += "\\subsection{Scalar anomalous dimensions}\n{\\allowdisplaybreaks\n"

            if model.saveSettings['ScalarAnomalous'] == 'All':
                s += '\n\\emph{All scalar anomalous dimensions were computed. Those equal to 0 do not appear below.}\n\n'

            for k,v in model.scalarAnomalous.items():
                if model.saveSettings['ScalarAnomalous'] == 'All':
                    if all([el[k] == 0 for el in model.couplingRGEs['ScalarAnomalous'].values()]):
                        continue

                gamma = lambda n: '\\gamma_S^{('+str(n+1)+')}' + self.totex(k, incrementInds=True)

                for n, RGEdic in model.couplingRGEs['ScalarAnomalous'].items():
                    RGE = RGEdic[k]

                    if model.gaugeFixing is not None:
                        RGE = RGE.subs(xi, model.gaugeFixing)

                    s += "\n\\begin{align*}\n\\begin{autobreak}\n"
                    s += gamma(n) + ' = '
                    s += self.totex(RGE, sort=True, cType='Vevs')
                    s += "\n\\end{autobreak}\n\\end{align*}"

            s += "\n}"

        self.string += s

    def groupTheory(self, model):
        # Only abelian groups : skip this part
        if all([g.abelian for g in model.gaugeGroups.values()]):
            return

        self.string += "\n\\clearpage\n\n\\appendix\n\\section{Group theoretical information}\n"

        # 1) Gauge groups table
        self.string += "\\subsection{Gauge groups}\n"
        self.string += r"""
\begin{table}[h]
\renewcommand{\arraystretch}{1.1}
\centering
\begin{tabular}{@{}cccccccc@{}}
\toprule
\multirow{2}{*}{Group} & \multirow{2}{*}{Lie algebra} & \multirow{2}{*}{Dim.} & \multirow{2}{*}{Rank} & \multicolumn{4}{c}{Representations}                 \\ \cmidrule(l){5-8}
                       &                              &                       &                       & Name / Dim. & Dynkin labels & Index & Reality   \\ \midrule
"""
        gDic = {}

        for grp in model.gaugeGroupsList:
            if grp.abelian:
                continue
            if grp.type not in gDic:
                gDic[grp.type] = grp
            else:
                # Merge rep dics
                for k,v in grp.repDic.items():
                    if k not in gDic[grp.type].repDic:
                        gDic[grp.type].repDic = v

        for gPos, grp in enumerate(gDic.values()):
            repList = sorted([v for v in grp.repDic.values() if v[0] > 1], key=lambda x: x[0])

            l = str(len(repList))
            for i, rep in enumerate(repList):
                name, labels, repType, index = rep[4], str(list(rep[1])), rep[2].capitalize(), self.totex(rep[5])
                if i == 0:
                    pre = '\\multirow{'+l+'}{*}'
                    self.string += pre + '{'+grp.type+'} & '
                    self.string += pre + '{'+grp.sName+'} & '
                    self.string += pre + '{'+str(grp.dim)+'} & '
                    self.string += pre + '{'+str(grp.rank)+'} & '
                else:
                    self.string += ' & '*4
                self.string += '$'+name+'$ & '
                self.string += labels + ' & '
                self.string += '$'+index+'$' + ' & '
                self.string += repType
                if rep[0] == grp.dim:
                    self.string += ' (adjoint)'
                self.string += r'\\'
                if i+1 < len(repList):
                    self.string += '\n'

            if i+1 < len(repList) and gPos+1 < len(gDic):
                self.string += '\\midrule\n'
            else:
                self.string += '\\bottomrule\n'

        self.string += "\\end{tabular}\n\\end{table}\n"


        if model.lagrangian.cgcs == {}:
            return

        # 2) CGCs expression
        self.string += "\\subsection{Clebsch-Gordan coefficients}\n\n"

        for k, v in model.lagrangian.cgcs.items():
            definition = self.cgcs[k]
            definition, expr = Printer.generateCGCexpr(definition[0], v)

            self.string += "\n\\begin{align*}\n\\begin{autobreak}\n"
            self.string += definition + ' = \n' + self.totex(expr).strip()
            self.string += "\n\\end{autobreak}\n\\end{align*}\n" #

        self.string += '\n\n'



    def totex(self, expr, sort=False, cType=None, baseAdd=False, baseMul=False, incrementInds=False):
        return Printer(self, self.latex, sort=sort, cType=cType,
                                         baseAdd=baseAdd, baseMul=baseMul,
                                         incrementInds=incrementInds).doprint(expr)

    def replacements(self, model):
        self.string = self.string.replace('\\\\\n&\\\\\n&', '\\\\\n&')

        #This is for explicit matrix elements
        for k,v in self.latex.items():
            if type(k) != str:
                continue

            self.string = self.string.replace(k, v)

class Printer(LatexPrinter):
    model = None
    RGE = False
    absReplacements = False

    def __init__(self, LatexExport, symbDic, sort=False, cType=None, baseAdd=False, baseMul=False, incrementInds=False):
        self.latex = LatexExport

        self.sort = sort
        self.cType = cType
        self.baseAdd = baseAdd
        self.baseMul = baseMul
        self.incrementInds = incrementInds

        self.dic = symbDic

        LatexPrinter.__init__(self, {'symbol_names': symbDic})

    def baseLatex(self, expr):
        return latex(expr, symbol_names=self.dic)

    def _print_Add(self, expr):
        args = expr.args

        if self.sort:
            args = self.sortTerms(args)

        if self.baseAdd:
            return super()._print_Add(expr)

        return self.splitEquation(args)

    def _print_Mul(self, expr):
        if expr.find(Indexed) != set() and expr.find(Pow) != set():
            print("\n\n indexed mul :", expr)

            if not all([int(el.exp) != el.exp for el in expr.find(Pow)]):
                return ' '.join([self._print(el) for el in splitPow(expr)])

        if isinstance(expr.args[0], Rational) and abs(expr.args[0]) != 1 and not self.baseMul:
            coeff = Mul(*[el for el in flatten(expr.as_coeff_mul()) if el.is_number])

            return self.latex.totex(coeff, baseMul=True) + ' ' + (self._print((expr/coeff).doit()) if expr != coeff else '')

        if self.absReplacements and expr.find(conjugate) != set():
        # Substitution x * conjugate(x) -> |x|^2
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

        return super()._print_Mul(expr)

    def _print_Indexed(self, expr):
        base, inds = expr.args[0], expr.args[1:]

        s = self._print(base)

        if self.incrementInds:
            inds = [el+1 for el in inds]

        if '_' in s:
            s = '{' + s + '}_{' + ','.join([self.latex.totex(el) for el in inds]) + '}'
        else:
            s += '_{' + ','.join([self.latex.totex(el) for el in inds]) + '}'

        return s

    def _print_list(self, expr):
        return self.splitEquation(expr)

    def _print_Trace(self, expr):
        mat = expr.arg
        return r"\tr\left(%s \right)" % self.latex.totex(mat)

    def _print_conjugate(self, expr, exp=None):
        s = self._print(expr.args[0])

        if s in self.latex.latex.values() and '^' in s:
            if exp is not None:
                return '{' + s + '}^{' + exp + '\\,*}'

            pos = s.find('^')
            if s[pos+1] != '{':
                exp = s[pos+1]
                rest = s[pos+2:]
                s = s[:pos]
            else:
                close = s.find('}', pos)
                exp = s[pos+2:close]
                rest = s[close+1:]
                s = s[:pos]

        else:
            rest = ''

        if exp is None:
            tex = r"%s^{*}" % self._print(expr.args[0])
        else:
            tex = s + '^{' + exp + '\\,*}' + rest
        return tex

    def _print_transpose(self, expr):
        matStr = self._print(expr.args[0])

        if '^' in matStr:
            return r"{%s}^{\trans}" % matStr
        else:
            return r"%s^{\trans}" % matStr


    def _print_Delta(self, expr):
        i,j,k,l = eval(expr.name)

        tex = r'\delta_{%s %s}\delta_{%s %s}' % (i, j, k, l)
        return tex

    def _print_MatrixBase(self, expr):
        tex = super()._print_MatrixBase(expr)
        tex = tex.replace('{matrix}', '{pmatrix}').replace(r'\left[', '').replace(r'\right]', '')
        return tex

    def _print_Identity(self, I):
        return r"\mathbbm{1}_{" + str(I.shape[0]) + "}"

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_Function(self, expr):
        s = '\\mathrm{' + expr.name + '} \\left('

        args = []
        for el in expr.args:
            if str(el) in self.latex.dynkin:
                args.append(self.latex.dynkin[str(el)])
            elif str(el) in self.model.Particles:
                args.append(self._print(self.model.Particles[str(el)]._name))
            else:
                args.append(str(el))

        s += ', '.join(args)
        s += '\\right)'

        return s

    def _print_adjoint(self, expr):
        mat = expr.args[0]
        matStr = self._print(mat)

        if '^' in matStr:
            return r"{%s}^{\dagger}" % matStr
        else:
            return r"%s^{\dagger}" % matStr

    def sortTerms(self, args):
        if self.cType == 'GaugeCouplings':
            priority = {'GaugeCouplings': 0,
                        'Yukawas': 10,
                        'QuarticTerms': 1}
        elif self.cType == 'Yukawas':
            priority = {'GaugeCouplings': 10,
                        'Yukawas': 0,
                        'QuarticTerms': 1}
        elif self.cType == 'QuarticTerms':
            priority = {'GaugeCouplings': 1,
                        'Yukawas': 10,
                        'QuarticTerms': 0}
        elif self.cType == 'TrilinearTerms':
            priority = {'GaugeCouplings': 1,
                        'Yukawas': 100,
                        'QuarticTerms': 10,
                        'FermionMasses': 1000,
                        'TrilinearTerms': 0}
        elif self.cType == 'ScalarMasses':
            priority = {'GaugeCouplings': 1,
                        'Yukawas': 10000,
                        'QuarticTerms': 100,
                        'FermionMasses': 1000,
                        'ScalarMasses': 0,
                        'TrilinearTerms': 10}
        elif self.cType == 'FermionMasses':
            priority = {'GaugeCouplings': 1,
                        'Yukawas': 10,
                        'QuarticTerms': 100,
                        'FermionMasses': 0,
                        'TrilinearTerms': 1000}
        elif self.cType == 'Vevs':
            priority = {'GaugeCouplings': 1,
                        'Yukawas': 10,
                        'QuarticTerms': 100,
                        'FermionMasses': 0,
                        'TrilinearTerms': 1000,
                        'Vevs': -10}


        sortedTypes = sorted(priority.items(), key=lambda k: k[1])
        sortedTypes = {k:i for i, (k,_) in enumerate(sortedTypes)}

        def sortFunc(term):
            if term.find(Delta) != set():
                return float("inf"), ()

            atoms = []
            for el in [subTerm for subTerm in splitPow(term) if not subTerm.is_number]:
                if isinstance(el, Trace):
                    for ell in splitPow(el.args[0]):
                        atoms.append(list(ell.atoms())[0])
                elif isinstance(el, Symbol):
                    atoms.append(el)
                else:
                    atoms.append(list(el.atoms())[0])
            trace = 'Trace(' in str(term)

            # Remove IdentityMatrices from atoms
            #  + Remove the gauge parameter 'xi' when dealing with vevs
            atoms = [el for el in atoms if not (hasattr(el, 'is_Identity') and el.is_Identity) and not str(el)=='_xiGauge' and not str(el) == 'n_g']

            score = sum([priority[Printer.model.allCouplings[str(el)][0]] for el in atoms])

            if trace:
                score += .5

            sortByCouplingName = []
            for _ in priority:
                sortByCouplingName.append([])

            for el in atoms:
                cType = Printer.model.allCouplings[str(el)][0]
                pos = Printer.model.couplingsPos[cType][str(el)]
                sortByCouplingName[sortedTypes[cType]].append(pos)

            return score, sortByCouplingName

        return sorted(args, key=sortFunc)


    def splitEquation(self, eq):
        finalStr = ""
        if type(eq) == list or type(eq) == tuple:
            for i, el in enumerate(eq):
                tmp = self.latex.totex(el, baseAdd=True).replace('+ -', '-')

                if tmp[0] != '-':
                    tmp = '+ ' + tmp
                else:
                    tmp = '- ' + tmp[1:]

                tmp = '\n' + tmp

                finalStr += tmp.replace('+', '\n+').replace('-', '\n-')

        return finalStr

    def generateCGCexpr(cgcName, cgc):
        inds = [Symbol(c) for c in 'ijkl'[:cgc.dim]]
        fields = [IndexedBase(c) for c in 'abcd'[:cgc.dim]]

        cgcName = '{' + cgcName + '}^{' + ','.join([str(i) for i in inds]) + '}'
        cgcName += '\\, ' + ' '.join([str(f)+'_'+str(i) for f,i in zip(fields, inds)])
        expr = 0
        for k,v in cgc.dic.items():
            coeff = v
            terms = []
            for pos,ind in enumerate(k):
                if ind is not None:
                    terms.append(fields[pos][ind+1])
            tmp = Mul(coeff, *terms, evaluate=False)
            expr += tmp

        return cgcName, expr


def isMinus(expr):
    return (NegativeOne in expr.args or -1 in expr.args or (not expr.args[0].is_complex and expr.args[0].is_number and expr.args[0] < 0))

class Delta(Symbol):
    def __new__(self, *args):
        obj = Symbol.__new__(self, str(tuple([str(el) for el in args])))
        return obj
