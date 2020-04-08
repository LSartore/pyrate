from sympy import Symbol


class GaugeGroup():
    realBasis = ''

    def __init__(self, name, gpType, idb):
        self.idb = idb
        self.name = name
        self.type = gpType
        self.abelian = False
        self.g = Symbol(f'g_{self.name}', real=True)

        if self.type == 'U1':
            self.dim = 1
            self.abelian = True
            self.latex = 'U(1)'
            return

        self.repDic = {}

        self.dim = idb.get(self.type, 'dim')
        self.rank = idb.get(self.type, 'rank')
        self.sName = idb.get(self.type, 'name')

        self.structureConstants = idb.get(self.type, 'struct')

        # Latex name
        for i, c in enumerate(self.type):
            if c.isdigit():
                break

        base, n = self.type[:i], self.type[i:]
        if base not in ('E', 'F', 'G'):
            self.latex = base + '(' + str(n) + ')'
        else:
            self.latex = base + '_{' + n + '}'

    def dimR(self, rep):
        if rep not in self.repDic:
            self.computeRepInfo(rep)
        return self.repDic[rep][0]

    def repMat(self, rep):
        if rep not in self.repDic:
            self.computeRepInfo(rep)
        return self.repDic[rep][3]

    def repName(self, rep):
        if rep not in self.repDic:
            self.computeRepInfo(rep)
        return self.repDic[rep][4]


    def computeRepInfo(self, rep):
        """ Compute some useful info about the rep """
        labels = rep
        dim = self.idb.get(self.type, 'dimR', rep)
        repMats = self.idb.get(self.type, 'repMatrices', rep, realBasis=self.realBasis)
        fs = self.idb.get(self.type, 'frobenius', rep)
        tex = self.idb.get(self.type, 'repname', rep, latex=True)

        if fs == 1:
            repType = 'complex'
        elif fs == 0:
            repType = 'real'
        elif fs == -1:
            repType = 'pseudo-real'

        self.repDic[rep] = (dim, labels, repType, repMats, tex)




