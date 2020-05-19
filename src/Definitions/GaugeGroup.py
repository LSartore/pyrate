from sympy import Symbol


class GaugeGroup():
    realBasis = ''

    def __init__(self, name, gpType, idb):
        self.idb = idb
        self.name = name
        self.type = gpType.upper()
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


    def computeRepInfo(self, rep, noRepMats=False):
        """ Compute some useful info about the rep """
        labels = rep
        dim = self.idb.get(self.type, 'dimR', rep)
        if not noRepMats:
            repMats = self.idb.get(self.type, 'repMatrices', rep, realBasis=self.realBasis)
        else:
            repMats = []
        fs = self.idb.get(self.type, 'frobenius', rep)
        tex = self.idb.get(self.type, 'repname', rep, latex=True)
        index = self.idb.get(self.type, 'dynkinIndex', rep)

        if fs == 1:
            repType = 'complex'
        elif fs == 0:
            repType = 'real'
        elif fs == -1:
            repType = 'pseudo-real'

        self.repDic[rep] = (dim, labels, repType, repMats, tex, index)


    def moreGroupInfo(self, N=10):
        """ Retrieve info about the first M irreps of the gauge group
            spanning the first possible N dimensions that the reps of the
            group may have (M >= N). """

        try:
            self.idb.load()
            a = self.idb.get(self.type)

            # Identify the reps
            dims = set()
            maxDim = 0
            step = self.idb.get(self.type, 'dim')
            depth = 1

            while len(dims) != N+1:
                reps = a.repsUpToDimN(maxDim)
                dims = set()
                for r in reps:
                    dims.add(self.idb.get(self.type, 'dimR', r))

                if len(dims) < N+1:
                    if depth == 1:
                        maxDim += round(step/depth)
                    else:
                        depth += 1
                        maxDim += round(step/depth)
                if len(dims) > N+1:
                    depth += 1
                    maxDim -= round(step/depth)

            # Remove the trivial representation
            reps = reps[1:]

            # Compute rep info
            for r in reps:
                self.computeRepInfo(tuple(r), noRepMats=True)

        except SystemExit:
            exit()
        finally:
            self.idb.close()


    def copy(self):
        return GaugeGroup(self.name, self.type, self.idb)
