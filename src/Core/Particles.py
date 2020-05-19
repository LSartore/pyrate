#!/usr/bin/env python
from sympy import Symbol, I

from Logging import loggingCritical

from sympy.parsing.sympy_parser import parse_expr
from copy import copy

from Definitions import GaugeGroup


class Particle(object):

    def __init__(self, name, dic, gaugeGroups, idb, fromCplx=False):
        if type(name) != Symbol:
            self._name = Symbol(name)
        else:
            self._name = name
        self.idb = idb
        self.groups = gaugeGroups
        self.Qnb = self.getQnb(dic['Qnb'], gaugeGroups)
        self.gen = self.getGen(dic)
        self.cplx = False
        self.fromCplx = fromCplx
        self.conj = False

        self.indicesRange = {}
        self.indexStructure = []
        self.fullIndexStructure = []

        for g, r in self.getIndicesRange(gaugeGroups).items():
            if r > 1:
                self.indicesRange[g] = r
                self.indexStructure.append(r)
            self.fullIndexStructure.append(r)

        self.indexStructure = tuple(self.indexStructure)
        self.fullIndexStructure = tuple(self.fullIndexStructure)

    def __repr__(self):
        return str(self._name)

    def getGen(self, dic):
        """convert the Gen into either a symbol or a number"""
        if 'Gen' not in dic:
            return 1
        try:
            return Symbol(dic['Gen'])
        except:
            return int(dic['Gen'])

    def getQnb(self, dic, gaugeGroups):
        """Get the Qnbs of the particle from the dic. The only thing to do is to transform the DimR notation into DynkinLabels"""

        for k,v in dic.items():
            g = gaugeGroups[k]

            if isinstance(v, str):
                if not g.abelian:
                    loggingCritical(f"Error while reading particle {self.name} : for non-abelian " +
                                    "gauge factors, quantum number must be integers or dynkin labels.")
                    exit()

                v = parse_expr(v.replace('i','I').replace('Sqrt','sqrt'))

            if isinstance(v, list):
                v = tuple(v)
            if not isinstance(v, tuple) and not g.abelian:
                dynk = tuple(self.idb.get(g.type, 'dynkinLabels', v, realBasis=GaugeGroup.realBasis))
                if type(dynk[0]) == list:
                    loggingCritical(f"Error : more than one representation of the group {g.type} have dimension {dynk} :")
                    loggingCritical(' -> ' + ', '.join([str(el) for el in dynk[:-1]]) + ' and ' + str(dynk[-1]))
                    loggingCritical("Please use the Dynkin-labels notation instead to remove the ambihuity.")
                    exit()

                v = dynk

            dic[k] = v

        return dic

    def getIndicesRange(self, gaugeGroups):
        ranges = {}
        for gName, g in gaugeGroups.items():
            if not g.abelian:
                dim = self.idb.get(g.type, 'dimR', self.Qnb[gName])
                ranges[gName] = dim

        return ranges

    def antiParticle(self):
        antiP = copy(self)
        antiP._name = Symbol(str(self._name)+'bar')
        antiP.conj = True
        antiP.Qnb = {}

        for gName, qnb in self.Qnb.items():
            g = self.groups[gName]

            if not g.abelian:
                antiP.Qnb[gName] = self.idb.get(g.type, 'conjugate', qnb, realBasis=GaugeGroup.realBasis)
            else:
                antiP.Qnb[gName] = -1 * qnb

        return antiP

class ComplexScalar(Particle):
    def __init__(self, name, dic, Groups, idb):
        self._name = Symbol(name)
        self.realFields = [Particle(n, dic, Groups, idb, self) for n in dic['RealFields']]
        self.realComponents = [1, I]

        # call the particle constructor
        self.norm = dic['Norm']
        dic['Gen'] = 1
        self.idb = idb
        Particle.__init__(self, name, dic, Groups, self.idb, False)
        self.cplx = True

    def antiParticle(self):
        antiP = Particle.antiParticle(self)
        antiP.realComponents = [1,-I]

        return antiP
