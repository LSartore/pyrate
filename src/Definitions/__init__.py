# -*- coding: utf-8 -*-

from .GaugeGroup import GaugeGroup
from .Math import expand
from .Symbols import mSymbol, mMul, Identity
from .Tensors import TensorDic, Tensor, tensorContract, tensorAdd, tensorMul
from .Trace import Trace, trace

from sympy import flatten, Function, Mul, Pow, Symbol, Wild


# Definition of global dummy indices across all modules
# No need to import / redefine them elsewhere
# All are in the form ' x_ '
import builtins

wildSymbs = ['A','B','C','D','E','F','G',
             'a','b','c','d','e','f','g','h',
             'i','j','k','l','m','n','o','p',
             'q','r','s','t']

for s in wildSymbs:
    exec('builtins.'+s+'_ = Wild(\''+s+'\')')



def splitPow(expr, deep=False):
    if type(expr) == list or type(expr) == tuple:
        coeff = Mul(*[el for el in expr if el.is_number])
        if coeff != 1:
           return [coeff] + splitPow([el for el in expr if not el.is_number], deep=deep)
        else:
           return flatten([splitPow(el, deep=deep) for el in expr if not el.is_number])
    if isinstance(expr, Pow):
        res = expr.args[1]*[expr.args[0]]
        if not deep:
            return res
        return flatten([splitPow(el, deep=deep) for el in res])
    if isinstance(expr, Mul):
        return splitPow(expr.args, deep=deep)
    if isinstance(expr, Trace):
        return [expr]
    else:
        return [expr]


def replaceKey(dic, oldKey, newKey, newVal = None):
    newDic = {}

    if type(newKey) != tuple:
        for k,v in dic.items():
            if k != oldKey:
                newDic[k] = v
            else:
                newDic[newKey] = newVal if newVal is not None else v
    else:
        for k,v in dic.items():
            if k != oldKey:
                newDic[k] = v
            else:
                for i, key in enumerate(newKey):
                    newDic[key] = newVal[i] if newVal is not None else v

    return newDic

def insertKey(dic, afterWhich, newKey, newVal):
    newDic = {}

    for k,v in dic.items():
        newDic[k] = v
        if k == afterWhich:
            newDic[newKey] = newVal

    return newDic
