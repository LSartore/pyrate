from sympy import MatAdd, MatMul, Mul, flatten
from sympy import expand as sympyExpand

import itertools

def isZero(expr):
    if (expr == 0 or
      (hasattr(expr, 'is_Zero') and expr.is_Zero) or
      (hasattr(expr, 'is_ZeroMatrix') and expr.is_ZeroMatrix) ):
        return True
    return False


def flattenAdd(expr, depth=0):
    if not isinstance(expr, MatAdd):
        return expr
    else:
        ret = []
        for term in [el for el in flatten(expr.as_coeff_add()) if el != 0]:
            ret.append(flattenAdd(term, depth=depth+1))

        if depth != 0:
            return ret

        # Simplification step
        res = 0
        for el in flatten(ret):
            if res == 0:
                res = el
            else:
                res += el
        return res

def expand(expr, factor=1, depth=0):
    return sympyExpand(expr)
    if depth == 0:
        return flattenAdd(expand(expr, depth=depth+1))

    if not isinstance(expr, MatMul) and not isinstance(expr, MatAdd):
        return sympyExpand(factor*expr, depth=depth+1)
    else:
        if isinstance(expr, MatAdd):
            ret = []
            for arg in expr.args:
                tmp = expand(arg, factor=factor, depth=depth+1)
                if not isZero(tmp):
                    ret.append(tmp)
            return MatAdd(*ret)

        scalar = factor
        mat = []

        for el in expr.args:
            if not (hasattr(el, 'is_Matrix') and el.is_Matrix):
                scalar *= el
            else:
                mat.append(el)

        scalar = [el for el in flatten(sympyExpand(scalar).as_coeff_add()) if el != 0]

        if len(scalar) == 0:
            return 0
        if len(scalar) > 1:
            ret = []
            for s in scalar:
                tmp = expand(MatMul(*mat), factor=s, depth=depth+1)
                if not isZero(tmp):
                    ret.append(tmp)

            return MatAdd(*ret)

        add = []
        toDistribute = False
        for el in mat:
            el = expand(el, depth=depth+1)
            if isinstance(el, MatAdd):
                add.append(el.args)
                toDistribute = True
            else:
                add.append([el])

        if toDistribute == False:
            ret = scalar[0]

            for el in mat:
                if not isinstance(el, MatMul):
                    ret *= el
                else:
                    for ell in el.args:
                        ret *= ell

            return ret

        add = itertools.product(*add)
        ret = []

        for el in add:
            if len(el) == 1:
                tmp = expand(el[0], factor=scalar[0], depth=depth+1)
            else:
                tmp = expand(MatMul(*el), factor=scalar[0], depth=depth+1)

            if tmp == 0:
                continue

            ret.append(tmp)
        return MatAdd(*ret)


def flattenedTensorPos(ranges, indices):
    weights = [1]
    for i in range(len(ranges)-1):
        weights.insert(0, Mul(*ranges[-1-i:]))

    return sum([weights[i]*indices[i] for i in range(len(ranges))])

def flattenedTensorInds(ranges, pos):
    N = len(ranges)
    ret = [0 for _ in range(N)]

    i = N-1
    p = pos
    while i >= 0:
        (p, ret[i]) = (p // ranges[i], p % ranges[i])
        i -= 1

    return tuple(ret)

