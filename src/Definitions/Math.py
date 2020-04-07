from sympy import MatAdd, MatMul, flatten
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
    # print("\n EXPAND HERE. Depth =",depth," ; EXPR = ", expr)
    # print("\t\t type = ", type(expr))
    # time.sleep(.2)
    if not isinstance(expr, MatMul) and not isinstance(expr, MatAdd):
        return sympyExpand(factor*expr, depth=depth+1)
    else:
        if isinstance(expr, MatAdd):
            # ret = []
            # for arg in expr.args:
            #     ret.append(expand(arg, factor=factor, depth=depth+1))
            # return ret
            ret = []
            for arg in expr.args:
                tmp = expand(arg, factor=factor, depth=depth+1)
                if not isZero(tmp):
                    ret.append(tmp)
            return MatAdd(*ret)
        
        # print(expr.args)
        
        scalar = factor
        mat = []
        
        for el in expr.args:
            if not (hasattr(el, 'is_Matrix') and el.is_Matrix):
                scalar *= el
            else:
                mat.append(el)
        # print("SCALAR : ", scalar)
        # scalar = [el for el in flatten(expand(scalar, depth=depth+1).as_coeff_add()) if el != 0]
        scalar = [el for el in flatten(sympyExpand(scalar).as_coeff_add()) if el != 0]
        # print("SCALAR-LIST : ", scalar)
        
        if len(scalar) == 0:
            return 0
        if len(scalar) > 1:
            # print("SCALAR : ", scalar)
            # print([s*MatMul(*mat) for s in scalar if s != 0])
            ret = []
            for s in scalar:
                tmp = expand(MatMul(*mat), factor=s, depth=depth+1)
                if not isZero(tmp):
                    ret.append(tmp)
            
            return MatAdd(*ret)
        
        add = []
        toDistribute = False
        for el in mat:
            # print(el)
            el = expand(el, depth=depth+1)
            if isinstance(el, MatAdd):
                add.append(el.args)#expand(el, factor=scalar, depth=depth+1)]
                toDistribute = True
            else:
                add.append([el])
        
        # print("ToDistribute : ", toDistribute)
        if toDistribute == False:
            # print("NO DISTRIBUTE !! ")
            # print(mat)
            # print([type(el) for el in mat])
            # print([el.args for el in mat])
            
            # print(scalar)
            # print(expr)
            ret = scalar[0]
            
            for el in mat:
                if not isinstance(el, MatMul):
                    ret *= el
                else:
                    for ell in el.args:
                        ret *= ell

            return ret
            
        add = itertools.product(*add)
        # print("\n\n" + (depth+1)*"##")
        
        # print(list(add))
        
        ret = []
        
        for el in add:
            # print("\n", el)
            if len(el) == 1:
                tmp = expand(el[0], factor=scalar[0], depth=depth+1)
            else:
                tmp = expand(MatMul(*el), factor=scalar[0], depth=depth+1)
            
            if tmp == 0:
                continue
            
            ret.append(tmp)
        return MatAdd(*ret)
        # if 
        