from sympy import Wild

import itertools
from .Math import isZero, expand
from .Symbols import mMul
from .Trace import trace, sortYukTrace



class TensorDic(dict):
    def __new__(self, *args, **kwargs):
        return dict.__new__(self)
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
        self.tilde = False
        self.tildeRef = None

    def __getitem__(self, t):
        if self.tilde:
            ret = self.tildeRef[dict.__getitem__(self, t)]
            
            if isZero(ret):
                del self[t]
                return 0

            return ret
        else:
            ret = dict.__getitem__(self, t)
            if ret is not None:
                return ret

            subs = {k:v for k,v in zip(self.kwargs['freeDummies'], t)}
            kwargs = {k:v for k,v in self.kwargs.items() if k != 'freeDummies' and k != 'dont'}
            args = [tuple([el[0]] + [(ell if ell not in subs else subs[ell]) for ell in el[1:]]) for el in self.args]
            ret = tensorContract(*args, **kwargs)
        
            if isZero(ret):
                del self[t]
                return 0
            self[t] = ret
            return ret

class Tensor():
    def __init__(self, ranges, dic=None, sym=False):
        self.dim = len(ranges)
        self.range = tuple(ranges)
        self.dic = dict() if dic == None else dic
        self.sym = sym

    # def createSparseTensor(self, dim, ran, density):
    #     iterator = itertools.product( *([range(r) for r in ran]) )
        
    #     dic = {}
    #     for tup in iterator:
    #         if random.random() < density:
    #             dic[tup] = random.randint(-10,10)
                
    #     return dic
    
    def iMatch(self, inds, dummySubs = {}, freeDummies=[]):
        """ Returns all indices matching a given form.
            e.g. T(1,i,j,2) -> all indices with T(1,#,#,2) will match """
        # print("DummySubs : ", dummySubs)

        dummies = {}
        contractedDummies = {}
        nonDummyPos = []
        for pos,i in enumerate(inds):
            if isinstance(i, Wild) and i in dummySubs:
                inds[pos] = dummySubs[i]
            if not isinstance(i, Wild) or i in dummySubs:
                nonDummyPos.append(pos)
            else:
                if not i in dummies:
                    dummies[i] = pos
                else:
                    if i not in contractedDummies:
                        contractedDummies[i] = (dummies[i], pos)
                        del dummies[i]
                    else:
                        print(f"Error: Index {i} cannot appear more than two times here.")
                        exit()
        
        dummList = list(dummies.keys())
        retList = []
        
        if not self.sym:
            for k in list(self.dic.keys()):
                for pos in nonDummyPos:
                    if k[pos] != inds[pos]:
                        break
                else:
                    for couplePos in contractedDummies.values():
                        if k[couplePos[0]] != k[couplePos[1]]:
                            break
                    else:
                        v = self.dic[k]
                        retList.append( (v, {**{d:k[p] for d,p in dummies.items()}, **dummySubs}) )
                    
        else:
            permFactor = 1
            for k in list(self.dic.keys()):
                symRemain = list(k)
                nonDummy = []
                for pos in nonDummyPos:
                    if inds[pos] not in symRemain:
                        break
                    else:
                        symRemain.remove(inds[pos])
                        nonDummy.append(inds[pos])
                else:
                    # print(symRemain)
                    for couplePos in contractedDummies.values():
                        if symRemain[0] not in symRemain[1:]:
                            break
                        else:
                            symRemain = symRemain[1:]
                            symRemain.remove(symRemain[0])
                    else:
                        # print(symRemain, nonDummy)
                        # permFactor = len(set(itertools.permutations(nonDummy)))
                        v = self.dic[k]
                        for perm in set(itertools.permutations(symRemain)):
                            # print(perm)
                            retList.append( (v*permFactor, {**{dummList[i]:dVal for i,dVal in enumerate(perm)}, **dummySubs}) )
                    
        return retList        
        
    def __repr__(self, content=False):
        s = f"Tensor of order {self.dim} with ranges {self.range}"
        if content:
            s += ": \n" + str(self.dic)
        return s

    def __getitem__(self, inds):
        if inds not in self.dic:
            return 0
        return self.dic[inds]
    
    def __call__(self, *args):
        return (self, *args)
    
    def __eq__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Cannot compare Tensor with object of type " + str(type(other)))
        if self.dim != other.dim or self.range != other.range or self.sym != other.sym:
            return False
        if self.dic != other.dic:
            return False
        return True
        
def readContraction(*tensorsWithInds, depth=0):
    nTensors = len(tensorsWithInds)
    
    tensors = [t[0] for t in tensorsWithInds]
    indices = [list(t[1:]) for t in tensorsWithInds]
    
    #Identify indices summed over, and their position
    dummies = {}
    
    for p1, ind in enumerate(indices):
        for p2, i in enumerate(ind):
            if isinstance(i, Wild):
                if not i in dummies:
                    dummies[i] = [(p1,p2, tensors[p1].range[p2])]
                elif len(dummies[i]) == 1:
                    if tensors[p1].range[p2] != dummies[i][0][2]:
                        raise ValueError(f"Inconsistent ranges for indice {i} : {tensors[p1].range[p2]} and {dummies[i][0][2]} .")
                    dummies[i].append((p1,p2, tensors[p1].range[p2]))
                else:
                    print(f"Error: dummy index {i} appears more than twice...")
                    break
    
    freeDummies = []
    if depth == 0:
        for k,v in dummies.items():
            if len(v) == 1:
                freeDummies.append(k)
            # if len(dum) != 2:
            #     print(f"Error: dummy index {i} does not appear twice... ({len(dum)})")
            #     exit()

    return nTensors, tensors, indices, freeDummies
    

def tensorContract(*tensorsWithInds, depth=0, value=1, dummySubs={}, freeDummies=[], doTrace=False, yukSorting=None, expandExpr=False, verbose=False, doit=False):
    n, tensors, indices, freeD = readContraction(*tensorsWithInds, depth=depth)
    
    if depth == 0 and freeDummies == []:
        freeDummies = freeD
        doit = True

    # print("Indices : ", indices)
        
    # dummyReplace, matches = 0,0
    # print("\n\n", tensors[0])
    # print(indices[0])

    # print("Depth =", depth, " DOIT ? ", doit)
    if not doit:
        if n == 0:
            return None
    
        if freeDummies == []:
            pass
        else:
            result = dict() if depth != 0 else TensorDic(*tensorsWithInds, freeDummies=freeDummies, doTrace=doTrace, yukSorting=yukSorting, expandExpr=expandExpr, verbose=verbose, doit=doit)
            for _,subs in tensors[0].iMatch(indices[0], dummySubs=dummySubs):
                if verbose:
                    print(depth*"## ", subs)
                tmp = tensorContract(*tensorsWithInds[1:], depth=depth+1, value=(), dummySubs=subs, freeDummies=freeDummies, verbose=verbose, doit=doit)
                
                if isinstance(tmp, dict):
                    for k in tmp.keys():
                        if k not in result:
                            result[k] = None
                else:
                    key = tuple([subs[fd] for fd in freeDummies])
                    if key not in result :
                        result[key] = None
            
            return result

    else:
        if n == 0:
            if not isZero(value):
                if doTrace:
                    value = trace(value)
                    if yukSorting:
                        value = sortYukTrace(value, yukSorting)
            # print("ENDchain : ", indices)
            return value
            
        # dummyReplace, matches = 0,0
        # print("\n\n", tensors[0])
        # print(indices[0])
        
        if freeDummies == []:
            result = 0
            for val in tensors[0].iMatch(indices[0], dummySubs=dummySubs, freeDummies=freeDummies):
                # print(depth*"  ", val)
                if verbose:
                    print(depth*"## ", val)
                    # print(tensorsWithInds[1:])
                # tmp = tensorContract(*tensorsWithInds[1:], depth=depth+1, value=value*val[0], dummySubs=val[1], freeDummies=freeDummies, doTrace=doTrace, yukSorting=yukSorting, verbose=verbose)
                tmp = tensorContract(*tensorsWithInds[1:], depth=depth+1, value=mMul(value, val[0]), dummySubs=val[1], freeDummies=freeDummies, doTrace=doTrace, yukSorting=yukSorting, verbose=verbose, doit=doit)
                    
                if not isZero(tmp):
                    if result == 0:
                        result = tmp
                    else:
                        result += tmp
        else:
            result = {}
            for val in tensors[0].iMatch(indices[0], dummySubs=dummySubs):
                if verbose:
                    print(depth*"## ", val)
                # print(val, value, val[0])
                # tmp = tensorContract(*tensorsWithInds[1:], depth=depth+1, value=value*val[0], dummySubs=val[1], freeDummies=freeDummies, doTrace=doTrace, yukSorting=yukSorting, verbose=verbose)
                tmp = tensorContract(*tensorsWithInds[1:], depth=depth+1, value=mMul(value, val[0]), dummySubs=val[1], freeDummies=freeDummies, doTrace=doTrace, yukSorting=yukSorting, verbose=verbose, doit=doit)
                
                if type(tmp) == dict:
                    for k,v in tmp.items():
                        if isZero(v):
                            continue
                        if k not in result:
                            result[k] = v
                        else:
                            result[k] += v
                elif not isZero(tmp):
                    key = tuple([val[1][fd] for fd in freeDummies])
                    if key not in result :
                        result[key] = tmp
                    else:
                        result[key] += tmp
            
            for k in list(result.keys()):
                if isZero(result[k]):
                    del result[k]
                elif expandExpr:
                    result[k] = expand(result[k])
            
        return result



def tensorAdd(*dics):
    if len(dics)==1:
        return dics[0]
    
    retDic = {}
    allKeys = itertools.chain(*[d.keys() for d in dics])
    setAllKeys = set(allKeys)

    # print("Total length : ", len(setAllKeys))
    
    for k in setAllKeys:
        retDic[k] = 0
        for dic in dics:
            if k in dic:
                if isZero(retDic[k]):
                    retDic[k] = dic[k]
                else:
                    retDic[k] += dic[k]
        
        retDic[k] = expand(retDic[k])
        if isZero(retDic[k]):
            del retDic[k]
            
    return retDic


def tensorMul(n, dic):
    for k in dic:
        dic[k] = n*dic[k]
    return dic