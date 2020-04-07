from sympy import (Expr, sympify, Basic, MatrixBase, Mul, flatten,
                   Add, transpose, adjoint, conjugate, expand, Symbol, Pow, Identity)

class Trace(Expr):
    is_Trace = True
    is_commutative = True
    
    def __new__(cls, mat):
        mat = sympify(mat)
        
        return Basic.__new__(cls, mat)

    def _eval_transpose(self):
        return self

    @property
    def arg(self):
        return self.args[0]

    def doit(self, **kwargs):
        if isinstance(self.arg, Symbol) and self.arg.is_commutative == True:
            return self.arg

        if kwargs.get('deep', True):
            arg = self.arg.doit(**kwargs)
            try:
                return arg._eval_trace()
            except (AttributeError, NotImplementedError):
                return Trace(arg)
        else:
            # _eval_trace would go too deep here
            if isinstance(self.arg, MatrixBase):
                return trace(self.arg)
            else:
                return Trace(self.arg)

def trace(expr):
    if not (hasattr(expr, 'is_Matrix') and expr.is_Matrix) and not (hasattr(expr, 'is_commutative') and not expr.is_commutative):
        return expr
    
    ret = Trace(expr).doit()
    
    return ret




# def vectorShaped(sortFunc):
#     """ This is a decorator to systematically check for products of vector-shaped
#         Yukawa matrices after the trace sorting. Reorganizes the trace accordingly"""
        
#     def inner(*args, **kwargs):
#         expr = sortFunc(*args, **kwargs)
        
#         return expr
#         if not Trace.handleVectorShaped:
#             return expr
        
#         args = [el for el in expr.args if isinstance(el, Trace)][0].args[0].args
        
#         shapes = []
#         for el in args:
#             c = list(el.atoms())[0]
            
#             if not isinstance(el, transpose) and not isinstance(el, adjoint):
#                 shapes.append(c.shape)
#             else:
#                 shapes.append(c.shape[::-1])
        
#         if 1 not in flatten(shapes):
#             return expr
    
#         coeff = Mul(*[el for el in expr.args if not isinstance(el, Trace)])
        
            
#         print("\nSorted trace : ", expr)
#         # print('\tShapes before :', shapes)
#         while not (flatten(shapes)[0] == 1 and flatten(shapes)[-1] == 1):
#             args = args[-1:] + args[:-1]
#             shapes = shapes[-1:] + shapes[:-1]
        
#         # print("\tShapes after : ", shapes)
#         print("New trace : ", coeff*Mul(*args))
        
#         return coeff*Mul(*args)
#     return inner



# @vectorShaped
def sortYukTrace(expr, yukPos, depth=0):
    expr = expand(expr)
    subTerms = flatten(expr.as_coeff_mul())
    coeff = Mul(*subTerms[:-1])
    tr = subTerms[-1]
    
    # print("\n ### SORT TRACE ###")
    # print(expr)
    # print("\t", subTerms)
    # print("\t coeff : ", coeff)
    # print(tr)
    
    if(depth > 10):
        print("\nERROR : SORT TRACE too deep")
        print(expr)
        exit()
        
    # print(tr)
    # print(type(tr))
    # print(tr.args)
    # print(tr.args[0])
    # print(type(tr.args[0]))
    
    if isinstance(tr, Add):
        return coeff*sum([sortYukTrace(t, yukPos, depth=depth+1) for t in tr.args])
    elif isinstance(tr, Trace) and len(tr.args) == 1:
        if isinstance(tr.args[0], Mul):
            args = []
            
            # The commutative part contributes to coeff
            # The non commutative part (matrix) is extracted
            for el in tr.args[0].args:
                if el.is_commutative:
                    coeff *= el
                else:
                    args.append(el)
            args = tuple(flatten([(arg if not isinstance(arg, Pow) else [arg.args[0]]*arg.args[1]) for arg in args]))

        elif isinstance(tr.args[0], Add):
            return coeff*sum([sortYukTrace(trace(t), yukPos, depth=depth+1) for t in tr.args[0].args])
        elif isinstance(tr.args[0], Pow):
            args = tuple([tr.args[0].args[0]]*tr.args[0].args[1])
        else:
            print(expr)
            raise NotImplementedError("Trace sorting error : Not implemented for type " + str(type(tr.args[0])))
        
    elif not isinstance(tr, Trace):
        return expr
    
    try:
        # args = tr.args[0].args
        
        # print("\n\n Args : ", args)
        # transp = [isinstance(el, transpose) or isinstance(el, conjugate) for el in args]
        transp = [isinstance(el, transpose) for el in args]
        # print('\t TRANSP : ', transp)
        chooseTranspose = False
        # print("Transp : ", transp)
        if all(transp):
            args = tuple([transpose(el) for el in args])[::-1]
        elif any(transp):
            transposed = tuple([transpose(el) for el in args])[::-1]
            if not any([isinstance(el, transpose) for el in transposed]):
                args = transposed
            else:
                count = str(args).count('transpose') + str(args).count('conjugate')
                countTransposed = str(transposed).count('transpose') + str(transposed).count('conjugate')
                if count < countTransposed:
                    pass
                elif count > countTransposed:
                        args = transposed
                else:
                    chooseTranspose = True
            
            
        
        # print("Args : ", args)
        
        newArgs = sorted(set([args[-i:] + args[:-i]  for i in range(len(args))]), 
                          key=lambda x:[yukSortKey(y, yukPos) for y in x])[0]
        
        if chooseTranspose:
            newTransposed = sorted(set([transposed[-i:] + transposed[:-i]  for i in range(len(transposed))]), 
                              key=lambda x:[yukSortKey(y, yukPos) for y in x])[0]
            
            newArgs = (newArgs, newTransposed)[ [str(newArgs), str(newTransposed)].index(sorted((str(newArgs), str(newTransposed)))[0]) ]
        
        # print("NEW ARGS : ", newArgs)
        return coeff * trace(Mul(*newArgs))
    
    except AttributeError:
        return sortYukTrace(trace(expand(tr.args[0])), yukPos, depth=depth+1)
    
def yukSortKey(term, yukPos):
    if isinstance(term, adjoint):
        pos = yukPos[term.args[0].name]
        return (2, pos)
    elif isinstance(term, transpose):
        pos = yukPos[term.args[0].name]
        return (3, pos)
    elif isinstance(term, conjugate):
        pos = yukPos[term.args[0].name]
        return (2, pos)
    else:
        pos = yukPos[term.name]
        return (1, pos)


        