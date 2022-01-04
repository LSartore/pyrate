from sympy import (Expr, sympify, Basic, MatrixBase, Mul, flatten,
                   Add, transpose, adjoint, conjugate, expand, Symbol, Pow)

from sys import exit

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


def isTranspose(symb):
    if isinstance(symb, Mul) and symb.args[0] == -1:
        return isinstance(symb.args[1], transpose)
    return isinstance(symb, transpose)

def isConjugate(symb):
    if isinstance(symb, Mul) and symb.args[0] == -1:
        return isinstance(symb.args[1], conjugate)
    return isinstance(symb, conjugate)

def isAdjoint(symb):
    if isinstance(symb, Mul) and symb.args[0] == -1:
        return isinstance(symb.args[1], adjoint)
    return isinstance(symb, adjoint)

def getName(symb):
    if isinstance(symb, Mul) and symb.args[0] == -1:
        return getName(symb.args[1])

    if isTranspose(symb) or isConjugate(symb) or isAdjoint(symb):
        return symb.args[0].name

    return symb.name


def sortYukTrace(expr, yukPos, depth=0):
    expr = expand(expr)
    subTerms = flatten(expr.as_coeff_mul())
    coeff = Mul(*subTerms[:-1])
    tr = subTerms[-1]

    if(depth > 10):
        print("\nERROR : SORT TRACE too deep")
        print(expr)
        exit()


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
        transp = [isTranspose(el) for el in args]
        conj = [isConjugate(el) for el in args]

        chooseTranspose = False
        if all(transp):
            args = tuple([transpose(el) for el in args])[::-1]
        elif any(transp) or any(conj):
            transposed = tuple([transpose(el) for el in args])[::-1]
            # if (not any([isinstance(el, transpose) for el in transposed])
            # and not any([isinstance(el, conjugate) for el in transposed])):
            if (not any([isTranspose(el) for el in transposed])
            and not any([isConjugate(el) for el in transposed])):
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

        newArgs = sorted(set([args[-i:] + args[:-i]  for i in range(len(args))]),
                          key=lambda x:[yukSortKey(y, yukPos) for y in x])[0]

        if chooseTranspose:
            newTransposed = sorted(set([transposed[-i:] + transposed[:-i]  for i in range(len(transposed))]),
                              key=lambda x:[yukSortKey(y, yukPos) for y in x])[0]

            newArgs = (newArgs, newTransposed)[ [str(newArgs), str(newTransposed)].index(sorted((str(newArgs), str(newTransposed)))[0]) ]

        return coeff * trace(Mul(*newArgs))

    except AttributeError:
        return sortYukTrace(trace(expand(tr.args[0])), yukPos, depth=depth+1)

def yukSortKey(term, yukPos):
    if isAdjoint(term):
        pos = yukPos[getName(term)]
        return (1, pos)
    elif isTranspose(term):
        pos = yukPos[getName(term)]
        return (2, pos)
    elif isConjugate(term):
        pos = yukPos[getName(term)]
        return (1, pos)
    else:
        pos = yukPos[getName(term)]
        return (3, pos)


