# -*- coding: utf-8 -*-

from sys import exit
from Logging import loggingCritical

from sympy import Symbol, Mul, MatMul, conjugate, SparseMatrix, Matrix, diag

            
class mSymbol(Symbol):
    """ mSymbol(name, n, m, **assumptions) """
    
    def __new__(cls, *args, symmetric=False, hermitian=False, real=False):
        # print(args)
        if args[1] != args[2]:
            if symmetric or hermitian:
                loggingCritical("Matrix <" + args[0] + "> cannot be symmetric nor hermitian since it is not a square matrix.") 
                exit()

        if not args[1] == args[2] == 1:
            obj = Symbol.__new__(cls, args[0], commutative=False, hermitian=hermitian)
            obj.is_symmetric = symmetric
            obj.is_realMatrix = real
        else:
            obj = Symbol.__new__(cls, args[0], real=real)
        obj.shape = (args[1], args[2])
        return obj

    def _eval_transpose(self):
        if self.shape == (1,1) or self.is_symmetric:
            return self
        return super()._eval_transpose()
    
    def _eval_adjoint(self):
        if self.shape == (1,1) or self.is_symmetric:
            return conjugate(self)
        return super()._eval_adjoint()

    def _eval_conjugate(self):
        if self.shape != (1,1) and self.is_realMatrix:
            return self
        
        return super()._eval_conjugate()
    
    def as_explicit(self, applyFunc=None):
        if self.shape == (1,1):
            return self
        
        if isinstance(self.shape[0], Symbol) or isinstance(self.shape[1], Symbol):
            loggingCritical("Error : a Matrix with symbolic number of generations as rows and/or columns cannot be given as an explicit matrix.")
            exit()
        
        mat = SparseMatrix(*self.shape, {})
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                assump = {}
                if self.is_realMatrix or (self.is_hermitian and i==j):
                    assump['real'] = True
                else:
                    assump['complex'] = True
                
                
                if i <= j:
                    mat[i, j] = Symbol('{' + str(self) + '}_{' + str(i+1) + str(j+1) + '}', **assump)
                else:
                    if self.is_symmetric:
                        mat[i, j] = mat[j, i]
                    elif self.is_hermitian:
                        mat[i, j] = conjugate(mat[j, i])
                    else:
                        mat[i, j] = Symbol('{' + str(self) + '}_{' + str(i+1) + str(j+1) + '}', **assump)

        if applyFunc is None:
            return Matrix(mat)
        
        # The following case is needed in Python export
        retList = mat.tolist()
        for i, row in enumerate(retList):
            for j, el in enumerate(row):
                retList[i][j] = applyFunc(el)
        
        return retList

class mMul(Mul):
    """ This class extends the behavior or the Mul sympy class. It mimics matrix
    multiplication when dealing with non-commutative Yukawa couplings """
    
    def __new__(cls, *args, **kwargs):
        
        # Go through all args. If all Matrices or all scalars, nothing to do.
        # If mix of scalars and matrix, it means that some Yukawa coupling is 
        # multiplied with identity
        
        nonComm = False
        mat = False
        for i, el in enumerate(args):
            if (hasattr(el, 'is_Matrix') and el.is_Matrix):
                mat = True
            elif (hasattr(el, 'is_commutative') and not el.is_commutative): 
                nonComm = True
        
        if not (mat and nonComm):
            return Mul(*args)
        else:
            newArgs = []
            for i, el in enumerate(args):
                if hasattr(el, 'is_Matrix') and el.is_Matrix:
                    if hasattr(el, 'is_Identity') and el.is_Identity:
                        pass
                    elif isinstance(el, MatMul):
                        for ell in el.args:
                            if not (hasattr(ell, 'is_Identity') and ell.is_Identity):
                                newArgs.append(ell)
                else:
                    newArgs.append(el)
            obj = Mul(*newArgs)
            
            return obj


class Identity(mSymbol):
    is_Identity = True
    is_Matrix = True
    
    def __new__(cls, n):
        return mSymbol.__new__(cls, "I", n, n)

    def _eval_transpose(self):
        return self

    def _eval_adjoint(self):
        return self

    def _eval_trace(self):
        return self.shape[0]

    def _eval_inverse(self):
        return self

    def conjugate(self):
        return self

    def _eval_determinant(self):
        return 1

    def as_explicit(self):
        return diag(*([1]*self.shape[0]))