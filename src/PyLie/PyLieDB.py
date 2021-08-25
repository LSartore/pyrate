# -*- coding: utf-8 -*-

import sys
import os
import time

wd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(wd)

from PyLie import LieAlgebra, CartanMatrix
from Math import sMat, sTensor

from sympy import Add, Pow, Rational, sqrt, I, Symbol

import numpy as np

import h5py
import gzip

class PyLieDB():
    """ This is the main class used to communicate with the Pylie module and
        the associated database. """

    def __init__(self, path=None, logLevel='Info', raiseErrors=False, altDB=False):
        global wd

        if path is not None:
            self.path = path
        elif altDB is True:
            self.path = os.path.join(wd, 'altPyLieDB.hd5f.gz')
        else:
            self.path = os.path.join(wd, 'PyLieDB.hd5f.gz')

        # self.tmpPath = self.path.replace('PyLieDB.hd5f.gz', '._PyLieDB.hd5f')
        self.tmpPath = os.path.join(os.path.dirname(self.path), '._' + os.path.basename(self.path).replace('.gz', ''))

        self.gzf = None
        self.f = None
        self.modified = False
        self.loaded = False

        self.algebras = {}

        self.logging = {}
        self.initLogging(logLevel)
        self.raiseErrors = raiseErrors

        self.basicTranslations = {
            'fullname' : lambda a: a.cartan._fullName,
            'name' : lambda a: a.cartan._name + str(a.cartan._id),
            'rank' : lambda a: a._n,
            'dimension' : lambda a: a.dimAdj,
            'cartanmatrix' : lambda a: a.cm,
            'adjointrep' : lambda a: a.adjoint
            }

        self.storedTranslations = {
            'structureconstants' : lambda a: a.structureConstants,
            'invariants' : lambda a : a.invariants,
            'repmatrices' : lambda a : a.repMatrices,
            'frobenius' : lambda a: a.frobeniusSchurIndicator,
            'conjugaterep' : lambda a: (lambda r: a.conjugateIrrep(r).tolist()),
            'dynkinlabels' : self.getDynkinLabels,
            'dimr' : self.dimR,
            'dynkinindex' : lambda a: a.dynkinIndex,
            }

        self.translations = {
            'conjugate' : self.conjugate,
            'repname' : self.repName,
            'repproduct': self.repProduct,
            'firstreps' : self.firstReps
            }


    ########################
    # Operations on the DB #
    ########################

    def load(self, force=False):
        if self.loaded:
            return
        busy = os.path.exists(self.tmpPath)

        if busy and not force:
            self.loggingCritical("Error : The database is already being used, exiting.")
            return

        created = not(os.path.exists(self.path))

        if not created:
             with gzip.open(self.path, 'rb') as gzFile, \
                       open(self.tmpPath, 'wb') as _bFile:
                _bFile.writelines(gzFile)

        self.f = h5py.File(self.tmpPath, 'a')

        if created:
            self.f.attrs['created'] = time.ctime()
            self.f.attrs['modified'] = ''
            self.modified = True

        self.loaded = True

    def push(self):
        """ Pushes the content of the .hd5f temporary file to the .gz DB file """

        self.close()
        self.load()

    def close(self):
        if self.modified:
            self.f.attrs['modified'] = time.ctime()
            self.f.close()

            with gzip.open(self.path, 'wb') as gzFile, \
                      open(self.tmpPath, 'rb') as _bFile:
                gzFile.writelines(_bFile)
        elif self.f is not None:
            self.f.close()

        self.f = None
        if os.path.exists(self.tmpPath):
            os.remove(self.tmpPath)

        self.loaded = False
        self.modified = False

    def clear(self):
        """ ."""

        if not self.loaded:
            self.loggingCritical("Database is not loaded.")
            return

        # Remove all items
        self.modified = True
        if self.f is not None:
            for el in self.f.keys():
                del self.f[el]

        self.close()

        # Delete the file
        if os.path.exists(self.path):
            os.remove(self.path)

        # Re-load the empty DB
        self.load()


    def visit(self, shorter=False, returnString=False):
        """ Prints (or returns as a string) the content of the DB """

        if not returnString and not self.loaded:
            # self.loggingCritical("Database is not loaded.")
            # return 'None'
            self.load()
            if returnString:
                return self.visit(shorter=shorter, returnString=returnString)
            else:
                self.visit(shorter=shorter, returnString=returnString)
            self.close()
            return


        # Reduce the threshold of np printer to be able to
        # have a global view on the DB
        tmpThr = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=10)

        self._s = ''

        # Format DB's size on disk
        size = os.path.getsize(self.tmpPath)
        units = ['B', 'kB', 'MB', 'GB']
        unit = 0
        while size//1000 > 0:
            unit += 1
            size /= 1000
        unit = units[unit]

        if not returnString:
            self.loggingInfo(f"Content of the DB (unzipped size = {size:.2f}{unit}):")
        else:
            self._s += f"Content of the DB (unzipped size = {size:.2f}{unit}):\n"

        def visitFunc(objName, obj):
            skip = False
            slash = objName.rfind('/')
            if slash == -1:
                slash = 0

            rs = ''
            depth = objName.count('/')

            if shorter and depth > 2:
                skip = True
            else:
                if isinstance(obj, h5py.Group):
                    rs = '   '*depth + objName[slash:]
                    if objName.count('/') == 0:
                        rs = '\n' + rs
                elif isinstance(obj, h5py.Dataset):
                    rs = '   '*depth + objName[slash:] + ' : '
                    pad = len(rs)
                    rs += str(obj[()])
                    rs = rs.replace('\n', '\n' + ' '*pad)

            if skip:
                return
            if not returnString:
                self.loggingInfo(rs)
            else:
                self._s += rs + '\n'

        self.f.visititems(visitFunc)

        # Restore np threshold
        np.set_printoptions(threshold=tmpThr)

        if returnString:
            return self._s

    def __repr__(self):
        if not self.loaded:
            return 'Database is not loaded.'
        return self.visit(returnString=True)


    def loadAlgebra(self, name):
        """ Load an algebra, store it in a dictionary and return it.
        If not already present in the database, create a new data-group"""

        cm = CartanMatrix(name)
        fn = cm._fullName
        if fn not in self.algebras:
            self.loggingDebug(f"\nLoading {name} algebra...")
            self.algebras[fn] = LieAlgebra(cm)
            self.algebras[fn].fn = fn
            self.loggingDebug(f"Done.")

        # Basic info such as dim, rank, cartan matrix, ... is systematically
        # stored in the DB. Not that this info is computationally costly, but
        # this way it is made available to external programs possibly reading the DB
        if self.loaded and fn not in self.f:
            self.f.create_group(fn)
            self.writeBasicInfo(self.algebras[fn], overWrite=False)

        return self.algebras[fn]

    class NotLoadedError(BaseException):
        pass

    # Logging system
    def initLogging(self, logLevel):
        logLevel = logLevel.lower()
        if logLevel == 'info':
            log = (True, True, False)
        if logLevel == 'critical':
            log = (False, True, False)
        if logLevel == 'debug':
            log = (True, True, True)

        (self.logging['info'],
         self.logging['critical'],
         self.logging['debug']) = log

    def loggingInfo(self, mess):
        if self.logging['info']:
            print(mess)

    def loggingCritical(self, mess):
        if self.logging['critical'] and not self.raiseErrors:
            print(mess)
        if self.raiseErrors:
            raise BaseException(mess)

    # def loggingDebug(self, mess):
    #     if self.logging['debug']:
    #         print(mess)

    def loggingDebug(self, mess):
        if self.logging['debug']:
            if mess[0] == "\n":
                mess = mess[1:]
                init = "\n[-DB-]"
            else:
                init = "[-DB-]"
            print(init, mess)

    ##################################
    # Write data : general functions #
    ##################################

    def writeObject(self, algebra, objType, objName=None, objVal=None, overWrite=False):
        """ This is the lowest-level function to write an object in the DB.
        If overWrite=False, nothing will happen if the object is already in. """

        fn = algebra.fn
        dbAlg = self.f[fn]
        objType = objType.lower()

        # If asked, overwrite the item by deleting it first
        if overWrite and objType in dbAlg:
            if objName is None:
                # Basic info
                self.loggingDebug(f"  In '{fn}' : Deleting object '{objType}' ...")
                del dbAlg[objType]
            else:
                self.loggingDebug(f"  In '{fn}' : Deleting object ('{objType}', '{objName}') ...")
                if objName in dbAlg[objType]:
                    del dbAlg[objType][objName]


        # Handle basic algebra info
        if objType in self.basicTranslations:
            if objType not in dbAlg:
                val = self.basicTranslations[objType](algebra)
                self.loggingDebug(f"  In '{fn}' : Setting '{objType}' to '{val}' (type={type(val)})")

                newVal = PyLieDB.convert(val, dataGroup=dbAlg, objName=objType)
                if newVal is not None:
                    dbAlg[objType] = newVal
                self.modified = True
            else:
                self.loggingDebug(f"  In '{fn}' : Object '{objType}' is already here, with value '{PyLieDB.parse(dbAlg[objType])}'")

        # Handle other info (repMatrices, invariants, ...)
        elif objType in self.storedTranslations:
            # In this case, objName must not be None
            if objName is None and objVal is None:
                self.loggingCritical(f"Error : object name and value must be provided")
                return

            # If the field is not yet in the DB, first create it
            if objType not in dbAlg:
                dbAlg.create_group(objType)

            # if not objName in dbAlg[objType]:
            self.loggingDebug(f"  In '{fn}' : Writing in ('{objType}', '{objName}') : (type={type(objVal)})")

            newVal = PyLieDB.convert(objVal, objType=objType, dataGroup=dbAlg[objType], objName=objName)

            if newVal is not None:
                dbAlg[objType][objName] = newVal
            self.modified = True

        else:
            self.loggingCritical(f"Error : unkown object type '{objType}'")


    def convert(val, objType=None, allStr=False, dataGroup=None, objName=None):
        """ Static function used to convert the various data-types to
            hd5-compatible types. """

        # Empty list
        if isinstance(val, list) and len(val) == 0:
            if objType == 'invariants':
                return sTensorDB(val).write(dataGroup, objName)
            return []

        # Sparse matrix
        if isinstance(val, sMat):
            return sMatDB(val).write(dataGroup, objName)

        # List of sparse matrices
        if isinstance(val, list) and isinstance(val[0], sMat):
            return sMatDB(val, depth=1).write(dataGroup, objName)

        # List of lists of sparse matrices
        if isinstance(val, list) and isinstance(val[0], list) and isinstance(val[0][0], sMat):
            return sMatDB(val, depth=2).write(dataGroup, objName)

        # List of sparse tensors
        if isinstance(val, list) and isinstance(val[0], sTensor):
            return sTensorDB(val).write(dataGroup, objName)

        # List/tuple of values
        if isinstance(val, list) or isinstance(val, tuple):
            return [PyLieDB.convert(el) for el in val]

        # String
        if isinstance(val, str):
            return val

        # Single integer value
        if allStr:
            s = str(val)
            s = s.replace('sqrt', 's')
            return s.encode('ASCII')
        else:
            if val is True:
                return -1
            if int(val) != val:
                return PyLieDB.convert(val, allStr=True)
            return int(val)


    #################################
    # Read data : general functions #
    #################################

    def abb(self, itemName):
        """ Define some abbreviations """

        loweredItemName = itemName.lower()

        if (loweredItemName in self.basicTranslations
         or loweredItemName in self.storedTranslations
         or loweredItemName in self.translations):
            return loweredItemName

        if loweredItemName == 'dim':
            return 'dimension'
        if loweredItemName == 'adjoint':
            return 'adjointrep'
        if loweredItemName == 'cartan':
            return 'cartanmatrix'

        if loweredItemName in ('struc', 'struct', 'f', 'fabc'):
            return 'structureconstants'
        if loweredItemName in ('repmat', 'repmats'):
            return 'repmatrices'
        if loweredItemName in ('inv', 'invs', 'cgc', 'cgcs'):
            return 'invariants'

        if loweredItemName in ('dimr', 'dimrep', 'repdim'):
            return 'dimr'
        if loweredItemName in ('dynkinlabel', 'labels', 'label'):
            return 'dynkinlabels'
        if loweredItemName in ('repname', 'name'):
            return 'repname'
        if loweredItemName in ('index'):
            return 'dynkinindex'

        return itemName

    def sympify(expr):
        """ Simplified (and faster) version of sympify.
        Only handle the cases expected to show up."""

        f = expr.find('^(')
        if f != -1:
            # Power of rational number
            closing = expr.find(')', f)
            p = expr[f:closing+1]
            expr = expr.replace(p, '**Rational'+p[1:].replace('/',','))
        if 's' in expr:
            return eval(expr.replace('s', 'sqrt'))

        if '/' in expr:
            i = ('I' in expr)
            if i:
                expr = expr.replace('I', '1')
            aux = expr.split('/')
            expr = 'Rational('+','.join(aux)+')'
            if i:
                expr = 'I*'+expr

        return eval(expr)

    def parse(val, objType=None):
        """ This is the inverse function of convert. It reads an object from the DB
        and converts it to a proper python/sympy object"""

        if isinstance(val, h5py.Group):
            unknownType = False
            if 'type' in val.attrs:
                if val.attrs['type'][:4] == 'sMat':
                    return sMatDB(val).read()
                if val.attrs['type'] == 'sTensor':
                    return sTensorDB(val).read()
                else:
                    unknownType = True
            else:
                unknownType = True

            if unknownType:
                print(f"Error : unable to parse the object {val} with type {type(val)}")
                return

        if isinstance(val, h5py.Dataset):
            return PyLieDB.parse(val[()], objType=objType)

        if isinstance(val, list) or (isinstance(val, np.ndarray) and len(val.shape) == 1):
            return [PyLieDB.parse(el, objType=objType) for el in val]
        if isinstance(val, tuple):
            return tuple([PyLieDB.parse(el, objType=objType) for el in val])

        if isinstance(val, str):
            return val

        if not isinstance(val, bytes):
            if objType == 'dynkinlabels' and isinstance(val, np.ndarray):
                return val.tolist()
            elif objType == 'dynkinlabels' and int(val) == val and val == -1:
                return True
            return int(val)
        else:
            if objType is None or objType not in ('name', 'fullname'):
                return PyLieDB.sympify(str(val, 'ASCII'))
            else:
                return str(val, 'ASCII')


    def get(self, *args, **kwargs):
        """ get() is the main and easiest way to retrieve info from the DB.
        If the info is not already present in the DB, it will be computed and
        stored before being returned. Examples of usage :
            - db.get('SU2', 'rank')
            - db.get('SU2', 'cartanMatrix')
            - db.get('SU2', 'invariants', [[1], [1,True]])
            - db.get('SU2', 'repMatrices', [1])

        """

        ret = None
        error = False

        try:
            # If the database if not loaded, try to get the result anyway
            if not self.loaded:
                load = False
                try:
                    ret = self.__getitem__((args, kwargs))
                except self.NotLoadedError:
                    load = True

                if load:
                # Load, get, close.
                    self.load()
                    ret = self.__getitem__((args, kwargs))
                    self.close()
            else:
                ret = self.__getitem__((args, kwargs))
        except TypeError as e:
            error = e
            raise TypeError(e)
            self.loggingCritical('Error : ' + str(e))

        if error is not False:
            raise TypeError(error)

        # Final enhancements to the result
        if type(ret) == tuple:
            ret, objType, algebra = ret

            if objType == 'invariants':
                if 'ordering' in kwargs:
                    for i, inv in enumerate(ret):
                        ret[i] = inv.permute(kwargs['ordering'])
                if 'fields' in kwargs:
                    for i, inv in enumerate(ret):
                        try:
                            inv.setFields(kwargs['fields'])
                            ret[i] = inv.expr()
                        except:
                            pass

            # Remove the True argument for dynkin labels of real reps
            if (objType == 'dynkinlabels' or objType == 'conjugate') and 'realBasis' in kwargs:
                if ret[-1] is True and algebra._goToRealBasis(ret[:-1], kwargs['realBasis']):
                    return type(ret)(ret[:-1])

        return ret

    def __getitem__(self, item):
        args, kwargs = item
        if len(args) == 1:
            arg = args[0]
            if self.loaded and arg.lower() in self.f.attrs:
                return self.f.attrs[arg.lower()]
            if arg not in self.algebras:
                self.loadAlgebra(arg)
            return self.algebras[arg]

        if len(args) >= 2:
            gp, objType = args[:2]
            args = args[2:]


        if not isinstance(gp, str):
            raise TypeError("The algebra must be specified as a string.")
        if not isinstance(objType, str):
            raise TypeError("The object type to read from the DB must be specified as a string.")

        objType = self.abb(objType)

        # For objects stored in the DB, check that it is loaded
        # and raise an error otherwise
        if objType in self.storedTranslations and not self.loaded:
            raise self.NotLoadedError

        algebra = self.loadAlgebra(gp)

        formatedInput = self.handleInput(algebra, objType, args, kwargs)

        if formatedInput is None:
            # Some error occurred during the interpretation of the query
            self.loggingCritical(f"Could not find item '{objType}' in DB.")
            return

        objName, args, kwargs = formatedInput

        if self.isInDB(algebra, objType, objName):
            # Read the object from the DB
            if objType in self.basicTranslations:
                return self.readBasicInfo(algebra, objType)
            if objName is not None:
                return PyLieDB.parse(self.f[algebra.fn][objType][objName], objType=objType), objType, algebra
            return PyLieDB.parse(self.f[algebra.fn][objType])
        else:
            # Compute the object
            obj = self.compute(algebra, objType, *args, **kwargs)

            # Write it in the DB
            if objType in self.storedTranslations and obj is not None:
                self.writeObject(algebra, objType, objName=objName, objVal=obj)

            # Return it
            return obj, objType, algebra


    def compute(self, algebra, objType, *args, **kwargs):
        try:
            if objType in self.basicTranslations:
                return self.basicTranslations[objType](algebra)
            elif objType in self.storedTranslations:
                func = self.storedTranslations[objType]
                if not (hasattr(func, '__self__') and func.__self__ == self):
                    return func(algebra)(*args, **kwargs)
                else:
                    return func(algebra, *args, **kwargs)
            elif objType in self.translations:
                return self.translations[objType](algebra, *args, **kwargs)
        except BaseException as e:
            self.loggingCritical(f"Error while computing '{objType}' : \n\t-> " + str(e))
            return
        # If the query is not in the translation dics, we consider that the user
        # wants to call any other function of the algebra class.
        # -> Try to call it, and if an error is raised return None
        try:
            call = 'algebra.' + objType + '('
            call += ','.join([str(arg) for arg in args])
            if kwargs != {}:
                call += ','
                call += ','.join([str(k)+'='+str(v) for k,v in kwargs.items()])
            call += ')'

            return eval(call)
        except:
            print("Request '" + call + "' failed.")
            return

    def isInDB(self, algebra, objType, objName=None):
        if self.loaded is False:
            return False
        if algebra.fn not in self.f:
            return False
        if objType not in self.f[algebra.fn]:
            return False
        if objName is None:
            return True
        if objName not in self.f[algebra.fn][objType]:
            return False
        return True


    ######################
    # Basic algebra info #
    ######################

    def writeBasicInfo(self, algebra, overWrite=False):
        """ Get and write the basic information associated with the Lie algebra.
        This occurs once in a run, and no info will be written if the algebra is
        already present in the database. """

        for obj in self.basicTranslations:
            self.writeObject(algebra, obj, overWrite=overWrite)


    def readBasicInfo(self, algebra, item):
        """ Read basic information from the DB. If not already stored in, the info
            will be retrieved and stored via self.loadAlgebra """

        return PyLieDB.parse(self.f[algebra.fn][item], objType=item)


    def handleInput(self, algebra, dataType, args, kwargs):
        """ This function handles the input provided by the user. It takes care of validating
        the input (format and content) and raises error whenever needed. If the input is valid,
        it is formatted into a standard string-form to read/store information from/in the DB.
        It may modify the args/kwargs if some input is 'slightly' invalid (e.g. redundancies or
        non-critical inconsistencies) to prevent the same info to be stored twice."""

        if dataType == 'structureconstants':
            if len(args) != 0:
                raise TypeError("'structureConstants' takes no argument")


        if dataType == 'invariants':
            # The name of the stored invariant is :
            #   " (rep1,rep2,...);conjs;pyrateNorm;realBasis "
            # where :
            #   rep[n] = tuple if conj=False else list
            #   conjs = n-digit sequence of 0 (False) or 1 (True)
            #   pyrateNorm = 0 (False) / 1 (True)
            #   realBasis = n-digit sequence of 0 (no rotation) or 1 (rotation to real basis)

            tag = []
            conjs = []

            # 0) Check the argument
            if len(args) == 1:
                arg = args[0]
            else:
                raise TypeError("'invariants' takes exactly one argument")
                return

            if not (isinstance(arg, tuple) or isinstance(arg, list)) or \
               not (isinstance(arg[0], tuple) or isinstance(arg[0], list)):
                   self.loggingCritical("Error in 'invariants' : argument must be a list/tuple of representations")
                   return

            for i, el in enumerate(arg):
                # 1) Reps converted into tuples
                if isinstance(el, tuple):
                    el = list(el)

                if len([r for r in el if r is not True and r is not False]) != algebra._n:
                    self.loggingCritical(f"Error in 'invariants' : {el} is not a valid representation of {algebra.fn}.")
                    return

                # 2) Conjugations are read directly from the rep list
                # e.g. SU2, [[1], [1,True]]
                conj = False
                if el[-1] is False or el[-1] is True:
                    if 'conj' in kwargs and kwargs['conj'] != []:
                        self.loggingCritical(f"Error in 'invariants' : ambiguous use of conjugations inside reps AND in kwargs.")
                        return
                    conj = el[-1]
                    el = el[:-1]
                    arg[i] = el

                tag.append(tuple(el))

                # 3) If useless (e.g. cplx rep), conj is set to False
                fs = algebra.frobeniusSchurIndicator(el)
                if conj is True and fs == 1:
                    conj = False
                if conj is True and fs == 0 and 'realBasis' in kwargs:
                    # If the rep is rotated to the real basis, no need to keep the conjugation
                    if algebra._goToRealBasis(el, kwargs['realBasis']):
                        conj = False

                conjs.append(conj)

            # Conjugations
            if 'conj' not in kwargs:
                kwargs['conj'] = conjs

            if all(kwargs['conj']):
                kwargs['conj'] = [False]*len(kwargs['conj'])

            conjSequence = ''
            for i, cj in enumerate(kwargs['conj']):
                if cj is True:
                    if algebra.frobeniusSchurIndicator(tag[i]) == 1:
                        tag[i] = self.conjugate(algebra, tag[i])
                        arg[i] = self.conjugate(algebra, arg[i])
                        kwargs['conj'][i] = False
                        conjSequence += '0'
                    else:
                        conjSequence += '1'
                else:
                    conjSequence += '0'

            # Pyrate normalization
            if 'pyrateNormalization' not in kwargs:
                # By default, let's apply the PyrateNormalization
                kwargs['pyrateNormalization'] = True
            pyNorm = kwargs['pyrateNormalization']

            # Real basis
            realBasis = None
            if 'realBasis' in kwargs:
                realBasis = kwargs['realBasis']


            RBsequence = ''
            for rep in tag:
                if algebra._goToRealBasis(rep, realBasis):
                    RBsequence += '1'
                else:
                    RBsequence += '0'

              # Determine the ordering and the final permutation to apply
            ordering = sorted(range(len(tag)), key=lambda i: (tag[i], conjSequence[i], RBsequence[i]))
            kwargs['ordering'] = [ordering.index(i) for i in range(len(tag))]

              # Sort the reps
            arg = [tag[i] for i in ordering]
            conjSequence = ''.join([conjSequence[i] for i in ordering])
            RBsequence = ''.join([RBsequence[i] for i in ordering])

            # Reps
            tag = tuple(arg)

            storeName = str(tag).replace(' ', '')
            storeName += ';' + conjSequence
            storeName += ';' + str(int(pyNorm))
            storeName += ';' + RBsequence

            return storeName, (arg,), {k:v for k,v in kwargs.items() if k not in ('fields', 'ordering')}

        if dataType == 'repmatrices':
            # StoreName :
            #   rep;conj;realBasis  [tuple, 0 / 1, 0 / 1 / 2]

            conj = False
            # 0) Check the argument

            if len(args) == 1:
                arg = args[0]
            else:
                self.loggingCritical("Error: 'repMatrices' takes exactly one argument")
                return

            if not (isinstance(arg, tuple) or isinstance(arg, list)):
                self.loggingCritical("Error in 'repMatrices' : argument must be a list/tuple of dynkin labels.")
                return
            if arg[-1] is False or arg[-1] is True:
                conj = arg[-1]
                arg = arg[:-1]

            if len(arg) != algebra._n:
                raise TypeError(f"Error in 'repMatrices' : {arg} is not a valid representation of {algebra.fn}.")
                return

            fs = algebra.frobeniusSchurIndicator(arg)

            if 'conj' in kwargs:
                conj = kwargs['conj']
            else:
                kwargs['conj'] = conj

            if conj and fs == 1:
                conj = False
                self.loggingInfo(f"Warning : representation {el} of {algebra.fn} is complex. " +
                                 "Keyword 'conj' will be skipped. If needed, use the dynkin labels of " +
                                 "the conjguated irrep instead : {algebra.conjugaeIrrep(el).tolist()}.")

            realBasis = 0
            if 'realBasis' in kwargs:
                realBasis = kwargs['realBasis']

                if algebra._goToRealBasis(arg, realBasis):
                    realBasis = 1
                else:
                    realBasis = 0

            tag = tuple(arg)
            storeName = str(tag).replace(' ', '')
            storeName += ';' + str(int(conj))
            storeName += ';' + str(realBasis)

            return storeName, (arg,), kwargs

        if dataType in ('frobenius', 'conjugaterep', 'dimr', 'dynkinindex') :
            # StoreName :
            #   rep

            # 0) Check the argument

            if len(args) == 1:
                arg = args[0]
            else:
                self.loggingCritical(f"Error: '{dataType}' takes exactly one argument")
                return

            if not (isinstance(arg, tuple) or isinstance(arg, list)):
                self.loggingCritical("Error in '{dataType}' : argument must be a list/tuple of dynkin labels.")
                return

            if arg[-1] is False or arg[-1] is True:
                if arg[-1] is True and dataType == 'conjugaterep':
                    kwargs['conj'] = True
                arg = arg[:-1]


            if len(arg) != algebra._n:
                raise TypeError(f"Error in '{dataType}' : {arg} is not a valid representation of {algebra.fn}.")

            storeName = str(tuple(arg)).replace(' ', '')

            return storeName, (arg,), kwargs

        if dataType == 'dynkinlabels':
            if len(args) == 1:
                arg = args[0]
            else:
                self.loggingCritical(f"Error: 'dynkinlabels' takes exactly one argument")
                return

            if isinstance(arg, tuple) or isinstance(arg, list):
                self.loggingCritical(f"Error: 'dynkinlabels' argument must be an integer (possibly negative to indicate conjugation).")
                return

            storeName = str(arg)

            return storeName, (arg,), {k:v for k,v in kwargs.items() if k != 'realBasis'}

        if dataType == 'conjugate':
            return None, args, {k:v for k,v in kwargs.items() if k != 'realBasis'}

        if dataType == 'firstreps':
            if len(args) != 1:
                self.loggingCritical("Error : 'firstReps' takes exactly one argument.")
                return

            return None, args, kwargs

        # else, the info is not stored in the DB
        return None, args, kwargs


    #############################
    # Custom requests to the DB #
    #############################

    def dimR(self, algebra, rep):
        """ Improved version of algebra.dimR, taking into account
        the [rep, bool] syntax"""

        if rep[-1] is True or rep[-1] is False:
            rep = rep[:-1]

        return algebra.dimR(rep)

    def getDynkinLabels(self, algebra, repDim):
        if isinstance(repDim, list) or isinstance(repDim, tuple):
            return repDim

        repsByDim = {}

        for rep in algebra.repsUpToDimN(abs(repDim)):
            dim = algebra.dimR(rep)
            if dim not in repsByDim:
                repsByDim[dim] = []

            fs = algebra.frobeniusSchurIndicator(rep)
            if fs != 1 or algebra.conjugateIrrep(rep).tolist() not in repsByDim[dim]:
                repsByDim[dim].append(rep)

        if abs(repDim) not in repsByDim:
            self.loggingCritical(f"Algebra {algebra.fn} doesn't contain any rep with dimension {repDim}")
            return

        reps = repsByDim[abs(repDim)]
        if len(reps) == 1:
            if repDim > 0:
                return reps[0]
            else:
                conj = algebra.conjugateIrrep(reps[0]).tolist()
                if conj == reps[0] and abs(repDim) != 1:
                    # Real or pseudo-real
                    conj = conj + [True]
                return conj

        return reps

    def conjugate(self, algebra, rep):
        """ Modified version of PyLie's 'conjugateIrrep'. The difference is that
        it handles the notation [rep, bool] for real and pseudo-real reps.
        Another difference is the type of the returned value : list instead of np.ndarray"""

        retType = type(rep)
        rep = list(rep)

        conj = False
        conjInRep = False
        if rep[-1] is True or rep[-1] is False:
            conjInRep = True
            conj = rep[-1]
            rep = rep[:-1]

        fs = algebra.frobeniusSchurIndicator(rep)

        if fs == 1:
            if conjInRep:
                self.loggingCritical(f"Error: the usage of the conjugation of type [rep, bool] is " +
                                     "not allowed for complex representations. Please use the dynkin " +
                                     "labels of the conjugated irrep (algebra.conjugaeIrrep(el).tolist()}) instead.")
                return

            return retType(algebra.conjugateIrrep(rep).tolist())

        if conj == True:
            return retType(rep)
        return retType(rep + [True])

    def repName(self, algebra, rep, latex=False, iLatex=False):
        """ Returns the name of the irrep in string/latex format.
        Note that the representation must be in its dynkin-labels form."""

        if iLatex:
            latex = True

        rep = list(rep)
        dimR = self.dimR(algebra, rep)
        allDynkins = self.getDynkinLabels(algebra, dimR)

        if not isinstance(allDynkins[0], list):
            allDynkins = [allDynkins]

        conjugatedDynkins = [self.conjugate(algebra, el) for el in allDynkins]

        if rep in allDynkins:
            conj = False
            primes = allDynkins.index(rep)
        if rep in conjugatedDynkins:
            conj = True
            primes = conjugatedDynkins.index(rep)

        if dimR == 1:
            if latex:
                return '\\mathbf{1}'
            return '1'

        if not latex:
            return '-'*int(conj) + str(dimR) + '\''*primes


        name = '\\mathbf{' + str(dimR) + '}'

        if primes > 0:
            name += '^{' + '\''*primes + '}'

        if conj:
            name = '\\overline{' + name + '}'

        if iLatex:
            return Symbol(name)

        return name

    def repProduct(self, algebra, reps, iLatex=False):
        ret = algebra.reduceRepProduct(reps)

        if not iLatex:
            return ret

        else:
            tmp = sorted(ret, key=lambda x: -1*self.dimR(algebra, x[0]))
            ret = ''

            for i, el in enumerate(tmp):
                for j in range(el[1]):
                    # ret.append(Symbol(self.repName(algebra, el[0], latex=True)))
                    ret += self.repName(algebra, el[0], latex=True)

                    if i+1 < len(tmp) or j+1 < el[1]:
                        ret += '\oplus'

            # return Add(*ret, evaluate=False)
            return Symbol(ret)


    def firstReps(self, algebra, N, table=True):
        # Identify the reps
        dims = set()
        maxDim = 0
        step = algebra.dimAdj
        depth = 1

        while len(dims) != N+1:
            reps = algebra.repsUpToDimN(maxDim)
            dims = set()
            for r in reps:
                dims.add(algebra.dimR(r))

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

        if not table:
            return reps

        print(f"-- First representations of {algebra.cartan._fullName} --\n")
        for r in reps:
            print(f"{r} : {self.repName(algebra, r)}")


class sMatDB():
    """ This is a class representing a sparseMartrix to write/read from the DB.
    Works for lists of sparse matrices and lists of lists of sparse matrices.
    When writing in the DB, the constructor is called with one argument : the sparse matrix (list).
    When reading from the DB, the constructor is called with one argument : the h5py.Group object"""

    def __init__(self, arg, depth=0):
        self.depth = 0
        self.shape = None
        self.keys = []
        self.values = []

        # Read from DB data
        if isinstance(arg, h5py.Group):
            self.depth = int(arg.attrs['type'][4:])
            self.shape = tuple(PyLieDB.parse(arg['s']))
            self.keys = arg['k']
            self.values = arg['v']
            return

        # Prepare for storing in DB
        if depth == 0:
            self.depth = depth
            self.shape = arg.shape

            # Read the sparse matrix
            for k,v in arg._smat.items():
                self.keys.append(tuple([int(el) for el in k]))
                self.values.append(PyLieDB.convert(v, allStr=True))

        if depth == 1:
            self.depth = depth
            self.shape = (len(arg),) + arg[0].shape

            for i, mat in enumerate(arg):
                for k,v in mat._smat.items():
                    self.keys.append((i,) + tuple([int(el) for el in k]))
                    self.values.append(PyLieDB.convert(v, allStr=True))

        if depth == 2:
            self.depth = depth
            self.shape = (len(arg), len(arg[0])) + arg[0].shape

            for i, l in enumerate(arg):
                for j, mat in enumerate(arg):
                    for k,v in mat._smat.items():
                        self.keys.append((i, j) + tuple([int(el) for el in k]))
                        self.values.append(PyLieDB.convert(v, allStr=True))



    def write(self, dataGroup, objName):
        """ Write the sparse matrix in the dataGroup """

        if objName is not None:
            dataGroup.create_group(objName)
            dataGroup = dataGroup[objName]

        dataGroup.attrs['type'] = 'sMat'+str(self.depth)
        dataGroup['s'] = self.shape
        dataGroup['k'] = self.keys
        dataGroup['v'] = self.values


    def read(self):
        """ Returns the matrix in the form of a proper sMat """

        if self.depth == 0:
            ret = sMat(*self.shape)
            for k,v in zip(self.keys, self.values):
                key = tuple([PyLieDB.parse(el) for el in k])
                ret[key] = PyLieDB.parse(v)

        elif self.depth == 1:
            ret = [sMat(*self.shape[1:]) for _ in range(self.shape[0])]
            for k,v in zip(self.keys, self.values):
                key = tuple([PyLieDB.parse(el) for el in k])
                ret[key[0]][key[1:]] = PyLieDB.parse(v)

        elif self.depth == 2:
            ret = [[sMat(*self.shape[2:]) for _ in range(self.shape[1])] for _ in range(self.shape[0])]
            for k,v in zip(self.keys, self.values):
                key = tuple([PyLieDB.parse(el) for el in k])
                ret[key[0]][key[1]][key[2:]] = PyLieDB.parse(v)

        return ret

class sTensorDB():
    """ This is a class representing a sparse tensor to write/read from the DB.
    When writing in the DB, the constructor is called with one argument : the sparse tensor list.
    When reading from the DB, the constructor is called with one argument : the h5py.Group object"""

    def __init__(self, arg):
        self.shape = None
        self.keys = []
        self.values = []

        if isinstance(arg, list):
            if arg == []:
                self.shape = ()
            else:
                self.shape = tuple([len(arg)] + [int(el) for el in arg[0].dim if el is not None])

            for i, el in enumerate(arg):
                for k,v in el.dic.items():
                    key = tuple([int(i)] + [int(j) for j in k if j is not None])
                    self.keys.append(key)
                    self.values.append(PyLieDB.convert(v, allStr=True))

        if isinstance(arg, h5py.Group):
            self.shape = tuple(PyLieDB.parse(arg['s']))

            if len(self.shape) > 0:
                self.keys = arg['k']
                self.values = arg['v']

    def write(self, dataGroup, objName):
        """ Write the sparse tensors in the dataGroup """

        dataGroup.create_group(objName)
        dataGroup[objName].attrs['type'] = 'sTensor'
        dataGroup[objName]['s'] = self.shape

        if self.shape == ():
            return

        dataGroup[objName]['k'] = self.keys
        dataGroup[objName]['v'] = self.values

    def read(self):
        """ Returns the matrix in the form of a proper list of sTensors """
        if self.shape == ():
            return []

        ret = [sTensor(*self.shape[1:]) for _ in range(self.shape[0])]
        for k,v in zip(self.keys, self.values):
            i = PyLieDB.parse(k[0])
            key = tuple([PyLieDB.parse(el) for el in k[1:]])
            ret[i][key] = PyLieDB.parse(v)

        return ret