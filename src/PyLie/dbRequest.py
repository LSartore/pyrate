""" This file is designed to be run from the shell. It sends a request to the
PyLie DB in order to compute some missing info and update it. This procedure is needed to update
the DB from an external program (such as Mathematica in the context of the FeynRules interface)"""

import sys

pyLiePath = sys.argv[1:]
sys.path.append(pyLiePath)

from PyLieDB import PyLieDB

args = sys.argv[2:]

# Collect lists and tuples together
i = 0
while i < len(args):
    o = args[i].count('[') + args[i].count('(') + args[i].count('{')
    c = args[i].count(']') + args[i].count(')') + args[i].count('}')
    if o > c:
        args[i] = args[i] + args[i+1]
        args.pop(i+1)
    else:
        i += 1

if len(args) < 2:
    sys.exit(1)
if len(args) == 2:
    gp, request = args
    arg, kwargs = None, {}
if len(args) == 3:
    gp, request, arg = args
    kwargs = {}
if len(args) == 4:
    gp, request, arg, kwargs = args

try:
    if arg is not None:
        arg = [eval(arg)]
    else:
        arg = []
    if kwargs != {}:
        kwargs = eval(kwargs)
except:
    sys.exit(2)

try:
    db = PyLieDB()
    db.load(force=True)
except:
    sys.exit(3)

try:
    if db.get(gp, request, *arg, **kwargs) is None:
        sys.exit(4)
except:
    sys.exit(4)

db.close()