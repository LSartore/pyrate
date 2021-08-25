# -*- coding: utf-8 -*-

from sys import exit
import importlib


requirements = {
    'python': ('python', '3.6.0', True),
    'yaml': ('PyYAML', '5.3', True),
    'sympy': ('sympy', '1.5', True),
    'h5py': ('h5py', '2.10', True),
    'numpy': ('numpy', '1.18', True),
    'scipy': ('scipy', '1.4', False),
    'matplotlib': ('matplotlib', '3.1', False)
    }



def vTuple(version):
    if not isinstance(version, tuple):
        return tuple([int(el) for el in version.split('.')])
    return version

def checkDependenciesAux(requirements):
    missing = []

    for module, (moduleName, targetVersion, mandatory) in requirements.items():
        if module == 'python':
            import sys
            v = '.'.join([str(el) for el in tuple(sys.version_info)[:3]])
        else:
            try:
                v = importlib.import_module(module).__version__
            except:
                v = (0,)

        if vTuple(v) < vTuple(targetVersion):
            missing.append((moduleName, v, targetVersion, mandatory))

    if missing == []:
        return (True, [], [])

    errorStr = []
    warningStr = []
    for (moduleName, v, targetVersion, mandatory) in missing:
        s = '\t-' + moduleName + ' >= ' + targetVersion + '   (current: ' + (v if v != (0,) else 'None') + ')'
        if mandatory:
            errorStr.append(s)
        else:
            warningStr.append(s)

    return (False, errorStr, warningStr)


def checkDependencies():
    from Logging import loggingCritical, loggingInfo

    dep = checkDependenciesAux(requirements)

    if dep[0] is False:
        ex = False
        if dep[1] != []:
            mess = 'Error: some dependencies are missing/outdated.\n' + '\n'.join(dep[1])
            loggingCritical(mess)
            ex = True

        if dep[2] != []:
            mess = 'Warning: some optional dependencies are missing/outdated.\n' + '\n'.join(dep[2])
            loggingInfo(mess)

        if ex:
            exit()



if __name__ == "__main__":
    dep = checkDependenciesAux(requirements)

    if dep[0] is True or dep[1] == []:
        exit(0)

    import os

    mess = 'Error: some PyR@TE 3 dependencies are missing/outdated.\n' + '\n'.join(dep[1])
    logFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'log')

    if not os.path.exists(logFolder):
        os.mkdir(logFolder)

    file = os.path.join(logFolder, 'dependencies.log')

    f = open(file, 'w')
    f.write(mess)
    f.close()

    exit(1)
