# -*- coding: utf-8 -*-
try:
    from sys import exit
    import argparse
    import os
    import time
    import re as reg
except ImportError as e:
    exit("Error while importing one of the modules `sys, os, argparse, Logging, time` : \n\t-> " + str(e))

try:
    import yaml
except ImportError:
    exit("Error while importing PyYAML")

try:
    from Logging import logging, pyLogger, loggingInfo, loggingCritical
except ImportError:
    exit("Error while importing 'Logging' module")



class Inputs():
    def __init__(self, wd):
        self.wd = wd

        # Load the default settings from 'default.settings' file
        settings = self.loadDefaultSettings()

        # Load the settings from user command
        runSettings = self.initParser(settings)

        # Combine the two dicts
        for k,v in runSettings.items():
            settings[k] = v

        if os.path.abspath(settings['Results']) != settings['Results']:
            settings['Results'] = os.path.abspath(os.path.join(wd, settings['Results']))
        if not (os.path.exists(settings['Results'])):
            os.makedirs(settings['Results'])

        if settings['ForceLog']:
            settings['LogFolder'] = 'log/'

        if os.path.abspath(settings['LogFolder']) != settings['LogFolder']:
            settings['LogFolder'] = os.path.abspath(os.path.join(wd, settings['LogFolder']))
        if not (os.path.exists(settings['LogFolder'])):
            os.makedirs(settings['LogFolder'])

        self.initLogging(settings)

        # A useful check before going on : no export was selected
        if not any((settings['LatexOutput'], settings['MathematicaOutput'], settings['PythonOutput'], settings['UFOfolder'])):
            loggingCritical("Error : No ouput would be produced after the computation. Please choose at least one export option (Latex, Mathematica, Python, UFO).")
            exit()

        if settings['UFOfolder'] is not None:
            if os.path.abspath(settings['UFOfolder']) != settings['UFOfolder']:
                settings['UFOfolder'] = os.path.abspath(os.path.join(wd, settings['UFOfolder']))
            if not (os.path.exists(settings['UFOfolder'])):
                os.makedirs(settings['UFOfolder'])

        self.yamlSettings = self.readModelFile(settings)
        self.settings = settings


    def getSettings(self):
        return self.settings, self.yamlSettings

    def loadDefaultSettings(self):
        fPath = os.path.join(self.wd, 'default.settings')
        if os.path.exists(fPath):
            try:
                f = open(fPath, 'r')
                if yaml.__version__ > '5.1':
                    yamlSettings = yaml.load(f.read(), Loader=yaml.FullLoader)
                    defaultSettings = yaml.load(self.generateDefaultSettings(), Loader=yaml.FullLoader)
                else:
                    yamlSettings = yaml.load(f.read())
                    defaultSettings = yaml.load(self.generateDefaultSettings())
                f.close()
            except yaml.scanner.ScannerError as err:
                exit(f"Check the default settings file, failed to load the settings:\n\n-> {err}")
            except yaml.parser.ParserError as err:
                exit(f"Check the default settings file, failed to parse the settings:\n\n-> {err}")

            # If some of the default settings are missing from the file, add them to the dic now
            for k,v in defaultSettings.items():
                if k not in yamlSettings:
                    yamlSettings[k] = v
            return yamlSettings
        else:
            try:
                file = open(fPath, 'w')
            except:
                exit('ERROR while creating the default settings file.')

            file.write(self.generateDefaultSettings())
            file.close()

            return self.loadDefaultSettings()

    def initParser(self, default):
        parser = argparse.ArgumentParser()

        # Model
        parser.add_argument('--Model', '-m', dest='Model', action='store', default='',
                            help="This option is used to specify the YAML file containing the model settings")

        # Verbose and log
        parser.add_argument('--VerboseLevel', '-vL', dest='VerboseLevel', action='store', default=default['VerboseLevel'],
                            help='Set up the level of printing')
        parser.add_argument('--Quiet', '-q', dest='Quiet', action='store_true', default=False,
                            help='Switch off most of the printing system')

        parser.add_argument('--ForceLog', '-log', dest='ForceLog', action='store_true', default=False,
                            help='Produce log files even if \'DisableLogFiles\' is set to True in the default settings')

        # Loop-level of the calculation
        parser.add_argument('--Loops', '-l', dest='Loops', action='store', default=default['DefaultLoopLevel'],
                            help='Set the calculation loop order')

        # Gauge invariance check
        parser.add_argument('--CheckGaugeInvariance', '-gi', dest='CheckGaugeInvariance', action='store_true', default=None,
                            help='Perform a gauge invariance check prior to the RGE computation')
        parser.add_argument('--no-CheckGaugeInvariance', '-no-gi', dest='CheckGaugeInvariance', action='store_false', default=None,
                            help='Perform a gauge invariance check prior to the RGE computation')
        parser.set_defaults(CheckGaugeInvariance=default['CheckGaugeInvariance'])

        # Result folder
        parser.add_argument('--Results', '-res', dest='Results', action='store', default=default['ResultsFolder'],
                            help='Store all the output files in the path')

        # Output files

            # Latex
        parser.add_argument('--LatexOutput', '-tex', dest='LatexOutput', action='store_true', default=None,
                            help='Produce Latex output')
        parser.add_argument('--no-LatexOutput', '-no-tex', dest='LatexOutput', action='store_false', default=None,
                            help='Switch off Latex output')
        parser.set_defaults(LatexOutput=default['LatexOutput'])

            # Mathematica
        parser.add_argument('--MathematicaOutput', '-math', dest='MathematicaOutput', action='store_true', default=None,
                            help='Produce Mathematica output')
        parser.add_argument('--no-MathematicaOutput', '-no-math', dest='MathematicaOutput', action='store_false', default=None,
                            help='Switch off Mathematica output')
        parser.set_defaults(MathematicaOutput=default['MathematicaOutput'])

            # Python
        parser.add_argument('--PythonOutput', '-py', dest='PythonOutput', action='store_true', default=None,
                            help='Produce Python output')
        parser.add_argument('--no-PythonOutput', '-no-py', dest='PythonOutput', action='store_false', default=None,
                            help='Switch off Python output')
        parser.set_defaults(PythonOutput=default['PythonOutput'])

            # UFO export
        parser.add_argument('--UFOfolder', '-ufo', dest='UFOfolder', action='store', default=None,
                            help='Ask PyR@TE to produce a UFO \'running.py\' file in the specified folder')

        # Some other options
        parser.add_argument('--no-KinMix', '-no-kin', dest='NoKinMix', action='store_true', default=False,
                            help='Switch off the kinetic mixing terms if multiple U(1) gauge groups are present.')

        return parser.parse_args().__dict__


    def initLogging(self, settings):
        # Create the config of the logging system
        if not settings['DisableLogFiles'] or settings['ForceLog']:
            lt = time.localtime()
            dateTime = f"{lt.tm_mday:02d}_{lt.tm_mon:02d}_{int(str(lt.tm_year)[-2:]):02d}-{lt.tm_hour:02d}_{lt.tm_min:02d}_{lt.tm_sec:02d}"

            logFile = os.path.join(settings['LogFolder'], 'PyLog' + dateTime + '.log')

            for handler in logging.root.handlers:
                logging.root.removeHandler(handler)

            logging.basicConfig(filename=logFile, level='DEBUG', format="%(message)s")

        if settings['Quiet'] is True:
            settings['VerboseLevel'] = 'Critical'

        # Setting up the verbose level
        if settings['VerboseLevel'] == 'Info':
            settings['vInfo'], settings['vDebug'], settings['vCritical'] = True, False, True
        elif settings['VerboseLevel'] == 'Debug':
            settings['vInfo'], settings['vDebug'], settings['vCritical'] = False, True, True
        elif settings['VerboseLevel'] == 'Critical':
            settings['vInfo'], settings['vDebug'], settings['vCritical'] = False, False, True
        elif settings['VerboseLevel'] == '':
            settings['vInfo'], settings['vDebug'], settings['vCritical'] = True, False, True
        else:
            print(f"Unknown VerboseLevel {settings['VerboseLevel']}... Setting it to 'Info'.")
            settings['vInfo'], settings['vDebug'], settings['vCritical'] = True, False, True

        pyLogger.init(logging.getLogger())
        pyLogger.initVerbose(settings)


    def readModelFile(self, RunSettings):
        if RunSettings['Model'] == '':
            loggingCritical("Error : Please, specify a .model file (using '-m' argument).")
            exit()
        else:
            if os.path.abspath(RunSettings['Model']) != RunSettings['Model']:
                RunSettings['Model'] = os.path.abspath(os.path.join(self.wd, RunSettings['Model']))
            try:
                # Open the Yaml file and load the settings
                f = open(RunSettings['Model'], 'r')
                RunSettings['StoreModelFile'] = f.read()
                f.close()

                fString = self.parseFile(RunSettings['StoreModelFile'])
                if yaml.__version__ > '5.1':
                    yamlSettings = yaml.load(fString, Loader=yaml.FullLoader)
                else:
                    yamlSettings = yaml.load(fString)
            except yaml.scanner.ScannerError as err:
                loggingCritical(f"Check the YAML file {RunSettings['Model']}, impossible to load the settings:\n\n-> {err}.")
                exit()
            except yaml.parser.ParserError as err:
                loggingCritical(f"Check the YAML file {RunSettings['Model']}, impossible to parse the settings:\n\n->{err}.")
                exit()
            except IOError as errstr:
                loggingCritical(f"Did not find the YAML file {RunSettings['Model']}, specify the path if not in the current directory.\n\n-> {errstr}.")
                exit()

            loggingInfo(f"Loading the YAML file: {RunSettings['Model']} ...", end=' ')

            # Now we want to process the settings before creating the model class
            # Let's first construct the dictionaries if the input is given as a list

            if 'Fermions' in yamlSettings and yamlSettings['Fermions'] != {}:
                for k,v in yamlSettings['Fermions'].items():
                    if type(v) == dict:
                        continue
                    elif type(v) == list:
                        if len(v) == len(yamlSettings['Groups']) + 1:
                            qnb = {grp:Q for (grp, Q) in zip(yamlSettings['Groups'], v[1:])}
                            yamlSettings['Fermions'][k] = {'Gen': v[0], 'Qnb': qnb}
                        else:
                            loggingCritical(f"Error : The length of the lists describing fermions should be 1 + {len(yamlSettings['Groups'])}, "
                                          + f"corresponding to generation + various quantum numbers. ('{k} : {v}')")
                            exit()
                    else:
                        loggingCritical(f"Error : Fermions should either be described by a dictionary or a list. ('{k} : {v}')")
                        exit()

            if 'RealScalars' in yamlSettings and yamlSettings['RealScalars'] != {}:
                for k,v in yamlSettings['RealScalars'].items():
                    if type(v) == dict:
                        if len(v) == 1 and 'Qnb' in v:
                            yamlSettings['RealScalars'][k] = v['Qnb']
                    elif type(v) == list:
                        if len(v) == len(yamlSettings['Groups']):
                            qnb = {grp:Q for (grp, Q) in zip(yamlSettings['Groups'], v)}
                            yamlSettings['RealScalars'][k] = qnb
                        else:
                            loggingCritical(f"Error : The length of the lists describing real scalars should be {len(yamlSettings['Groups'])}, "
                                          + f"corresponding to the various quantum numbers. ('{k} : {v}')")
                            exit()
                    else:
                        loggingCritical(f"Error : Real scalars should either be described by a dictionary or a list. ('{k} : {v}')")
                        exit()

            # For complex scalars, also check that the pairs [Pi, Sigma] are only used once
            if 'CplxScalars' in yamlSettings and yamlSettings['CplxScalars'] != {}:
                realFieldsDic = {}
                for k,v in yamlSettings['CplxScalars'].items():
                    if type(v) == dict:
                        pass
                    elif type(v) == list:
                        if len(v) == len(yamlSettings['Groups']) + 3:
                            qnb = {grp:Q for (grp, Q) in zip(yamlSettings['Groups'], v[3:])}
                            yamlSettings['CplxScalars'][k] = {'RealFields': [v[0], v[1]], 'Norm': v[2], 'Qnb': qnb}
                        else:
                            loggingCritical(f"Error : The length of the lists describing complex scalars should be 3 + {len(yamlSettings['Groups'])}, "
                                          + f"corresponding to Re + Im + norm + various quantum numbers. ('{k} : {v}')")
                            exit()
                    else:
                        loggingCritical(f"Error : Complex scalars should either be described by a dictionary or a list. ('{k} : {v}')")
                        exit()

                    rf = tuple(yamlSettings['CplxScalars'][k]['RealFields'])
                    if rf not in realFieldsDic:
                        realFieldsDic[rf] = k
                    else:
                        loggingCritical(f"Error in complex scalar '{k}' : the real fields couple {rf} is already used in '{realFieldsDic[rf]}'")
                        exit()

            if 'Potential' in yamlSettings and yamlSettings['Potential'] != {}:
                labels = ('QuarticTerms', 'Yukawas', 'TrilinearTerms', 'ScalarMasses', 'FermionMasses')
                for cType in labels:
                    if cType in yamlSettings['Potential'] and yamlSettings['Potential'][cType] != {}:
                        for coupling, term in yamlSettings['Potential'][cType].items():
                            if type(term) == str:
                                # This is an explicit math expression
                                pass
                            elif type(term) == dict:
                                # This is a dict with no values, containing :
                                #    { 'mathExpression', assumption1, assumption2, ... }
                                tup = list(term.keys())
                                if len(tup) > 1:
                                    tup = tuple([tup[0]] + [el.lower() for el in tup[1:]])
                                else:
                                    tup = tup[0]
                                yamlSettings['Potential'][cType][coupling] = tup
                            else:
                                loggingCritical(f"Could not understand the term : {term}")
                                exit()

        loggingInfo("Done.")
        return yamlSettings


    def parseFile(self, s):
        """ Automatically add quotes where needed in the model file, so the user doesn't have to do it """

        preambleKW = ('Author', 'Date', 'Name', 'Groups',
                      'Fermions', 'RealScalars', 'CplxScalars',
                      'Potential')

        kwList = ('Definitions', 'Yukawas', 'QuarticTerms',
                  'FermionMasses', 'TrilinearTerms', 'ScalarMasses', 'Vevs',
                  'Substitutions', 'Latex', 'UFOMapping')
        kwNoDic = ('BetaFactor', 'FermionAnomalous', 'ScalarAnomalous')

        allKW = preambleKW + kwList + kwNoDic

        def findClosingBrackets(s, kw):
            originalS = s
            findKw = s.find(kw)

            if findKw == -1:
                return None

            s = s[findKw:]
            opening = s.find('{')

            if any([el in originalS[findKw:findKw+opening] for el in allKW if el!=kw]):
                return None

            cursor = opening
            depth = 0

            while cursor < len(s):
                cursor += 1
                if s[cursor] == '{':
                    depth +=1
                elif s[cursor] == '}':
                    if depth > 0:
                        depth -=1
                    else:
                        return (findKw+opening, findKw+cursor+1)

        def insertQuotes(s, kw, dic=True):
            # print("Insert quotes : ", kw)
            cb = findClosingBrackets(s, kw)
            if cb is None:
                pos = s.find(kw)
                nextLine = s.find('\n', pos)

                line = s[pos:nextLine].strip()

                if line == '':
                    return s

                newLine = line.replace('all', 'All').replace('All', '{All}')
                newLine = newLine.replace('none', 'None').replace('None', '{}')
                if newLine[-1] == ',':
                    newLine = newLine[:-1]

                # if kw == 'BetaFactor':
                #     newLine = newLine.replace('(', '[').replace(')', ']')

                return s.replace(line, newLine)

            originalStr = s[cb[0]:cb[1]]
            subStr = originalStr.strip()[1:-1]

            matches = reg.findall(r'(?:[,:\s]*([^,:\n]+]*))', subStr)

            # print(matches)
            newMatches = []

            i = 0
            while i < len(matches):
                el = matches[i]
                brackets = (el.count('['), el.count('('), el.count('{'))
                if brackets == (0,0,0):
                    newMatches.append(el)
                else:
                    nO = brackets
                    nC = (el.count(']'), el.count(')'), el.count('}'))

                    newMatches.append(el)
                    nextEl = matches[i]
                    while nC != nO:
                        i += 1
                        if i == len(matches):
                            i -= 1
                            break
                        nextEl = matches[i]
                        if dic:
                            newMatches[-1] += ',' + nextEl
                        else:
                            if nO[0] != nC[0]:
                                newMatches[-1] += ',' + nextEl
                            else:
                                newMatches[-1] += ';' + nextEl

                        nO = (newMatches[-1].count('['), newMatches[-1].count('('), newMatches[-1].count('{'))
                        nC = (newMatches[-1].count(']'), newMatches[-1].count(')'), newMatches[-1].count('}'))
                i += 1

            newStr = ' {\n'
            i = 0
            while i < len(newMatches):
                el = newMatches[i].strip()
                if el == '':
                    i+=1
                    continue

                if el[0] == "'" and el[-1] == "'":
                    el = el[1:-1]


                if (el[0] == '[' and el[-1] == ']') or el.find("'") != -1:
                    el = '"' + el + '"'
                else:
                    el = "'" + el + "'"

                if dic and i%2 == 0:
                    newStr += el
                    newStr += ' : '
                else:
                    el = el.replace(' ', '')
                    if el[1] == '{' and el[-2] == '}':
                        el = el[2:-2].split(',')

                        j = 0
                        newEl = []
                        sub = el[j]
                        nO = sub.count('[')
                        nC = sub.count(']')
                        newEl.append(sub)
                        nextEl = sub
                        while nC != nO:
                            j += 1
                            if j == len(el):
                                j -= 1
                                break
                            nextEl = el[j]
                            newEl[-1] += ',' + nextEl
                            nO = newEl[-1].count('[')
                            nC = newEl[-1].count(']')

                        newEl = '{\'' + newEl[0] + '\''
                        if len(el[j+1:]) >= 1:
                            newEl += ', '
                            newEl += ','.join(el[j+1:])
                        el = newEl + '}'

                    newStr += el
                    newStr += ',\n'
                i += 1

            newStr += '}'

            return s.replace(originalStr, newStr)

        # Remove comments:
        comment = s.find('#')
        while comment != -1:
            newLine = s.find('\n', comment)
            s = s.replace(s[comment:newLine], '', 1)
            comment = s.find('#')

        for kw in kwList:
            s = insertQuotes(s, kw)

        for kw in kwNoDic:
            s = insertQuotes(s, kw, dic=False)

        return s



    def generateDefaultSettings(self):
        s = """\
#########################################################################
#           This is the default settings file for PyR@TE 3.             #
# Feel free to adjust these settings depending on your usage of PyR@TE. #
#########################################################################
#     Note : if missing, this file will be automatically generated      #
#                 by PyR@TE in its original form                        #
#########################################################################


# Printing

VerboseLevel : Info

# Default folders

ResultsFolder : results/
LogFolder : log/
DisableLogFiles : False

# Computation

DefaultLoopLevel : 1
CheckGaugeInvariance : True
PrintComputationTimes : True

RealBasis : adjoint

# Output

CreateFolder : True
CopyModelFile : False

LatexOutput : True
AbsReplacements : True   #Automatic replacements x*conjugate(x) -> \\abs{x}^2
GroupTheoryInfo : True

MathematicaOutput : True
MathematicaSolver : True

PythonOutput : True

# User-defined commands to run after the computation

EndCommands : "pdflatex [name].tex, pdflatex [name].tex"
"""
        return s



