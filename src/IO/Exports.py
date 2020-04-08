# -*- coding: utf-8 -*-

from Logging import loggingInfo, loggingCritical
import os
import sys
import time
from subprocess import DEVNULL, STDOUT, run, CalledProcessError

def exports(runSettings, model):
    loggingInfo("Exporting results...")

    tmpWD = os.getcwd()
    # Create a folder with the name of the model
    if runSettings['CreateFolder'] is True:
        path = os.path.join(runSettings['Results'], model._Name)
        if not (os.path.exists(path)):
            os.makedirs(path)
    else:
        path = runSettings['Results']

    if runSettings['LatexOutput'] is True:
        from Latex import LatexExport

        loggingInfo("\tExporting to Latex...", end=' ')
        latex = LatexExport(model)
        latex.write(os.path.join(path, model._Name + '.tex'))
        loggingInfo("Done.")

    if runSettings['MathematicaOutput'] is True:
        from Mathematica import MathematicaExport

        loggingInfo("\tExporting to Mathematica...", end=' ')
        mathematica = MathematicaExport(model)
        mathematica.write(os.path.join(path, model._Name + '.m'))
        loggingInfo("Done.")

    if runSettings['PythonOutput'] is True:
        from Python import PythonExport

        # If Latex export is disabled, create a Latex object anyway
        # to get the latex substitutions
        if runSettings['LatexOutput'] is False:
            from Latex import LatexExport
            latex = LatexExport(model, getLatexSubs=True)

        loggingInfo("\tExporting to Python...", end=' ')
        try:
            python = PythonExport(model, latexSubs=latex.latex)
            python.write(path)
        except TypeError as e:
            print('\n' + str(e))
        else:
            loggingInfo("Done.")

    if runSettings['UFOfolder'] is not None:
        from UFO import UFOExport

        loggingInfo("\tExporting to UFO...", end=' ')
        try:
            ufo = UFOExport(model)
            ufo.write(runSettings['UFOfolder'])
        except TypeError as e:
            print('\n' + str(e))
        else:
            loggingInfo("Done.")


    # Copy the .model file in the results folder
    if runSettings['CopyModelFile']:
        fName = os.path.join(path, os.path.basename(runSettings['Model']))
        s = "# This model file was automatically copied by PyR@TE 3 on " + time.ctime() + "\n"
        s += runSettings['StoreModelFile']
        try:
            f = open(fName, 'w')
            f.write(s)
            f.close()
        except:
            loggingCritical("Error while copying the model file in the results folder.")

    # Now apply possible user-defined commands from 'default.settings'
    commands = [cmd.strip() for cmd in runSettings['EndCommands'].replace('[name]', model._Name).split(',')]
    loggingInfo("Running user-defined commands : ")
    os.chdir(path)
    shell = (sys.platform.startswith('win'))
    for cmd in commands:
        loggingInfo("\t-> '" + cmd + "'")
        try:
            run(cmd.split(' '), shell=shell, stdout=DEVNULL, stderr=STDOUT, check=True)
        except CalledProcessError as e:
            loggingCritical("An error occurred when running the command. Skipping.")
            loggingCritical(' >> ' + str(e))

    os.chdir(tmpWD)

    # This is for debugging, remove later
    model.latex = latex
    # model.python = python