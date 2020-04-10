import sys
import logging
import time

#################
## Logging system
#################

class pyLogger():
    """ PyR@TE logger class -- Possesses only static methods """

    noNewLine = False
    gVB = {'vInfo': False, 'vDebug': False, 'vCritical': False}
    logger = None

    progressBar = False
    lastMessage = ''

    pt = 0 # This is a timer for the progress bar

    def init(logger):
        pyLogger.noNewLine = False
        pyLogger.gVB = {'vInfo': False, 'vDebug': False, 'vCritical': False}

        logging.root.handlers[0].terminator = ''
        pyLogger.logger = logging.LoggerAdapter(logger, None)
        pyLogger.logger.process = pyLogger.process

    def process(mess, kwargs):
        end = '\n'
        prefix = '[Log] '

        if pyLogger.noNewLine:
            end = ''
        if 'switch' in kwargs and kwargs['switch'] is True:
            prefix = ''

        if 'noNewLine' in kwargs:
            if kwargs['noNewLine']:
                end = ''
            prefix = ''

        pyLogger.lastMessage = (prefix + mess + end, len(prefix))
        return pyLogger.lastMessage[0], {}

    def properPrint(mess, **kwargs):
        if pyLogger.noNewLine and 'Done' not in mess and 'OK' not in mess:
            mess = '\n' + mess

        print(mess, end=(kwargs['end'] if 'end' in kwargs else '\n'))

        if 'end' in kwargs and '\n' not in kwargs['end']:
            pyLogger.noNewLine = True
        elif pyLogger.noNewLine :
            pyLogger.noNewLine = False
            return True

    def info(mess, **kwargs):
        switch = None
        if pyLogger.gVB['vInfo'] and not('noPrint' in kwargs and kwargs['noPrint']):
            switch = pyLogger.properPrint(mess, **kwargs)

        if switch is True:
            kwargs['switch'] = True
        pyLogger.logger.info(mess, **kwargs)

    def debug(mess, **kwargs):
        if pyLogger.gVB['vDebug'] and not('noPrint' in kwargs and kwargs['noPrint']):
            pyLogger.properPrint(mess, **kwargs)

        pyLogger.logger.debug(mess)

    def critical( mess, **kwargs):
        if pyLogger.gVB['vCritical'] and not('noPrint' in kwargs and kwargs['noPrint']):
            pyLogger.properPrint(mess, **kwargs)

        pyLogger.logger.critical(mess)

    def setVerbose(tup):
        pyLogger.gVB = {k:tup[i] for i, k in enumerate(('vInfo', 'vDebug', 'vCritical'))}

    def initVerbose(RunSettings):
        pyLogger.gVB = {k:RunSettings[k] for k in ('vInfo', 'vDebug', 'vCritical')}


def loggingInfo(mess, **kwargs):
    pyLogger.info(mess, **kwargs)

def loggingCritical(mess, **kwargs):
    pyLogger.info(mess, **kwargs)

def loggingDebug(mess, **kwargs):
    pyLogger.info(mess, **kwargs)



# Print iteration progress
# Adapted from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50, verbose='Info', printTime=False, logProgress=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    # Depending on the verbose level, do or do not print progress
    if pyLogger.gVB['v' + verbose] == False:
        return

    if pyLogger.pt == 0:
        pyLogger.pt = time.time()
    if iteration == total:
        if printTime:
            suffix = f" ({(time.time()-pyLogger.pt):.3f} seconds)"

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + ' ' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if logProgress:
        previous_length = int(round(bar_length * (iteration-1) / float(total)))
        if pyLogger.progressBar is False:
            pyLogger.progressBar = True
            m, l = pyLogger.lastMessage
            padString = l*' ' + m[l:].replace(m[l:].lstrip(), '')
            loggingInfo(padString + '|'+'#'*filled_length, noNewLine=True, noPrint=True)
        elif iteration < total:
            loggingInfo('#'*(filled_length - previous_length), noNewLine=True, noPrint=True)
        else:
            loggingInfo('#'*(filled_length - previous_length)+'|'+f" ({(time.time()-pyLogger.pt):.3f} seconds)", noNewLine=False, noPrint=True)
            pyLogger.progressBar = False

    if iteration == total:
        sys.stdout.write('\n')
        pyLogger.pt = 0
