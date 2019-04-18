import logging
import sys

from termcolor import colored


class _Formatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibilty
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_Formatter, self).format(record)
def _get_logger(log_level=logging.DEBUG):
    logger = logging.getLogger('pytorch-voice-conversion')
    logger.propagate = False
    logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Formatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger