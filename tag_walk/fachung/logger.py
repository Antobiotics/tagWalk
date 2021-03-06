import logging
import coloredlogs


coloredlogs.install(level='INFO')

def setup():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger

LOGGER = setup()
INFO = LOGGER.info
WARN = LOGGER.warning
ERROR = LOGGER.error
