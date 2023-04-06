import logging
import time
from os.path import join
import coloredlogs

BLUE = '\033[94m'
ENDC = '\033[0m'

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def pretty_print(s):
    print(BLUE + "[" + get_time() + "] " + str(s) + ENDC)

def get_logger(log_dir, name):
    coloredlogs.install()
    logger = logging.getLogger(name)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger.addHandler(logging.FileHandler(join(log_dir, f'{name}.log')))
    return logger