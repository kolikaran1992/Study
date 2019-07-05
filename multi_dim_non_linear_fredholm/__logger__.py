import logging
from system_of_equation_solver.__common_parameters__ import LOGGER_NAME
from system_of_equation_solver.__paths__ import path_to_logs
import datetime

logger = logging.getLogger(LOGGER_NAME)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')

# define file handler and set formatter
file_handler = logging.FileHandler(path_to_logs.joinpath(datetime.datetime.now().strftime("%Y-%m-%d")).as_posix())
file_handler.setFormatter(formatter)

# define console handler and set formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.CRITICAL)