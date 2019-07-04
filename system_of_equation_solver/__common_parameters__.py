import numpy as np

l_key = {
    'l_inf': np.inf,
    'l1': 1,
    'l2': 2
}

LOGGER_NAME = 'system_of_equation_solver'

MAX_ITER = 10 ** 2
CONVERGENCE_THRESH = 10 ** -4

DTYPE = np.float32
