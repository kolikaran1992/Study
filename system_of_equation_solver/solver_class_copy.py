import re
from sympy import lambdify, symbols, MatrixSymbol, Matrix, diff, srepr
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, implicit_application)
transformations = standard_transformations + (implicit_application,)
import numpy as np
from system_of_equation_solver.__common_parameters__ import LOGGER_NAME, DTYPE, CONVERGENCE_THRESH, MAX_ITER, l_key
import system_of_equation_solver.__logger__
from system_of_equation_solver.__paths__ import root_path_obj
import logging
logger = logging.getLogger(LOGGER_NAME)

import json
with open(root_path_obj.joinpath('__params__.json'), 'r') as f:
    params = json.load(f)

class SysEqnSolver(object):
    """
    --> takes path of func text as input
    --> find zeros of that function via NKM if initial conditions are satisfied
    """
    def __init__(self,
                 func_str_path,
                 mode = params['mode'],
                 vec_norm = params['vec_norm']):

        self._vec_norm_type = vec_norm
        self._mode = mode
        if mode == 'NKM':
            logger.info('The current mode is Newton Kantorovich')
        elif mode == 'MNKM':
            logger.info('The current mode is Modifoed Newton Kantorovich')
        else:
            logger.error('Mode value should be "NKM" or "MNKM"')
            return

        func = []

        self._func_path = func_str_path
        try:
            with open(self._func_path, 'r') as file_obj:
                lines = [line.strip() for line in file_obj.readlines()]
                func = lines[1:]
                self._syms = symbols(lines[0])
        except:
            logger.exception('file at path could not be read')

        del lines

        self._F = Matrix([parse_expr(row, transformations=transformations) for row in func])
        logger.debug('F{} = {}'.format(self._syms, self._F))

        self._DF = Matrix([[diff(func, sym) for sym in self._syms] for func in self._F])
        logger.debug('DF{} = {}'.format(self._syms, self._DF))

        self._x_conv = None
        self._x0 = None
        self._steps = 0

        self._vec_shape = (len(self._syms), 1)

        self._f_vals = []
        self._f_norms = []
        self._all_x = []

        del func

    def make_array(self, *x):
        return np.array(x, dtype=DTYPE).reshape(self._vec_shape)

    def initialize_x_conv(self, *args):
        self._x_conv = self.make_array(*args)
        self._x0 = self.make_array(*args)
        self._steps = 0

        self._f_vals = []
        self._f_norms = []
        self._all_x = []

        logger.debug('initial value of x = {}'.format(self._x_conv))

    def operator_norm(self, numerical_matrix):
        """
        --> the corresponding norm for an operator defined from l_inf to l_inf
        :param numerical_matrix:
        :return:
        """
        norm = None

        if numerical_matrix is None:
            return norm

        try:
            norm = np.linalg.norm(numerical_matrix, l_key[self._vec_norm_type])
            logger.debug('operator norm = {}'.format(norm))
        except:
            logger.exception('operator norm could not be calculated')

        return norm

    def vector_norm(self, numerical_vector):
        """
        --> l_inf norm
        :param numerical_vector:
        :return:

        """
        norm = None

        if numerical_vector is None:
            return norm

        try:
            norm = np.linalg.norm(numerical_vector, l_key[self._vec_norm_type])
            logger.debug('vector norm = {}'.format(norm))
        except:
            logger.exception('vector norn could not be calculated')

        return norm

    def calc_numeric_F(self, *args):
        f = None

        try:
            f = lambdify(self._syms, self._F, 'numpy')(*args).reshape(self._vec_shape).astype(DTYPE)
            logger.debug('F{} = {}'.format(args, f))
        except:
            logger.exception('F = {}, could not be calculated at {}'.format(self._F, args))

        return f

    def calc_numeric_DF(self, *args):
        df = None

        try:
            df = lambdify(self._syms, self._DF, 'numpy')(*args).astype(DTYPE)
            logger.debug('DF{} = {}'.format(args, df))
        except:
            logger.exception('DF = {}, could not be calculated at {}'.format(self._DF, args))

        return df

    def calc_numeric_DF_inv(self, *args):
        df_inv = None

        try:
            df = lambdify(self._syms, self._DF, 'numpy')(*args).astype(DTYPE)

            if df is not None:
                df_inv = np.linalg.inv(df)
                logger.debug('DF_inv{} = {}'.format(args, df_inv))
        except:
            logger.exception('DF_inv = {}, could not be calculated at {}'.format(self._DF, args))

        return df_inv


    def calc_F_lip(self):
        with open(self._func_path, 'r') as file_obj:
            text = file_obj.read()
            for sym in text.split('\n')[0].split(' '):
                text = re.sub(r'[{}]'.format(sym), '{}_1'.format(sym), text)

            lines = [line.strip() for line in text.split('\n')]
            func1 = lines[1:]
            syms1 = symbols(lines[0])

        del lines, text

        _F1 = Matrix([parse_expr(row, transformations=transformations) for row in func1])
        _DF1 = Matrix([[diff(func, sym) for sym in syms1] for func in _F1])

        return self._DF - _DF1

    def calc_centre_F_lip(self):
        with open(self._func_path, 'r') as file_obj:
            text = file_obj.read()
            for sym in text.split('\n')[0].split(' '):
                text = re.sub(r'[{}]'.format(sym), '{}_1'.format(sym), text)

            lines = [line.strip() for line in text.split('\n')]
            func1 = lines[1:]
            syms1 = symbols(lines[0])

        del lines, text

        _F1 = Matrix([parse_expr(row, transformations=transformations) for row in func1])
        _DF1 = Matrix([[diff(func, sym) for sym in syms1] for func in _F1])

        _temp = self._x0.reshape((self._vec_shape[0],)).tolist()

        df_x0_inv = self.calc_numeric_DF_inv(*_temp)

        return Matrix(df_x0_inv) * (self._DF - _DF1)


    def calc_centre0_F_lip(self):
        _temp = self._x0.reshape((self._vec_shape[0],)).tolist()

        df_x0_inv = self.calc_numeric_DF_inv(*_temp)

        return Matrix(df_x0_inv) * self._DF - Matrix.eye(self._vec_shape[0])


    def calculate_b0(self):
        _temp = self._x0.reshape((self._vec_shape[0],)).tolist()
        df_x0_inv = self.calc_numeric_DF_inv(*_temp)


        if df_x0_inv is None:
            return None

        b0 = self.operator_norm(df_x0_inv)
        logger.debug('b0 = {}'.format(b0))

        return b0

    def calculate_eta0(self):
        _temp = self._x0.reshape((self._vec_shape[0],)).tolist()

        df_x0_inv = self.calc_numeric_DF_inv(*_temp)

        f_x0 = self.calc_numeric_F(*_temp)
        logger.debug('F{} = {}'.format(_temp, f_x0))

        if df_x0_inv is None or f_x0 is None:
            return None

        eta0 = self.vector_norm(np.matmul(df_x0_inv, f_x0))
        logger.debug('eta0 = {}'.format(eta0))

        return eta0

    def calculate_initial_conditions(self, l):

        if self._x_conv is None:
            logger.error('initialize x_conv before calculating initial conditions')
            return None


        logger.debug('calculating initial conditions')

        _temp = self._x0.reshape((self._vec_shape[0],)).tolist()
        df_x0_inv = self.calc_numeric_DF_inv(*_temp)

        f_x0 = self.calc_numeric_F(*_temp)
        logger.debug('F{} = {}'.format(_temp, f_x0))

        if df_x0_inv is None or f_x0 is None:
            return None

        b0 = self.operator_norm(df_x0_inv)
        logger.debug('b0 = {}'.format(b0))

        f_x0 = self.calc_numeric_F(*_temp)
        logger.debug('F{} = {}'.format(_temp, f_x0))

        eta0 = self.vector_norm(np.matmul(df_x0_inv, f_x0))
        logger.debug('eta0 = {}'.format(eta0))


        if b0 is None or eta0 is None:
            return None

        h = l * b0 * eta0
        logger.debug('h = {}'.format(h))

        if h > 0.5:
            logger.error('h must be less than 0.5 but is {}'.format(round(h, 2)))
            return None

        r0 = np.multiply(1 - np.sqrt(1 - 2 * h), eta0) / h
        logger.debug('r0 = {}'.format(r0))

        return {
            'b0': b0,
            'eta0': eta0,
            'h': h,
            'r0': r0
        }

    def update_x_conv(self, f_prev = None):
        logger.debug('updating x_conv')

        _temp = self._x_conv.reshape((self._vec_shape[0],)).tolist()

        df_x0_inv = None

        if self._mode == 'NKM':
            df_x0_inv = self.calc_numeric_DF_inv(*_temp)

        elif self._mode == 'MNKM':
            _x0 = self._x0.reshape((self._vec_shape[0],)).tolist()
            df_x0_inv = self.calc_numeric_DF_inv(*_x0)

        if df_x0_inv is None:
            logger.error('The update step calculation has failed')
            return

        if f_prev is None:
            f_x0 = self.calc_numeric_F(*_temp)
        else:
            f_x0 = f_prev

        self._x_conv -= np.matmul(df_x0_inv, f_x0)
        self._steps += 1

    def simulate(self, l):
        logger.debug('simulating {}'.format(self._mode))

        init = self.calculate_initial_conditions(l)

        if init is None:
            logger.error('initial conditions could not be calculated for {}'.format(self._mode))
            return

        while self._steps < MAX_ITER:
            logger.debug('calculating f norm at updated x')
            _temp = self._x_conv.reshape((self._vec_shape[0],)).tolist()
            f_updated = self.calc_numeric_F(*_temp)
            f_norm = self.vector_norm(f_updated)

            self._f_norms.append(f_norm)
            self._f_vals.append(f_updated)
            self._all_x.append(_temp)

            if f_updated is None:
                logger.error('method could not converge since f can not be calculated at the updated x')
                break

            if f_norm <= CONVERGENCE_THRESH:
                logger.info('{} converged succesfully in {} iterations'.format(self._mode, self._steps))
                break

            self.update_x_conv(f_prev=f_updated)
