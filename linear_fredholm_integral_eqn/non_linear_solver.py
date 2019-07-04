from linear_fredholm_integral_eqn.__paths__ import root_path_obj
from linear_fredholm_integral_eqn.__common_parameters__ import DTYPE

from linear_fredholm_integral_eqn.solver_class import LinearFredholmSolver

from functools import reduce

from ast import literal_eval

import numpy as np

import json

import scipy.integrate as integral

from sympy import lambdify, symbols, Function, integrate
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, implicit_application)
transformations = standard_transformations + (implicit_application,)


def function_reader(path):
    _syms = symbols('s t')

    with open(path, 'r') as f:
        lines = f.readlines()

    [_k, _g, _F, _mu, _lambda, _upper_limit, _lower_limit] = \
        [parse_expr(row, transformations=transformations) for row in lines]

    return {
        'k' : _k,
        'g' : _g,
        'F' : _F,
        'mu' : _mu,
        'lambda' : _lambda,
        'upper_limit' : _upper_limit,
        'lower_limit' : _lower_limit,
        'syms' : _syms
    }

class FunctionParser(object):
    def __init__(self, path):
        self._syms = symbols('s t x')

        with open(path, 'r') as f:
            lines = f.readlines()

        [self._k,self._F, self._g, self._mu, self._lambda, self._upper_limit, self._lower_limit] = \
            [parse_expr(row, transformations=transformations) for row in lines]

        self._F_prime = self._F.diff(self._syms[2])

class NonLinearFredholmSolver(object):
    def __init__(self,
                 path,
                 mode='NKM'):
        """
        eqn is of the form : mu * phi(s) - lambda * integral[K(s,t) * phi(t)] = g(s)
        :param path_to_file:
        """
        self._steps = 0
        self._func_parser = FunctionParser(path)

        self._x0 = None
        self._xn = None
        self._mu = float(self._func_parser._mu)
        self._lambda = float(self._func_parser._lambda)
        self._upper_limit = float(self._func_parser._upper_limit)
        self._lower_limit = float(self._func_parser._lower_limit)

        self._all_xn = []

        with open(root_path_obj.joinpath('__params__.json').as_posix(), 'r') as f:
            self._params = json.load(f)

        self._si = np.linspace(self._lower_limit, self._upper_limit, self._params['NKM_grid_size']['s'] + 1)
        self._ti = np.linspace(self._lower_limit, self._upper_limit, self._params['NKM_grid_size']['t'] + 1)

        #self._g = lambdify(self._func_parser._syms[0], self._func_parser._g, 'numpy')(self._si)
        g = lambda s: self._func_parser._g.subs({self._func_parser._syms[0] : s})
        self._g = np.array([g(s) for s in self._si]).astype(DTYPE)

        k = lambda s,t :self._func_parser._k.subs({self._func_parser._syms[0]: s, self._func_parser._syms[1]: t})

        self._kernel_mat = np.zeros((self._si.shape[0], self._si.shape[0]))
        for i, s in enumerate(self._si):
            for j, t in enumerate(self._si):
                self._kernel_mat[i,j] = k(s,t)

        self._kernel_mat = self._kernel_mat.astype(DTYPE)

        self._mode = mode

    def get_current_kernel(self):
        """
        --> substitute value of self._xn in k and k_prime
        :return:
        """
        F = lambda x: self._func_parser._F.subs({self._func_parser._syms[2]: x})
        F_arr = np.array([F(x) for x in self._xn]).astype(DTYPE)

        if self._mode == 'NKM':
            F = lambda x: self._func_parser._F_prime.subs({self._func_parser._syms[2]: x})
            F_prime_arr = np.array([F(x) for x in self._xn]).astype(DTYPE)
        elif self._mode == 'MNKM':
            F = lambda x: self._func_parser._F_prime.subs({self._func_parser._syms[2]: x})
            F_prime_arr = np.array([F(x) for x in self._x0]).astype(DTYPE)


        k = self._kernel_mat * F_arr
        k_prime = self._kernel_mat * F_prime_arr

        return k, k_prime

    def inf_norm(self, func):
        return np.max(np.abs(func(self._si)))

    def set_x0(self, x0):
        """
        --> init self._x0, self._xn
        --> self._x0, self._xn will be arrays
        :param x0: numpy function
        :return:
        """
        self._x0 = np.array([x0(s) for s in self._si]).astype(DTYPE)
        self._xn = np.array([x0(s) for s in self._si]).astype(DTYPE)
        self._steps = 0

    def next_iterate(self):
        k, k_prime = self.get_current_kernel()

        h = (self._upper_limit - self._lower_limit)/self._params['NKM_grid_size']['t']

        f = self._g - \
        self._mu * self._xn + \
        self._lambda * np.array([integral.romb(k[row_idx, :], dx = h) for row_idx in range(k.shape[0])])


        linear_integral_eqn_solver = LinearFredholmSolver(
                                        k=k_prime,
                                        g=f,
                                        mu=self._mu,
                                        lambd=self._lambda,
                                        integral_limit=(self._lower_limit, self._upper_limit)
                                    )
        self._xn = self._xn + linear_integral_eqn_solver.approximate_solution()
        self._steps += 1

    def approximate_solution(self):
        """
        --> x0 is the initial guess
        --> returns final approximate function
        :param x0: numpy function
        :return: numpy function
        """
        true_sol = lambda s : np.pi * s**2
        itr = 0
        while self.inf_norm(lambda s: self._xn - true_sol(s)) > self._params['convergence_threshold']:
            self.next_iterate()
            if itr > 100:
                break
            itr += 1