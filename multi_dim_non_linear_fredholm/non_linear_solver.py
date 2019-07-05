from multi_dim_non_linear_fredholm.__paths__ import root_path_obj
from multi_dim_non_linear_fredholm.__common_parameters__ import DTYPE

from multi_dim_non_linear_fredholm.solver_class import FredholmSolver

from functools import reduce

import numpy as np

import json

import scipy.integrate as integral
from scipy.integrate import simps

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
        self._syms = symbols('u x y s t')

        with open(path, 'r') as f:
            lines = f.readlines()

        [self._k, self._F, self._g] = \
            [parse_expr(row, transformations=transformations) for row in lines[:3]]

        [self._mu, self._lambda, self._upper_limit1, self._lower_limit1, self._upper_limit2, self._lower_limit2] = \
            [float(numeral) for numeral in lines[3:]]

        self._F_prime = self._F.diff(self._syms[0])

    def k_mat(self, sj, tj):
        k_x_y = lambda s, t: self._k.subs({self._syms[3]: s, self._syms[4]: t})
        a = np.array([[k_x_y(s, t) for t in tj] for s in sj])
        q = lambda x, y: [exp.subs({self._syms[1]: x, self._syms[2]: y}) for exp in a.reshape(1, -1)[0]]
        b = np.array(reduce(lambda x, y: x + y, [[q(s, t) for t in tj] for s in sj])).astype(DTYPE)
        return b

    def g_vec(self, sj, tj):
        g_x_y = lambda x, y: self._g.subs({self._syms[1]: x, self._syms[2]: y})
        a = np.array([[g_x_y(s, t) for t in tj] for s in sj]).astype(DTYPE)
        return a.reshape(-1, 1)


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
        self._mu = self._func_parser._mu
        self._lambda = self._func_parser._lambda
        self._upper_limit = (self._func_parser._upper_limit1, self._func_parser._upper_limit2)
        self._lower_limit = (self._func_parser._lower_limit1, self._func_parser._lower_limit2)

        with open(root_path_obj.joinpath('__params__.json').as_posix(), 'r') as f:
            self._params = json.load(f)

        self._si = np.linspace(self._lower_limit[0], self._upper_limit[0], self._params['steps']['s'] + 1)
        self._ti = np.linspace(self._lower_limit[1], self._upper_limit[1], self._params['steps']['t'] + 1)

        self._g = self._func_parser.g_vec(self._si, self._ti)

        self._kernel_mat = self._func_parser.k_mat(self._si, self._ti)

        self._mode = mode

    def get_current_kernel(self):
        """
        --> substitute value of self._xn in k and k_prime
        :return:
        """
        F = lambda x: self._func_parser._F.subs({self._func_parser._syms[0]: x})
        F_arr = np.array([F(x) for x in self._xn]).astype(DTYPE)

        if self._mode == 'NKM':
            F = lambda x: self._func_parser._F_prime.subs({self._func_parser._syms[0]: x})
            F_prime_arr = np.array([F(x) for x in self._xn]).astype(DTYPE)
        elif self._mode == 'MNKM':
            F = lambda x: self._func_parser._F_prime.subs({self._func_parser._syms[0]: x})
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
        self._x0 = np.array([[x0(s, t) for t in self._ti] for s in self._si]).reshape(-1,1).astype(DTYPE)
        self._xn = np.array([[x0(s, t) for t in self._ti] for s in self._si]).reshape(-1,1).astype(DTYPE)
        self._steps = 0

    def next_iterate(self):
        k, k_prime = self.get_current_kernel()

        s = self._params['steps']['s']
        t = self._params['steps']['t']

        f = self._g - \
        self._mu * self._xn + \
        self._lambda * np.array([simps(simps(k_i, self._ti), self._si) for
                                 k_i in k.reshape((s+1)*(t+1), s+1, t+1)]).astype(DTYPE).reshape(-1, 1)

        linear_integral_eqn_solver = FredholmSolver(
                                        k=k_prime,
                                        g=f,
                                        mu=self._mu,
                                        lambd=self._lambda,
                                        lower_limit = self._lower_limit,
                                        upper_limit = self._upper_limit
                                    )

        self._xn = self._xn + linear_integral_eqn_solver.approximate_solution(reshape=False)
        self._steps += 1
        return linear_integral_eqn_solver.approximate_solution(reshape=False)

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