from linear_fredholm_integral_eqn.__paths__ import root_path_obj
from linear_fredholm_integral_eqn.__common_parameters__ import DTYPE

from functools import reduce

from ast import literal_eval

import numpy as np

import json

from sympy import lambdify, symbols, Matrix
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, implicit_application)
transformations = standard_transformations + (implicit_application,)

def parse_exp(exp):
    return parse_expr(exp, transformations=transformations)

def function_reader(path):
    _syms = symbols('s t')

    with open(path, 'r') as f:
        lines = f.readlines()

    [_k, _g] = \
        [parse_expr(row, transformations=transformations) for row in lines[:2]]

    [_mu, _lambda, _upper_limit, _lower_limit] = \
        [literal_eval(numeral) for numeral in lines[2:]]

    return {
        'k' : _k,
        'g' : _g,
        'mu' : _mu,
        'lambda' : _lambda,
        'upper_limit' : _upper_limit,
        'lower_limit' : _lower_limit,
        'syms' : _syms
    }


class LinearFredholmSolver(object):
    def __init__(self,
                 k=None,
                 g=None,
                 mu=None,
                 lambd=None,
                 integral_limit = None):
        """
        eqn is of the form : mu * phi(s) - lambda * integral[K(s,t) * phi(t)] = g(s)
        :param path_to_file:
        """
            ## numpy matrix of size (sj, ti)
        self._k = k

            ## numpy array of size (sj,)
        self._g = g

        self._mu = mu
        self._lambda = lambd
        self._upper_limit = integral_limit[1]
        self._lower_limit = integral_limit[0]

        with open(root_path_obj.joinpath('__params__.json').as_posix(), 'r') as f:
            self._params = json.load(f)

        self._ti = np.linspace(self._lower_limit, self._upper_limit, self._params["steps"]["t"]+1)
        self._sj = np.linspace(self._lower_limit, self._upper_limit, self._params["steps"]["s"]+1)

    def D(self):
        a = np.array([[1, 0, 0], [-1, 1, 0], [1, -2, 1]])
        num_zero_cols = self._params["steps"]["s"] + 1 - a.shape[0]
        zeros = np.zeros((3, num_zero_cols))
        a = np.concatenate((a,zeros), axis=1)
        rows = [a] + [np.roll(a.copy(), 2*(i), axis = 1) for i in range(1, self._params["steps"]["t"]//2)]
        return np.concatenate(rows, axis = 0)

    def Q_ele(self, j = 0):
        h = (self._upper_limit - self._lower_limit)/self._params['steps']['t']

        alpha_j = (h/3) * (self._k[:, j] + 4*self._k[:, j+1] + self._k[:, j+2])

        beta_j = (h/3) * (4*self._k[:, j+1] + 2*self._k[:, j+2])
        gamma_j = (h/3) * self._k[:, j+2]

        return alpha_j, beta_j, gamma_j

    def G(self):
        return self._g.astype(DTYPE)

    def approximate_solution(self):
        row_ = reduce(lambda x, y: x+y, [self.Q_ele(j=_j) for _j in range(0, self._params['steps']['t'], 2)])
        row_s = np.array([f for f in row_]).astype(DTYPE)
        Q = row_s.T

        D = self.D()

        a = np.matmul(Q, D)
        a = self._mu * np.eye(a.shape[0]) - a
        b = self.G()
        y = np.linalg.solve(a,b)

        return y
        #f = lambda s: ((self._g(s).reshape(-1, 1) + self._lambda*np.matmul(np.matmul(row_s(s).T, D), y.reshape(-1, 1)).astype(DTYPE))/self._mu).reshape(-1,).tolist()

        #return f


class LinearFredholmSolver_func_of_s(object):
    def __init__(self,
                 k=None,
                 g=None,
                 mu=None,
                 lambd=None,
                 integral_limit = None):
        """
        eqn is of the form : mu * phi(s) - lambda * integral[K(s,t) * phi(t)] = g(s)
        :param path_to_file:
        """
        self._k = k
        self._g = g
        self._mu = mu
        self._lambda = lambd
        self._upper_limit = integral_limit[1]
        self._lower_limit = integral_limit[0]

        with open(root_path_obj.joinpath('__params__.json').as_posix(), 'r') as f:
            self._params = json.load(f)

        self._ti = np.linspace(self._lower_limit, self._upper_limit, self._params["steps"]["t"]+1)
        self._sj = np.linspace(self._lower_limit, self._upper_limit, self._params["steps"]["s"]+1)

    def D(self):
        a = np.array([[1, 0, 0], [-1, 1, 0], [1, -2, 1]])
        num_zero_cols = self._params["steps"]["s"] + 1 - a.shape[0]
        zeros = np.zeros((3, num_zero_cols))
        a = np.concatenate((a,zeros), axis=1)
        rows = [a] + [np.roll(a.copy(), 2*(i), axis = 1) for i in range(1, self._params["steps"]["t"]//2)]
        return np.concatenate(rows, axis = 0)

    def Q_ele(self, j = 0):
        h = (self._upper_limit - self._lower_limit)/self._params['steps']['t']

        alpha_j = lambda s: (h/3) * (self._k(s, self._ti[j]) + 4*self._k(s, self._ti[j+1]) + self._k(s, self._ti[j+2]))
        beta_j = lambda s: (h/3) * (4*self._k(s, self._ti[j+1]) + 2*self._k(s, self._ti[j+2]))
        gamma_j = lambda s: (h/3) * self._k(s, self._ti[j+2])

        return alpha_j, beta_j, gamma_j

    def G(self):
        return self._g(self._sj).astype(DTYPE)

    def approximate_solution(self):
        row_ = reduce(lambda x, y: x+y, [self.Q_ele(j=_j) for _j in range(0, self._params['steps']['t'], 2)])
        row_s = lambda s: np.array([f(s) for f in row_]).astype(DTYPE)
        Q = row_s(np.array(self._sj)).T

        D = self.D()

        a = np.matmul(Q, D)
        a = self._mu * np.eye(a.shape[0]) - a
        b = self.G()
        y = np.linalg.solve(a,b)

        f = lambda s: ((self._g(s).reshape(-1, 1) + self._lambda*np.matmul(np.matmul(row_s(s).T, D), y.reshape(-1, 1)).astype(DTYPE))/self._mu).reshape(-1,).tolist()

        return f

