from multi_dim_non_linear_fredholm.__paths__ import root_path_obj
from multi_dim_non_linear_fredholm.__common_parameters__ import DTYPE

from functools import reduce

import numpy as np

import json

from sympy import lambdify, symbols, Matrix
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, implicit_application)
transformations = standard_transformations + (implicit_application,)

def parse_exp(exp):
    return parse_expr(exp, transformations=transformations)

def function_reader(path):
    _syms = symbols('x y s t')

    with open(path, 'r') as f:
        lines = f.readlines()

    [_k, _g] = \
        [parse_expr(row, transformations=transformations) for row in lines[:2]]

    [_mu, _lambda, _upper_limit1, _lower_limit1, _upper_limit2, _lower_limit2] = \
        [float(numeral) for numeral in lines[2:]]

    return {
        'k' : _k,
        'g' : _g,
        'mu' : _mu,
        'lambda' : _lambda,
        'upper_limit' : [_upper_limit1, _upper_limit2],
        'lower_limit' : [_lower_limit1, _lower_limit2],
        'syms' : _syms
    }


def return_k_mat(k, sj, tj, syms):
    k_x_y = lambda s, t: k.subs({syms[2]: s, syms[3]: t})
    a = np.array([[k_x_y(s, t) for t in tj] for s in sj])
    q = lambda x, y: [exp.subs({syms[0]: x, syms[1]: y}) for exp in a.reshape(1, -1)[0]]
    b = np.array(reduce(lambda x, y: x + y, [[q(s, t) for t in tj] for s in sj])).astype(DTYPE)

    return b

def return_g_vec(g, sj, tj, syms):
    g_x_y = lambda x, y: g.subs({syms[0]: x, syms[1]: y})
    a = np.array([[g_x_y(s, t) for t in tj] for s in sj]).astype(DTYPE)
    return a.reshape(-1,1)

class FredholmSolver(object):
    def __init__(self,
                 k=None,
                 g=None,
                 mu=None,
                 lambd=None,
                 upper_limit = None,
                 lower_limit = None):
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
        self._upper_limit = upper_limit
        self._lower_limit = lower_limit

        with open(root_path_obj.joinpath('__params__.json').as_posix(), 'r') as f:
            self._params = json.load(f)


    def approximate_solution(self, reshape=True):
        h1 = (self._upper_limit[0] - self._lower_limit[0])/self._params["steps"]["s"]
        h2 = (self._upper_limit[1] - self._lower_limit[1]) / self._params["steps"]["t"]

        y = np.linalg.solve(np.eye(self._k.shape[0]) * self._mu - self._lambda * h1 * h2 * self._k, self._g)

        if reshape:
            return y.reshape(self._params["steps"]["s"]+1, self._params["steps"]["t"] + 1)
        else:
            return y