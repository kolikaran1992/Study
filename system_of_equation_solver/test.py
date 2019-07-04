import numpy as np
from system_of_equation_solver.solver_class import SysEqnSolver

class combined(object):
    def __init__(self,
                 solver1,
                 solver2,
                 l1,
                 l2):
        self._solver1 = solver1
        self._solver2 = solver2
        self._l1 = l1
        self._l2 = l2

    def initialize_x(self, *x0):
        self._solver1.initialize_x_conv(*x0)
        self._solver2.initialize_x_conv(*x0)

    def calculate_initial_conditions(self):
        init1 = self._solver1.calculate_initial_conditions(self._l1)
        init2 = self._solver2.calculate_initial_conditions(self._l2)

        return init1, init2

    def simulate(self):
        self._solver1.simulate(self._l1)
        self._solver2.simulate(self._l2)



path = '/home/aptara/STUDY/NKM/notebooks/functions/func9'

l_inf = 6
l2 = np.sqrt(6)
points = []
max_ = 3
min_ = -3
x = np.arange(min_, max_, 0.1).tolist()
y = np.arange(min_, max_, 0.1).tolist()
c = combined(SysEqnSolver(path, mode='NKM', vec_norm='l_inf'),
             SysEqnSolver(path, mode='NKM', vec_norm='l2'),
             6, np.sqrt(6))
from itertools import product
prod = list(product(x, y))
for a, b in prod:
    x0 = [a, b]
    c.initialize_x(*x0)
    init1, init2 = c.calculate_initial_conditions()

    if init1 is not None and init2 is not None:
        points.append(x0)

lengths = []
for p in points:
    c.initialize_x(*p)
    c.simulate()
    lengths.append((len(c._solver1._f_norms), len(c._solver2._f_norms)))

print([(prod[i],x) for i, x in enumerate(lengths) if x[0] != x[1]])