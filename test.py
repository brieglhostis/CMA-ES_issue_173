
from cma.evolution_strategy import fmin_con
from cma import ff
import numpy as np

"""
# Impossible
x, es = fmin_con(
    ff.sphere, 3 * [0], 1, g=lambda x: [1 - x[0]**2, -(1 - x[0]**2) - 1e-6], post_optimization=True,
    options={'termination_callback': lambda es: -1e-5 < es.mean[0]**2 - 1 < 1e-5,
             'verbose': -9})
"""

c = np.array([0, 0])
f = lambda x: np.sum(np.square(x - c))
# g = lambda x: [1 - x[0]**2, 0.5 - x[1]**2]  # |x1| >= 1 and |x2| >= sqrt(0.5)
g = lambda x: [x[0]**2 - x[1], 0.5**0.5 - x[0]]  # x2 >= x1**2 and x1 >= sqrt(0.5)
# g = lambda x: [1 - x[0] - x[1], 1 + x[0] - x[1]]  # x2 >= 1 - X1 and x2 >= 1 + x1

x, es = fmin_con(
    f, 2 * [0], 1, g=g, post_optimization=True,
    options={'termination_callback': lambda es: -1e-5 < sum(g(es.mean)) < 1e-5,
             'seed': 1, 'verbose': -9})

print("xfavorite:", es.mean)
print("cov matrix:", es.sm.covariance_matrix)

print("f(xfavorite):", f(es.mean))
print("g(xfavorite):", g(es.mean))
print()
print(es.best_feasible)
if hasattr(es, 'best_feasible_post_opt'):
    print(es.best_feasible_post_opt)
