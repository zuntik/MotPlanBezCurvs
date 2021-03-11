from scipy.integrate import solve_ivp
from bernsteinlib import *
from scipy.optimize import minimize
import numpy as np

print(np.array([1, 2, 3]))

b = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]],
              [[17, 18], [19, 20], [21, 22], [23, 24]]])
print(b)
print(b[1, 1, 0])

cons = ({'type': 'eq', 'fun': lambda x: [x[1] - x[0] - 1, x[1]]})


def fun(x):
    return x[0] ** 2 + x[1] ** 2


res = minimize(fun, (0, 0), method='SLSQP', constraints=cons)

print(res)


def recplot(x, CONSTANTS):
    def odefunc(t, val, v, w):
        x, y, psi = val
        dx = v(t) * np.cos(psi)
        dy = v(t) * np.sin(psi)
        dpsi = w(t)
        return np.array([dx, dy, dpsi])

    def v(t): return bernsteinEval(x[:, 0], CONSTANTS['T'], t)

    def w(t): return bernsteinEval(x[:, 1], CONSTANTS['T'], t)

    sol = solve_ivp(odefunc, [0, CONSTANTS['T']], CONSTANTS['xi'], args=(v, w), dense_output=True, vectorized=True)
    return np.linspace(0, CONSTANTS['T'], 1000), sol.sol(np.linspace(0, CONSTANTS['T'], 1000))


x = np.array([[1, 0.5, 2, 6, 2, 1], [0, 0, 1, 2, -1, -3]]).T
CONSTANTS = {'T': 10, 'xi': np.array([0, 0, 0])}

t, xy = recplot(x, CONSTANTS)

plt.plot(t, xy[:2, :].T)
plt.show()
plt.plot(xy[0, :], xy[1, :].T)
plt.show()
