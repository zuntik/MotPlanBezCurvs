import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from functools import lru_cache


def basis(k, n, tau):
    """Returns the evaluation of a bernstein basis"""
    return comb(n, k) * (1 - tau) ** (n - k) * tau ** k


@lru_cache
def antiderivmat(n, tf):
    """Returns an integral/antiderivative matrix

    Returns an integral/antiderivative matrix that must be multiplied by a column vector
    to obtain the control points of the integral function"""
    return np.tril(np.ones((n + 2, n + 1)) * tf / (n + 1), -1)


def antideriv(p, tf, p0):
    """Returns the control points of the integral function of for the control points of p"""
    return p0 + antiderivmat(p.shape[0] - 1, tf) @ p


@lru_cache
def degrelevmat(n, m):
    """Returns a matrix for degree elevation"""
    return np.array([[comb(i, j) * comb(m - i, n - j) if max(0, i - m + n) <= j <= min(n, i) else 0
                      for j in range(n + 1)] for i in range(m+1)]) / comb(m, n)


def degrelev(p, m):
    """Performs degree elevation"""
    return p if p.shape[0] - 1 > m else degrelevmat(p.shape[0] - 1, m) @ p


@lru_cache
def derivmat(n, tf):
    """Returns a matrix to perform derivation"""
    return n / tf * (np.hstack((np.zeros((n, 1)), np.eye(n))) - np.hstack((np.eye(n), np.zeros((n, 1)))))


def deriv(p, tf):
    """Performs derivation of control points p with final time t"""
    return derivmat(p.shape[0] - 1, tf) @ p


@lru_cache
def derivelevmat(n, tf):
    """Performs derivation while preserving order"""
    return derivmat(n + 1, tf) @ degrelevmat(n, n + 1)


def derivelev(p, tf):
    """Performs degree elevation"""
    return derivelevmat(p.shape[0] - 1, tf) @ p


@lru_cache
def evalmat(n, tf, times):
    """Returns a matrix to perform evaluation on times times"""
    return np.array([[basis(j, n, ti / tf) for j in range(n + 1)] for ti in np.array(times).flatten()])


def eval(p, tf, times):
    """Performs evaluation on times times"""
    return evalmat(p.shape[0] - 1, tf, times) @ p


def integr(p, tf):
    """Calculates the integral"""
    return tf / p.shape[0] * np.sum(p, 0)


@lru_cache
def mulmat(m, n):
    pass


def mul(p1, p2):
    """Control points for multiplication"""
    if p1.shape[0] < p2.shape[0]:
        p1, p2 = p2, p1
    m, n = p1.shape[0]-1, p2.shape[0]-1
    return np.array([np.sum(
        [comb(i, j) * comb(m + n - i, m - j) * p1[j, :] * p2[i - j, :] for j in range(max(0, i - n), min(m, i) + 1)], 0)
                     for i in range(m + n + 1)]) / comb(m + n, n)

def mul2(p1, p2):
    if p1.shape[0] < p2.shape[0]:
        p1,p2 = p2, p1
    m, n = p1.shape[0]-1, p2.shape[0]-1
    out = np.zeros((m+n+1, p1.shape[1]))
    for i in range(m + n + 1):
        for j in range(m+1):
            if j < i-n or j > i:
                out[i] += 0
            else:
                out[i] += comb(i, j) * comb(m + n - i, m - j) * p1[j, :] * p2[i - j, :] / comb(m+n,n)

    return out

a = np.array([1,2,3]).reshape((-1,1))
b = np.array([4,6]).reshape((-1, 1))
print(mul(a,b))
print(mul2(a,b))

def pow(p, y):
    """Control points for power"""
    if y == 0:
        return np.ones((1, p.shape[1]))
    temp_p = pow(p, y // 2)
    if y % 2 == 0:
        return mul(temp_p, temp_p)
    else:
        return mul(p, mul(temp_p, temp_p))


def add(p1, p2):
    """Calulates the addition of two polynomials"""
    if p1.shape[0] > p2.shape[0]:
        p2 = degrelev(p2, p1.shape[0] - 1)
    elif p2.shape[0] > p1.shape[0]:
        p1 = degrelev(p1, p2.shape[0] - 1)
    return p1 + p2


@lru_cache
def tomonmat(n, tf):
    """Calculates a matrix to convert the control points to monomial polynomails"""
    return np.flipud([[0 if i > k else comb(n, k) * comb(k, i) * (-1) ** (k - i) for i in range(n + 1)] for k in
                      range(n + 1)]) / np.array([tf ** i for i in range(n, -1, -1)]).reshape((-1, 1))


def tomon(p, tf):
    """Convers the control points to monomial coefficients"""
    return tomonmat(p.shape[0] - 1, tf) @ p


def plot(p, t, plotcpts=False, ax=None):
    """Plots the polynomial"""
    n, dim = p.shape
    times = np.linspace(0, t, 100)
    vals = eval(p, t, times)
    curveplot = None
    pointsplot = None
    if ax is None:
        if dim == 3:
            ax = plt.figure().add_subplot(111, projection='3d')
        elif dim == 1 or dim == 2:
            _, ax = plt.subplots()
    if dim != 1 and dim != 2 and dim != 3:
        raise ValueError('Unsupported dim')
    if dim == 1:
        curveplot, = ax.plot(times, vals)
        if plotcpts:
            pointsplot = ax.scatter(np.linspace(0, t, n), p)
    elif dim == 2:
        curveplot, = ax.plot(vals[:, 0], vals[:, 1])
        if plotcpts:
            pointsplot = ax.scatter(p[:, 0], p[:, 1])
    elif dim == 3:
        ax.plot(vals[:, 0], vals[:, 1], vals[:, 2])
    return curveplot, pointsplot
