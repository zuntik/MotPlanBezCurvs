import numpy as np
from scipy.special import comb

def bezProductCoefficients(m, n=None):
    if n is None:
        n = m

    coefMat = np.zeros(((m+1)*(n+1), m+n+1))

    for k in range(m+n+1):
        den = comb(m+n, k)
        for j in range(max(0, k-n), min(m, k)+1):
            coefMat[m*j+k, k] = comb(m, j)*comb(n, k-j)/den

    return coefMat

def multiplyBezCurves(multiplier, multiplicand, coefMat=None):
    multiplier = np.atleast_2d(multiplier)
    multiplicand = np.atleast_2d(multiplicand)
    m = multiplier.shape[1] - 1
    n = multiplicand.shape[1] - 1

    augMat = np.dot(multiplier.T, multiplicand)
    newMat = augMat.reshape((1, -1))

    if coefMat is None:
        coefMat = bezProductCoefficients(m, n)

    return np.dot(newMat, coefMat)


a = np.array([1,2,3])
b = np.array([4,6])

print(multiplyBezCurves(b,a))
print(multiplyBezCurves(a,b))
