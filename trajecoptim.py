from scipy.optimize import minimize
import bernsteinlib as bern
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import time


def logbarrierfunc(delta, z, usesigma):
    """Performs a log barrier functional on z"""
    if usesigma:
        z = np.where(z >= delta, np.tanh(z), z)
    k = 2
    return np.where(z > delta, -np.log(np.abs(z)), ((k-1)/k)*(((z-k*delta)/((k-1)*delta))**k-1) - np.log(delta))


def matrify(x, constants):
    """Transforms a flattened vector of control points to a matrix"""
    x = x.reshape((constants['Nv'], -1))
    x_mat = [
        np.concatenate((
            np.concatenate((
                constants['xi'][i, :].reshape((1, -1)),
                x[i, :(constants['N']-1)*constants['numvars']].reshape((-1, constants['numvars'])),
                constants['xf'][i, :].reshape((1, -1))
            ), axis=0),
            x[i, (constants['N']-1)*constants['numvars']:].reshape((constants['N']+1, constants['numinputs']))
        ), axis=1)[:, :, np.newaxis]  # .reshape((constants['N']+1, constants['numvars']+constants['numinputs'], 1))
        for i in range(constants['Nv'])]
    # return np.reshape(x, (constants['N'] - 1, constants['numvars'], constants['Nv']))
    return np.concatenate(x_mat, axis=2)


def costfun(x, constants):
    """Calculates the cost functional"""
    j = 0
    if constants['uselogbar']:
        c = ineqconstr(x, constants)
        ceq = eqconstr(x, constants)
        j += np.sum(logbarrierfunc(0.1, c, constants['usesigma']))
        j += 1e5 * np.sum(logbarrierfunc(0.01, -(ceq ** 2), constants['usesigma']))

    x = matrify(x, constants)
    j += np.sum([constants['costfun_single'](x[:, :, i], constants) for i in range(constants['Nv'])])

    return j


def eqconstr(x, constants):
    """Deals with the equality constraints"""
    x = matrify(x, constants)
    return np.concatenate([constants['dynamics'](x[:, :, i], constants) for i in range(constants['Nv'])])
    #    # initial and final conditions
    #    constraints = [
    #        (constants['xi'] - x[0, :, :].T).flatten(),
    #        (constants['xf'] - x[-1, :, :].T).flatten()
    #    ]
    #    # dynamics
    #    constraints += [constants['dynamics'](x[:, :, i], constants) for i in range(constants['Nv'])]
    #    return np.concatenate(constraints)


def variablebounds(constants):
    """Creats a vector of lower and upper bounds"""
    return ([
                (constants['statebounds'][0, var], constants['statebounds'][1, var])
                for _ in range(constants['N']-1)
                for var in range(constants['numvars'])
            ]+[
                (constants['inputbounds'][1, inp], constants['inputbounds'][1, inp])
                for _ in range(constants['N']+1)
                for inp in range(constants['numinputs'])
            ])*constants['Nv'] if constants['statebounds'] is not None else None

    # return [(constants['vehiclebounds'][0, var], constants['vehiclebounds'][1, var])
    #         for _ in range(constants['N'] + 1)
    #         for var in range(constants['numvars'])
    #         for __ in range(constants['Nv'])] if constants['vehiclebounds'] is not None else None


def ineqconstr(x, constants):
    """ Deals with nonlinear inequality constraints"""
    x = matrify(x, constants)
    c = []

    # inter vehicles
    c += [veh_coll_avoid(x[:, :2, v1], x[:, :2, v2], constants)
          for v1 in range(constants['Nv']) for v2 in range(v1+1, constants['Nv'])]

    # obstacles
    c += [obs.avoid(x[:, :2, veh]) for obs in constants['obstacles'] for veh in range(constants['Nv'])]
    return np.concatenate(c) if c else np.array([])


def randinitguess(constants):
    """Calulates a random initial guess"""
    return np.random.rand(
        (constants['N'] - 1) * constants['numvars'] + (constants['N'] + 1) * constants['numinputs'], constants['Nv'])


def processconstants(constants_orig):
    """Returns a new dictionary with more fields that the rest of the functions need"""
    constants = constants_orig.copy()
    constants = {**constants, **{
        # common parameters
        'DiffMat': bern.derivelevmat(constants['N'], constants['T']),
        'ElevMat': bern.degrelevmat(constants['N'], constants['N'] * 10),
        # 'EvalMat': bern.evalmat(constants['N'], constants['T'], np.linspace(0,constants['T'], constants['N']*40)),
        'EvalMat': bern.evalmat(constants['N'], constants['T'], np.linspace(0, constants['T'], 1000)),
        'numvars': constants['xi'].shape[1],
        'Nv': constants['xi'].shape[0],
    }}

    constants.setdefault('obstacles_circles', [])
    constants.setdefault('obstacles_polygons', [])
    # noinspection PyTypeChecker
    constants = {**constants, **{
        'obstacles':
            [TOLCircle(c[:-1], c[-1], constants['ElevMat'], constants['min_dist_obs'])
             for c in constants['obstacles_circles']] +
            [TOLPolygon(m) for m in constants['obstacles_polygons']]
    }}

    constants.setdefault('statebounds', None)
    constants.setdefault('inputbounds', None)
    constants.setdefault('numinputs', 0)
    constants.setdefault('uselogbar', False)
    constants.setdefault('usesigma', True)
    return constants


def run_problem(constants):
    """Returns the control points for the optimized variables"""

    constants = processconstants(constants)  # preserves the original constants dict

    xin = constants.get('init_guess', randinitguess)(constants)
    opts = {'disp': True, 'maxiter': 1000}
    xuc = []
    res = []
    singtimes = []
    for i in range(constants['Nv']):
        constants2 = constants.copy()
        constants2['xi'] = constants['xi'][i, :].reshape((1, -1))
        constants2['xf'] = constants['xf'][i, :].reshape((1, -1))
        constants2['Nv'] = 1
        constants2['obstacles'] = []
        cons2 = {'type': 'eq', 'fun': lambda x: eqconstr(x, constants2)} if not constants['uselogbar'] else []
        t = time()
        if constants['uselogbar']:
            print('Doing alg for vehicle: ' + str(i))
            res = minimize(costfun, xin[:, i], args=constants2, method='Nelder-Mead', options=opts)
            xuc.append(res.x)
        else:
            res = minimize(costfun, xin[:, i], args=constants2, method='SLSQP', options=opts, constraints=cons2)
            xuc.append(res.x)
        singtimes.append(time()-t)
        print('Elapsed time for vehicle '+str(i) + ': ' + str(singtimes[-1]) + ' s.')

    xuc = np.concatenate(xuc)

    if not constants['obstacles'] and constants['Nv'] == 1:
        return res, singtimes[0], singtimes

    cons = (
        {'type': 'eq', 'fun': lambda x: eqconstr(x, constants)},
        {'type': 'ineq', 'fun': lambda x: ineqconstr(x, constants)}
    ) if not constants['uselogbar'] else []
    bnds = variablebounds(constants) if constants['uselogbar'] else None
    t = time()
    if constants['uselogbar']:
        res = minimize(costfun, xuc, args=constants, options=opts)
    else:
        # noinspection PyTypeChecker
        res = minimize(costfun, xuc, args=constants, method='SLSQP', bounds=bnds, constraints=cons, options=opts)
    elapsedtime = time()-t
    print('Elapsed time for joined up problem: ' + str(elapsedtime) + ' s.')
    return res, elapsedtime, singtimes


def plot_xy(res, constants):
    """Plots the variables"""
    constants = processconstants(constants)
    x = res.x
    _, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    x = matrify(x, constants)
    for i in range(constants['Nv']):
        curveplot, pointsplot = bern.plot(np.fliplr(x[:, :2, i]), constants['T'], ax=ax)
        curveplot.set_label('Bernstein Polynomial for vehicle ' + str(i))
        t, xy = constants['recoverxy'](x[:, :, i], constants)
        recoveredplot, = ax.plot(xy[1, :], xy[0, :].T)
        recoveredplot.set_label('ODE solution for vehicle ' + str(i))
        ax.legend(loc='upper right', fontsize='x-small')
        points = bern.eval(x[:, :, i], constants['T'], np.linspace(0, constants['T'], 10))
        for ti in range(10):
            ax.add_patch(plotboat(points[ti, 1], points[ti, 0], np.pi/2-points[ti, 2], 0.5))
    for obs in constants['obstacles']:
        obs.plot(plotinverted=True, ax=ax)
    plt.show()


def veh_coll_avoid(x1, x2, constants):
    """Calculates """
    return np.min(np.sqrt(np.sum((constants['ElevMat']@(x1-x2))**2, axis=1))).flatten()-constants['min_dist_int_veh']
    # return np.min(np.sum((constants['ElevMat']@(x1-x2))**2, axis=1)).flatten()-constants['min_dist_int_veh']**2
    # return np.sqrt(np.min(np.sum((constants['ElevMat']@(x1-x2))**2, axis=1))).flatten()-constants['min_dist_int_veh']
    # return np.sqrt(np.sum((constants['ElevMat'] @ (x1-x2))**2, axis=1)).flatten() - constants['min_dist_int_veh']


def plotboat(x, y, yaw, size):
    points = np.array([
        [size / 2 * np.cos(yaw + np.pi - np.pi / 6), size / 2 * np.sin(yaw + np.pi - np.pi / 6)],
        [size / 2 * np.cos(yaw + np.pi / 6), size / 2 * np.sin(yaw + np.pi / 6)],
        [size / 1.5 * np.cos(yaw), size / 1.5 * np.sin(yaw)],
        [size / 2 * np.cos(yaw - np.pi / 6), size / 2 * np.sin(yaw - np.pi / 6)],
        [size / 2 * np.cos(yaw - np.pi + np.pi / 6), size / 2 * np.sin(yaw - np.pi + np.pi / 6)],
    ])
    points = points + np.array([[x, y]])
    return Polygon(points, facecolor='.9', edgecolor='.5')


class TOLCircle:
    def __init__(self, centre, rad, elevmat, mindist):
        self.centre = centre
        self.rad = rad
        self.elevmat = elevmat
        self.mindist = mindist

    def avoid(self, poly):
        return np.sqrt(np.min(np.sum((self.elevmat@(poly-self.centre))**2, axis=1))).flatten()-self.rad-self.mindist
        # return np.sqrt(np.min(np.sum(self.elevmat@(poly-self.centre)**2, axis=1))).flatten()-self.rad-self.mindist

    def plot(self, plotinverted=False, ax=None):
        x = self.centre[1*plotinverted] + self.rad * np.cos(np.linspace(0, 2*np.pi, 100))
        y = self.centre[1*(not plotinverted)] + self.rad * np.sin(np.linspace(0, 2*np.pi, 100))
        if ax is None:
            plt.plot(x, y)
        else:
            ax.plot(x, y)


class TOLPolygon:
    def __init__(self, matrix):
        self.matrix = matrix

    def avoid(self, poly):
        pass

    def obs(self):
        pass
