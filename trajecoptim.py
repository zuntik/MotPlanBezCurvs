from scipy.optimize import minimize  # , Bounds
import bernsteinlib as bern
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import time


def logbarrierfunc(delta, z, use_sigma):
    """Performs a log barrier functional on z"""
    if use_sigma:
        z = np.where(z >= 0, np.tanh(z), z)
    k = 2
    return np.where(z > delta, -np.log(np.abs(z)),
                    ((k - 1) / k) * (((z - k * delta) / ((k - 1) * delta)) ** k - 1) - np.log(delta))


def matrify_simpler(x, problem):
    """Transforms a flattened vector of control points to a matrix"""
    return x[:-1 if problem['T'] == 0 else None].\
        reshape((problem['N']+1, problem['num_states']+problem['num_inputs'], problem['Nv'])),\
        x[-1] if problem['T'] == 0 else problem['T']


def matrify(x, problem):
    """Transforms a flattened vector of control points to a matrix"""
    t_final = x[-1] if problem['T'] == 0 else problem['T']
    x = x[:-1 if problem['T'] == 0 else None]

    x = x.reshape((problem['Nv'], -1))
    x_mat = [
        np.concatenate((
            np.concatenate((
                problem['xi'][i, :].reshape((1, -1)),
                x[i, :(problem['N'] - 1) * problem['num_states']].reshape((-1, problem['num_states'])),
                problem['xf'][i, :].reshape((1, -1))
            ), axis=0),
            x[i, (problem['N'] - 1) * problem['num_states']:].reshape((problem['N'] + 1, problem['num_inputs']))
        ), axis=1)[:, :, np.newaxis]  # .reshape((problem['N']+1, problem['num_states']+problem['num_inputs'], 1))
        for i in range(problem['Nv'])]
    # return np.reshape(x, (problem['N'] - 1, problem['num_states'], problem['Nv']))
    return np.concatenate(x_mat, axis=2), t_final


def flatify(x, problem):
    return np.concatenate([np.concatenate(
        x[1:-1, :problem['num_states'], i].flatten(),
        x[:, problem['num_states']:, i].flatten()
    ) for i in range(problem['Nv'])])


def cost_fun(x, problem):
    """Calculates the cost functional"""
    j = 0
    if problem['use_log_bar']:
        c = ineqconstr(x, problem)
        j += np.sum(logbarrierfunc(0.1, c, problem['use_sigma']))

    x, t_final = matrify(x, problem)
    j += np.sum([problem['cost_fun_single'](x[:, :, i], t_final, problem) for i in range(problem['Nv'])])
    return j


def eqconstr(x, problem):
    """Deals with the equality constraints"""
    x, t_final = matrify(x, problem)
    return np.concatenate([problem['dynamics'](x[:, :, i], t_final, problem) for i in range(problem['Nv'])])
    #    # initial and final conditions
    #    constraints = [
    #        (problem['xi'] - x[0, :, :].T).flatten(),
    #        (problem['xf'] - x[-1, :, :].T).flatten()
    #    ]
    #    # dynamics
    #    constraints += [problem['dynamics'](x[:, :, i], problem) for i in range(problem['Nv'])]
    #    return np.concatenate(constraints)


def variable_bounds_simpler(problem):
    all_bounds = \
        (problem['state_bounds'] if problem['state_bounds'] is not None else
            [(-np.inf, np.inf)]*problem['num_states']) + \
        (problem['state_bounds'] if problem['state_bounds'] is not None else
            [(-np.inf, np.inf)] * problem['num_inputs'])
    bounds = []
    for i in range(problem['N']+1):
        for j in range(problem['num_states']+problem['num_inputs']):
            for k in range(problem['Nv']):
                bounds.append(all_bounds[j] if all_bounds[j] is not None else (-np.inf, np.inf))
    bounds += ([(.01, np.inf)] if problem['T'] == 0 else [])
    return np.array(bounds)[:, 0], np.array(bounds)[:, 1]


def variable_bounds_more_or_less(problem):

    state_bounds = np.array([b if b is not None else (-np.inf, np.inf) for b in problem['state_bounds']]) if\
        problem['state_bounds'] is not None else np.array([(-np.inf, np.inf)] * problem['num_states'])
    input_bounds = np.array([b if b is not None else (-np.inf, np.inf) for b in problem['input_bounds']]) if\
        problem['input_bounds'] is not None else np.array([(-np.inf, np.inf)] * problem['num_inputs'])

    bounds = np.concatenate((state_bounds, input_bounds.reshape((-1, 2))), axis=0)

    lower_bounds = np.repeat(bounds[:, 0].reshape((1, -1)), problem['N']+1, axis=0)
    upper_bounds = np.repeat(bounds[:, 1].reshape((1, -1)), problem['N']+1, axis=0)

    lower_bounds = np.repeat(lower_bounds[:, :, np.newaxis], problem['Nv'], axis=2)
    upper_bounds = np.repeat(upper_bounds[:, :, np.newaxis], problem['Nv'], axis=2)

    #    lower_bounds[0, :problem['num_states'], :] = problem['xi'].T
    #    lower_bounds[-1, :problem['num_states'], :] = problem['xf'].T
    #    upper_bounds[0, :problem['num_states'], :] = problem['xi'].T
    #    upper_bounds[-1, :problem['num_states'], :] = problem['xf'].T

    lower_bounds = np.concatenate((lower_bounds.flatten(), ([0.01] if problem['T'] == 0 else [])))
    upper_bounds = np.concatenate((upper_bounds.flatten(), ([np.inf] if problem['T'] == 0 else [])))

    return lower_bounds, upper_bounds

    # return [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]\
    # + ([(0, np.inf)] if problem['T'] == 0 else [])


def first_and_last(x, problem):
    x, _ = matrify(x, problem)
    return np.concatenate((
        (x[0, :problem['num_states'], :] - problem['xi'].T).flatten(),
        (x[-1, :problem['num_states'], :] - problem['xf'].T).flatten(),
    ))


def variable_bounds(problem):
    """Creates a vector of lower and upper bounds"""
    return ([
                # (problem['state_bounds'][0, var], problem['state_bounds'][1, var])
                problem['state_bounds'][var] if problem['state_bounds'][var] is not None else (-np.inf, np.inf)
                for _ in range(problem['N'] - 1)
                for var in range(problem['num_states'])
            ] + [
                # (problem['input_bounds'][1, inp], problem['input_bounds'][1, inp])
                problem['input_bounds'][inp] if problem['input_bounds'][inp] is not None else (-np.inf, np.inf)
                for _ in range(problem['N'] + 1)
                for inp in range(problem['num_inputs'])
            ]) * problem['Nv'] + ([(0, np.inf)] if problem['T'] == 0 else []) \
        if problem['state_bounds'] is not None else None

    # return [(problem['vehicle_bounds'][0, var], problem['vehicle_bounds'][1, var])
    #         for _ in range(problem['N'] + 1)
    #         for var in range(problem['num_states'])
    #         for __ in range(problem['Nv'])] if problem['vehicle_bounds'] is not None else None


def ineqconstr(x, problem):
    """ Deals with nonlinear inequality constraints"""
    x, t_final = matrify(x, problem)
    c = []

    # inter vehicles
    c += [veh_coll_avoid(x[:, :2, v1], x[:, :2, v2], problem)
          for v1 in range(problem['Nv']) for v2 in range(v1 + 1, problem['Nv'])]

    # obstacles
    c += [obs.avoid(x[:, :2, veh]) for obs in problem['obstacles'] for veh in range(problem['Nv'])]
    return np.concatenate(c) if c else np.array([])


def rand_init_guess_simpler(problem):
    lower_bounds, upper_bounds = variable_bounds(problem)
    bounds = np.concatenate((lower_bounds.reshape((-1, 1)), upper_bounds.reshape((-1, 1))), axis=1)
    bounds = np.where(bounds == np.inf, 1, bounds)
    bounds = np.where(bounds == -np.inf, -1, bounds)
    return np.random.uniform(bounds[:, 0], bounds[:, 1])


def lin_init_guess(problem):
    """Calculates an initial guess based on linspace() for the state variables"""
    xin = []
    for i in range(problem['Nv']):
        xin.append(np.linspace(problem['xi'][i, :], problem['xf'][i, :], problem['N'] + 1)[1:-1, :].flatten())
        xin.append(np.random.rand((problem['N']+1)*problem['num_inputs']))
    # now find initial guess for time if it is also a variable to optimize:
    if problem['T'] == 0:
        xin.append(np.array([np.sqrt(np.sum((problem['xi'][0, :2]-problem['xf'][-1, :2])**2))]))

    return np.concatenate(xin)


def rand_init_guess(problem):
    """Calculates a random initial guess"""
    bnds = variable_bounds(problem)
    # lower_bounds, upper_bounds = variable_bounds(problem)
    # bnds = np.concatenate((lower_bounds.reshape((-1, 1)), upper_bounds.reshape((-1, 1))), axis=1)
    if bnds is None:
        _n = problem['N']
        return np.random.rand(((_n - 1) * problem['num_states'] + (_n+1) * problem['num_inputs']) * problem['Nv'])
    else:
        bnds = np.array(bnds)
        bnds = np.where(bnds == np.inf, 1, bnds)
        bnds = np.where(bnds == -np.inf, -1, bnds)
        # return np.array([np.random.uniform(bnds[i, 0], bnds[i, 1]) for i in range(bnds.shape[0])])
        return np.random.uniform(bnds[:, 0], bnds[:, 1])


def process_problem(problem_orig):
    """Returns a new dictionary with more fields that the rest of the functions need"""
    problem = problem_orig.copy()
    problem.setdefault('state_bounds', None)
    problem.setdefault('input_bounds', None)
    problem.setdefault('num_inputs', 0)
    problem.setdefault('use_log_bar', False)
    problem.setdefault('use_sigma', True)
    problem.setdefault('T', 0)
    problem.setdefault('N', 20)
    problem |= {
        # common parameters
        # 'DiffMat': bern.derivelevmat(problem['N'], problem['T'] if problem['T'] != 0 else 1),
        'DiffMat': bern.derivelevmat(problem['N'], 1),
        'elev_mat': bern.degrelevmat(problem['N'], problem['N'] * 10),
        # 'EvalMat': bern.evalmat(problem['N'], problem['T'], np.linspace(0,problem['T'], problem['N']*40)),
        'EvalMat': bern.evalmat(problem['N'], problem['T'] if problem['T'] != 0 else 1,
                                np.linspace(0, problem['T'] if problem['T'] != 0 else 1, 1000)),
        'num_states': problem['xi'].shape[1],
        'Nv': problem['xi'].shape[0],
    }
    problem.setdefault('obstacles_circles', [])
    problem.setdefault('obstacles_polygons', [])
    problem.setdefault('min_dist_obs', 0)
    problem.setdefault('min_dist_int_veh', 2)
    # noinspection PyTypeChecker
    problem |= {
        'obstacles':
            [TOLCircle(c[:-1], c[-1], problem['elev_mat'], problem['min_dist_obs'])
             for c in problem['obstacles_circles']] +
            [TOLPolygon(m) for m in problem['obstacles_polygons']]
    }
    problem.setdefault('plot_control_points', False)
    problem.setdefault('recover_xy', None)
    problem.setdefault('boat_size', np.linalg.norm(problem['xi'][:2, 0]-problem['xf'][:2, 0])/13)

    return problem


def plot_xy(x, t_final, problem):
    """Plots the variables"""
    problem = process_problem(problem)
    _, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    for i in range(problem['Nv']):
        curve_plot, _ = bern.plot(np.fliplr(x[:, :2, i]), t_final, plotcpts=problem['plot_control_points'], ax=ax)
        curve_plot.set_label('Bernstein Polynomial for vehicle ' + str(i))
        if problem['recover_xy'] is not None:
            _, xy = problem['recover_xy'](x[:, :, i], t_final, problem)
            recovered_plot, = ax.plot(xy[1, :], xy[0, :].T)
            recovered_plot.set_label('ODE solution for vehicle ' + str(i))
        ax.legend(loc='upper right', fontsize='x-small')
        points = bern.eval(x[:, :, i], t_final, np.linspace(0, t_final, 10))
        for ti in range(10):
            ax.add_patch(plot_boat(points[ti, 1], points[ti, 0], np.pi / 2 - points[ti, 2], problem['boat_size']))
    for obs in problem['obstacles']:
        obs.plot(plot_inverted=True, ax=ax)
    plt.show()


def veh_coll_avoid(x1, x2, problem):
    """Calculates """
    return np.min(np.sqrt(np.sum((problem['elev_mat'] @ (x1 - x2)) ** 2, axis=1))).flatten() - problem[
        'min_dist_int_veh']
    # return np.min(np.sum((problem['elev_mat']@(x1-x2))**2, axis=1)).flatten()-problem['min_dist_int_veh']**2
    # return np.sqrt(np.min(np.sum((problem['elev_mat']@(x1-x2))**2, axis=1))).flatten()-problem['min_dist_int_veh']
    # return np.sqrt(np.sum((problem['elev_mat'] @ (x1-x2))**2, axis=1)).flatten() - problem['min_dist_int_veh']


def plot_boat(x, y, yaw, size):
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
    def __init__(self, centre, rad, elev_mat, min_dist):
        self.centre = centre
        self.rad = rad
        self.elev_mat = elev_mat
        self.min_dist = min_dist

    def avoid(self, poly):
        return np.sqrt(
            np.min(np.sum((self.elev_mat @ (poly - self.centre)) ** 2, axis=1))).flatten() - self.rad - self.min_dist
        # return np.sqrt(np.min(np.sum(self.elev_mat@(poly-self.centre)**2, axis=1))).flatten()-self.rad-self.min_dist

    def plot(self, plot_inverted=False, ax=None):
        x = self.centre[1 * plot_inverted] + self.rad * np.cos(np.linspace(0, 2 * np.pi, 100))
        y = self.centre[1 * (not plot_inverted)] + self.rad * np.sin(np.linspace(0, 2 * np.pi, 100))
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


def run_problem(problem):
    """Returns the control points for the optimized variables"""

    problem = process_problem(problem)  # preserves the original problem dict

    xin = problem.get('init_guess', lin_init_guess)(problem)
    # xin = np.concatenate((xin, np.array([1]))) if problem['T'] == 0 else xin

    opts = {'disp': True, 'maxiter': 1000}

    constr = []
    constr += [{'type': 'eq', 'fun': lambda x: eqconstr(x, problem)}]
    if not problem['use_log_bar']:
        constr += [{'type': 'ineq', 'fun': lambda x: ineqconstr(x, problem)}]

    # constr += [{'type': 'eq', 'fun': lambda x: first_and_last(x, problem)}]

    #    lower_bounds, upper_bounds = variable_bounds(problem)
    #    bnds = Bounds(lower_bounds, upper_bounds) if not problem['use_log_bar'] else None
    bnds = variable_bounds(problem) if not problem['use_log_bar'] else None

    t = time()
    # noinspection PyTypeChecker
    res = minimize(cost_fun, xin, args=problem, method='SLSQP', bounds=bnds, constraints=constr, options=opts)
    elapsed_time = time() - t
    x_out, t_final = matrify(res.x, problem)
    return x_out, t_final, res.fun, elapsed_time


def planner(xi, xf, **keyword_args):

    problem = {
        'xi': xi,  # initial states
        'xf': xf,  # final states
    }

    problem |= keyword_args

    x_out, t_final = run_problem(problem)[:2]

    # evaluators = [
    #     [lambda t: bern.eval(x_out[:, j, i], t_final, t) for j in range(xi.shape[1])]
    #     for i in range(xi.shape[0])]
    evaluators = [lambda t: bern.eval(x_out[:, :, i], t_final, t) for i in range(xi.shape[0])]

    return evaluators, t_final
