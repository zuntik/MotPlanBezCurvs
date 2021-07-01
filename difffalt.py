from scipy.integrate import solve_ivp
from . import bernsteinlib as bern
import numpy as np
from scipy.optimize import minimize  # , Bounds
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import time


def main():

    # first illustrative example
    offset = np.array([1234, 185943, 0, 0, 0])
    problem = {
        # 'T': 100,  # runtime
        'xi': np.array([[-5, 0, 0, 9, 0]])+ offset,  # initial states
        'xf': np.array([[15, 10, 0, 0, 0]])+offset,  # final states
        'N': 50,  # order of the polynomials
        'obstacles_circles': [[5, 0, 3], [6,6,4]],  # n lines for n circles where columns are position x, position y, radius
        #'obstacles_circles': [[5+offset[0], 0+offset[1], 3]],  # n lines for n circles where columns are position x, position y, radius
        'state_bounds': [None, None, None, (-.01, 1), (-2, 2)],
        'dim': 2
    }

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics5vars,
        'recover_xy': recover_xy,
    }}

    x_out, t_final, cost_final, elapsed_time = run_problem(problem)
    print('The final cost is ' + str(cost_final) + ' and the computation time was ' + str(elapsed_time))
    plot_xy(x_out, t_final, problem)

    # print('Now do the same thing but with the planner function...')
    # x_out_planners, t_final_planner = planner(**problem)
    # vehicle1_plot = x_out_planners[0](np.linspace(0, t_final_planner, 1000))
    # import matplotlib.pyplot as plt
    # plt.plot(vehicle1_plot[:, 1], vehicle1_plot[:, 0])  # axis are flipped
    # plt.show()


################################################################################
# functions
################################################################################
def recover_xy(x, t_final, _):
    def ode_func(t, val, v, w):
        _, _, psi = val
        dx = v(t) * np.cos(psi)
        dy = v(t) * np.sin(psi)
        dpsi = w(t)
        return np.array([dx, dy, dpsi])

    def pol_v(t): return bern.eval(x[:, 3], t_final, t)

    def pol_w(t): return bern.eval(x[:, 4], t_final, t)

    sol = solve_ivp(ode_func, [0, t_final], x[0, :3], args=(pol_v, pol_w), dense_output=True, vectorized=True)

    return np.linspace(0, t_final, 1000), sol.sol(np.linspace(0, t_final, 1000))


def dynamics5vars(x, t_final, problem):
    """the vehicle dynamics"""
    diff_mat = problem['DiffMat'] / t_final
    xp, yp, psi, v, w = x.T
    return np.vstack((
        diff_mat @ xp - v * np.cos(psi),
        diff_mat @ yp - v * np.sin(psi),
        diff_mat @ psi - w,
    )).flatten()


def cost_fun_single(x, t_final, problem):
    """the running cost for a singular vehicle"""
    if problem['T'] == 0:
        return t_final
    v = x[:, 3]
    w = x[:, 4]
    a = (problem['DiffMat'] / t_final) @ v
    # return np.sum((problem['elev_mat']@a)**2)+2*np.sum((problem['ElevMat']@w)**2)
    return np.sum(a ** 2) + 2 * np.sum(w ** 2)


def dubinscar_planner(xi, xf, **problem):

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics5vars,
        'recover_xy': recover_xy,
    }}

    return planner(xi, xf, **problem)

################################################################################
# trajectoptimpart
################################################################################


def logbarrierfunc(delta, z, use_sigma):
    """Performs a log barrier functional on z"""
    if use_sigma:
        z = np.where(z >= 0, np.tanh(z), z)
    k = 2
    return np.where(z > delta, -np.log(np.abs(z)),
                    ((k - 1) / k) * (((z - k * delta) / ((k - 1) * delta)) ** k - 1) - np.log(delta))


def matrify(x, problem):
    """Transforms a flattened vector of control points to a matrix"""
    t_final = x[-1] if problem['T'] == 0 else problem['T']
    x = x[:-1 if problem['T'] == 0 else None].rehape(problem['N']+1, problem['dim'], problem['Nv'])
    return x, t_final


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


def lineqconstr(problem):
    A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
    Aeq = []
    beq = []
    for i in range(problem['N_v']):
        # deal with first and last control point
        for j in range(problem['dim']):
            A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line[0, j, i] = 1
            Aeq.append(A_line.reshape((1,-1)))
            beq.append(problem['xi'][i,0])
            A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line[-1, j, i] = 1
            Aeq.append(A_line.reshape((1, -1)))
            beq.append(problem['xf'][i,0])
        # deal with initial and final headings
        # find some perpendicular vectors
        if problem['dim'] == 2:
            vec_i = np.array([[-np.sin(problem['xi'][i, 2]), np.cos(problem['xi'][i, 2])]])
            vec_f = np.array([[-np.sin(problem['xf'][i, 2]), np.cos(problem['xf'][i, 2])]])
        elif problem['dim'] == 3:
            alignedvec_i = np.array([np.cos(problem['azimuth_i'])*np.cos(problem['elevation_i']), np.sin(problem['azimuth_i'])*np.cos(problem['elevation_i']), np.sin(problem['elevation_i'])])
            vec_i = np.cross(alignedvec_i, np.array([1,0,0])
            alignedvec_f = np.array([np.cos(problem['azimuth_f'])*np.cos(problem['elevation_f']), np.sin(problem['azimuth_f'])*np.cos(problem['elevation_f']), np.sin(problem['elevation_f'])])
            vec_f = np.cross(alignedvec_f, np.array([1,0,0])
            if np.linalg.norm(vec_i) == 0:
                vec_r = np.cross(alignedvec_i, np.array([0, 1, 0]))
            if np.linalg.norm(vec_f) == 0:
                vec_f = np.cross(alignedvec_f, np.array([0, 1, 0]))
            vec_i = np.concatecante(vec_i.reshape((1,-1)), np.cross(alignedvec_i, np.array([0, 0, 1])).reshape((1, -1), axis=0)
            vec_f = np.concatecante(vec_f.reshape((1,-1)), np.cross(alignedvec_f, np.array([0, 0, 1])).reshape((1, -1), axis=0)
        # for each of the perpendicular vectors (which can be 2 or 1), define a line on the matrix
        for j in range(vec_i.shape[0]):
        n = m            A_line_i = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line_f = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            for k in range(problem['dim']):
                A_line_i[0, k, i] = -vec[j, k]
                A_line_f[-1, k, i] = vec[j, k]
                if np.linealg.norm(problem['xi'][i,:problem['dim']) == 0:
                    A_line[2, k, i] = vec[j, k]
                else:
                    A_line[1, k, i] = vec[j, k]
                if np.linealg.norm(problem['xf'][i,:problem['dim']) == 0:
                    A_line_i[-3, k, i] = -vec[j, k]
                else:
                    A_line_f[-2, k, i] = -vec[j, k]
            Aeq.append(A_line_i.reshape((1,-1)))
            beq.append(0)
            Aeq.append(A_line_f.reshape((1,-1)))
            beq.append(0)

    return np.concatenate(Aeq, axis=0), np.array(beq)


def linineqconstr(problem):
    A = []
    b = []
    # deal with initial headings
    for i in range(problem['N_v']):
        if problem['dim'] == 2:
            alignedvec_i = np.array([[np.cos(problem['xi'][i, 2]), np.sin(problem['xi'][i, 2])]])
            alignedvec_f = np.array([[np.cos(problem['xf'][i, 2]), np.sin(problem['xf'][i, 2])]])
        elif problem['dim'] == 3:
            alignedvec_i = np.array([[np.cos(problem['azimuth_i'])*np.cos(problem['elevation_i']), np.sin(problem['azimuth_i'])*np.cos(problem['elevation_i']), np.sin(problem['elevation_i'])]])
            alignedvec_f = np.array([[np.cos(problem['azimuth_f'])*np.cos(problem['elevation_f']), np.sin(problem['azimuth_f'])*np.cos(problem['elevation_f']), np.sin(problem['elevation_f'])]])

        for j in range(vec.shape[0]):
            A_line_i = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line_f = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            for k in range(problem['dim']):
                A_line_i[0, k, i] = -alignedvec_i[j, k]
                A_line_f[-1, k, i] = alignedvec_f[j, k]
                if np.linealg.norm(problem['xf'][i,:problem['dim']) == 0:
                    A_line_i[2, k, i] = vec[j, k]
                else:
                    A_line_i[1, k, i] = vec[j, k]
                if np.linealg.norm(problem['xf'][i,:problem['dim']) == 0:
                    A_line_f[-3, k, i] = -vec[j, k]
                else:
                    A_line_f[-2, k, i] = -vec[j, k]
            A.append(A_line_i.reshape((1,-1)))
            b.append(0)
            A.append(A_line_f.reshape((1,-1)))
            b.append(0)


def ineqconstr(x, problem):
    """ Deals with nonlinear inequality constraints"""
    x, t_final = matrify(x, problem)
    c = []

    # speeds
    dx = problem['derivmat'] @ x
    ddx = problem['derivmat'] @ dx

    dx_s = evalmat @ dx
    ddx_s = evalmat @ ddx

    # maximum speed
    v = np.linalg.norm(dx_s, axis=1)
    c.append(problem['v_max'] - v)

    # maximum yaw rate
    c.append(bern.degrelev( bern.sum(problem['r_max'] * (bern.pow(dx, 2) + bern.pow(dy, 2)), - bern.mul(dx, ddy) + bern.mul(ddx, dy)), problem['N']*10 ).flatten())
    c.append(bern.degrelev( bern.sum(problem['r_max'] * (bern.pow(dx, 2) + bern.pow(dy, 2)), - bern.mul(dx, ddy) + bern.mul(ddx, dy)), problem['N']*10 ).flatten())

    # inter vehicles
    c += [veh_coll_avoid(x[:, :2, v1], x[:, :problem['dim'], v2], problem)
          for v1 in range(problem['Nv']) for v2 in range(v1 + 1, problem['Nv'])]

    # obstacles
    c += [obs.avoid(x[:, :2, veh]) for obs in problem['obstacles'] for veh in range(problem['Nv'])]
    return np.concatenate(c) if c else np.array([])


def rand_init_guess(problem):
    """Calculates a random initial guess"""
    return np.random.rand(problem['N']+1, 2 if problem['dim'], problem['Nv']


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
    problem = {**problem, **{
        # common parameters
        'DiffMat': bern.derivelevmat(problem['N'], 1),
        'elev_mat': bern.degrelevmat(problem['N'], problem['N'] * 10),
        'EvalMat': bern.evalmat(problem['N'], problem['T'] if problem['T'] != 0 else 1,
                                np.linspace(0, problem['T'] if problem['T'] != 0 else 1, 1000)),
        'num_states': problem['xi'].shape[1],
        'Nv': problem['xi'].shape[0],
    }}
    problem.setdefault('obstacles_circles', [])
    problem.setdefault('obstacles_polygons', [])
    problem.setdefault('min_dist_obs', 0)
    problem.setdefault('min_dist_int_veh', 2)
    # noinspection PyTypeChecker
    problem = {**problem, **{
        'obstacles':
            [TOLCircle(c[:-1], c[-1], problem['elev_mat'], problem['min_dist_obs'])
             for c in problem['obstacles_circles']] +
            [TOLPolygon(m) for m in problem['obstacles_polygons']]
    }}
    problem.setdefault('plot_control_points', False)
    problem.setdefault('recover_xy', None)
    problem.setdefault('boat_size', np.linalg.norm(problem['xi'][:2, 0]-problem['xf'][:2, 0])/13)
    problem.setdefault('dim', 2)

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
    return np.min(np.sqrt(np.sum((problem['elev_mat'] @ (x1 - x2)) ** 2, axis=1))).flatten() - problem['min_dist_int_veh']
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

    xin = problem.get('init_guess', rand_init_guess)(problem)

    algorithm = {
        'method': 'SLSQP',
        'options': {'disp': True, 'ftol': 1e-02, 'maxiter': 1000}
    }
    algorithm = {'method':'trust-constr','options':None}

    constr = []

    Aeq, beq = lineqconstr(problem)
    constr+= [{'type': 'eq', 'fun', lambda x: A@x.reshape((-1,1))-b, 'jac', lambda x: A.ravel() }]

    A, b = linineqconstr(problem)

    constr += [{'type': 'eq', 'fun': lambda x: eqconstr(x, problem)}]
    if not problem['use_log_bar']:
        constr += [{'type': 'ineq', 'fun': lambda x: ineqconstr(x, problem)}]

    t = time()
    # noinspection PyTypeChecker
    res = minimize(cost_fun, xin, args=problem, method=algorithm['method'], constraints=constr, options=algorithm['options'])
    elapsed_time = time() - t
    x_out, t_final = matrify(res.x, problem)
    return x_out, t_final, res.fun, elapsed_time


def planner(xi, xf, **keyword_args):

    problem = {
        'xi': xi,  # initial states
        'xf': xf,  # final states
    }

    problem = {**problem, **keyword_args}

    x_out, t_final = run_problem(problem)[:2]

    evaluators = [lambda t: bern.eval(x_out[:, :, i], t_final, t) for i in range(xi.shape[0])]

    return evaluators, t_final


if __name__ == "__main__":
    main()
