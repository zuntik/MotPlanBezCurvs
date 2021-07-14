from . import bernsteinlib as bern
import numpy as np
from scipy.optimize import minimize  # , Bounds
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from time import time

def main():

    # first illustrative example
    #    problem = {
    #        'T': 10,  # runtime
    #        'xi': np.array([[0, 0]]),  # initial states
    #        'xf': np.array([[5, 5]]),  # final states
    #        'vi': np.array([1]), # initial speeds
    #        'vf': np.array([1]), # final speeds
    #        'hi': np.array([0]), # initial heading
    #        'hf': np.array([np.pi/2]), # final heading
    #        'N': 10,  # order of the polynomials
    #        # 'obstacles_circles': [[5, 0, 3], [6,6,4]],  # n lines for n circles where columns are position x, position y, radius
    #        'v_max': 1.1,
    #        'r_max': 5,
    #        'a_max': None,
    #        'dr_max': 1,
    #    }

    #    problem = {
    #        # 'T': 15,
    #        'xi': np.array([[0, 5], [5, 0]]),
    #        'xf': np.array([[10, 5], [5, 10]]),
    #        'vi': np.array([1, 1]),
    #        'vf': np.array([1, 1]),
    #        'hi': np.array([0, np.pi/2]),
    #        'hf': np.array([0, np.pi/2]),
    #        'N': 5,
    #        'v_max': 1.1,
    #        'r_max': 5,
    #        'a_max': None,
    #        'dr_max': 1,
    #    }

    problem = {
        'N': 20,
        'T': 20,
        'xi': np.array([
            [-10, 4 ],
            [-10, -4],
            [-10, 0 ],
            [0, -10 ],
        ]),
        'xf': np.array([
            [10, -3],
            [10, 3 ],
            [10, 0 ],
            [0, 10 ],
        ]),
        #'obstacles_circles': [[0, 0, 3]],
        'min_dist_int_veh': 1,
        'vi': np.array([1,1,1,1]),
        'vf': np.array([1,1,1,1]),
        'hi': np.array([0,0,0,0]),
        'hf': np.array([0,0,0,0]),
        'N': 10,
        'v_max': 1.1,
        'r_max': 5,
        'a_max': None,
        'dr_max': 1,
    }

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        # 'recover_xy': recover_xy,
        'plot_boats': False,
        #'plot_control_points': True,
    }}

    x_out, t_final, cost_final, elapsed_time = run_problem(problem)
    print(x_out[ 0,:,:] - problem['xi'].T)
    print(x_out[-1,:,:] - problem['xf'].T)
    print('The final cost is ' + str(cost_final) + ' and the computation time was ' + str(elapsed_time))
    plot_xy(x_out, t_final, problem)
    plt.figure()
    #plt.plot(np.linspace(0,t_final,100),np.sqrt(bern.eval(np.sum(bern.pow(bern.deriv(x_out[:,:2,0],t_final)),axis=1), t_final, tuple(np.linspace(0,t_final,100).tolist()))))
    plt.plot(np.linspace(0,t_final,100),np.sqrt(bern.evalspace(np.sum(bern.pow(bern.deriv(x_out[:,:2,0],t_final)),axis=1), t_final, (0,t_final,100))))
    plt.show()

    # print('Now do the same thing but with the planner function...')
    # x_out_planners, t_final_planner = planner(**problem)
    # vehicle1_plot = x_out_planners[0](np.linspace(0, t_final_planner, 1000))
    # import matplotlib.pyplot as plt
    # plt.plot(vehicle1_plot[:, 1], vehicle1_plot[:, 0])  # axis are flipped
    # plt.show()


################################################################################
# functions
################################################################################
def cost_fun_single(x, t_final, problem):
    """the running cost for a singular vehicle"""
    #return np.sum(bern.pow(bern.deriv(x, t_final))) + t_final
    dx = x[1:,:]-x[:-1,:]
    ddx = dx[1:,:]-dx[:-1,:]
    ddx = 0
    # https://stackoverflow.com/a/47443343
    return 1000*np.sqrt(np.sum((dx)**2)) + 1000*np.sqrt(np.sum((ddx)**2))+ t_final
    #return 100* (np.sum(np.linalg.norm(dx,axis=1))+np.linalg.norm(ddx, axis=1)) + t_final
    #    if problem['T']==0:
    #        return t_final
    #    else:
    #        return np.sum(bern.pow(bern.deriv(x, t_final)))


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
    x = x[:-1 if problem['T'] == 0 else None].reshape(problem['N']+1, problem['dim'], problem['Nv'])
    return x, t_final


def cost_fun(x, problem):
    """Calculates the cost functional"""
    j = 0
    if problem['use_log_bar']:
        c = ineqconstr(x, problem)
        j += np.sum(logbarrierfunc(0.1, c, problem['use_sigma']))

    x, t_final = matrify(x, problem)
    j += np.sum([problem['cost_fun_single'](x[:, :, i], t_final, problem) for i in range(problem['Nv'])])
    # return j
    return j



def lineqconstr(problem):
    Aeq = []
    beq = []
    # for each vehicle i
    for i in range(problem['Nv']):

        # deal with first and last control point
        for j in range(problem['dim']):
            # first
            A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line[0, j, i] = 1
            Aeq.append(A_line.flatten())
            if problem['T']==0:
                Aeq.append(np.array([0]))
            beq.append(problem['xi'][i, j])
            # last
            A_line = np.zeros((problem['N']+1, problem['dim'], problem['Nv']))
            A_line[-1, j, i] = 1
            Aeq.append(A_line.flatten())
            if problem['T']==0:
                Aeq.append(np.array([0]))
            beq.append(problem['xf'][i, j])

        # deal with initial and final speeds (inc. if they are equal to 0)
        # deal with initial and final headings if the speed is 0
        vi = getvec(problem['vi'][i], problem['hi'][i])
        for j in range(problem['dim']):
            A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line[0, j, i] = -1
            A_line[1, j, i] = 1
            Aeq.append(A_line.flatten())
            if problem['T'] == 0:
                Aeq.append(np.array([-vi[j]/problem['N']]))
                b = 0
            else:
                b = vi[j]*problem['T']/problem['N']
            beq.append(b)

        # if speed is 0, find some perpendicular vectors
        if problem['vi'][i] == 0:
            if   problem['dim'] == 2:
                vec_i = np.array([[-np.sin(problem['hi'][i]), np.cos(problem['hi'][i])]])
            elif problem['dim'] == 3:
                alignedvec_i = getvec(1, problem['azimuth_i'], problem['elevation_i'])
                vec_i = np.cross(alignedvec_i, np.array([1,0,0]))
                if np.linalg.norm(vec_i) == 0:
                    vec_r = np.cross(alignedvec_i, np.array([0, 1, 0]))
                vec_i = np.concatecante(vec_i.reshape((1,-1)), np.cross(alignedvec_i, np.array([0, 0, 1])).reshape((1, -1)), axis=0)
            # for each of the perpendicular vectors (which can be 2 or 1), define a line on the matrix
            for j in range(vec_i.shape[0]):
                A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
                for k in range(problem['dim']):
                    A_line[0, k, i] = -vec_i[j, k]
                    if np.linalg.norm(problem['xi'][i,:problem['dim']]) == 0:
                        A_line[2, k, i] = vec_i[j, k]
                    else:
                        A_line[1, k, i] = vec_i[j, k]
                Aeq.append(A_line.flatten())
                if problem['T']==0:
                    Aeq.append(np.array([0]))
                beq.append(0)

        vf = getvec(problem['vf'][i], problem['hf'][i])
        for j in range(problem['dim']):
            A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
            A_line[-2, j, i] = -1
            A_line[-1, j, i] = 1
            Aeq.append(A_line.flatten())
            if problem['T'] == 0:
                Aeq.append([-vf[j]/problem['N']])
                b = 0
            else:
                b = vf[j]*problem['T']/problem['N']
            beq.append(b)

        # if speed is 0, find some perpendicular vectors
        if problem['vf'][i] == 0:
            if   problem['dim'] == 2:
                vec_f = np.array([[-np.sin(problem['hf'][i]), np.cos(problem['hf'][i])]])
            elif problem['dim'] == 3:
                alignedvec_f = getvec(1, problem['azimuth_f'], problem['elevation_f'])
                vec_f = np.cross(alignedvec_f, np.array([1,0,0]))
                if np.linalg.norm(vec_f) == 0:
                    vec_r = np.cross(alignedvec_f, np.array([0, 1, 0]))
                vec_f = np.concatecante(vec_f.reshape((1,-1)), np.cross(alignedvec_f, np.array([0, 0, 1])).reshape((1, -1)), axis=0)
            # for each of the perpendicular vectors (which can be 2 or 1), define a line on the matrix
            for j in range(vec_f.shape[0]):
                A_line = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
                for k in range(problem['dim']):
                    A_line[0, k, i] = -vec_f[j, k]
                    if np.linalg.norm(problem['xi'][i,:problem['dim']]) == 0:
                        A_line[2, k, i] = vec_f[j, k]
                    else:
                        A_line[1, k, i] = vec_f[j, k]
                Aeq.append(A_line.flatten())
                if problem['T']==0:
                    Aeq.append([0])
                beq.append(0)

    return np.concatenate(Aeq).reshape((len(beq), -1)), np.array(beq).reshape((-1, 1))


def linineqconstr(problem):
    A = []
    b = []
    # deal with initial and final headings if initial or final headings are 0
    for i in range(problem['Nv']):
        if problem['dim'] == 2:
            alignedvec_i = getvec(1, problem['hi'][i])
            alignedvec_f = getvec(1, problem['hf'][i])
        elif problem['dim'] == 3:
            alignedvec_i = getvec(1, problem['azimuth_i'], problem['elevation_i'])
            alignedvec_f = getvec(1, problem['azimuth_f'], problem['elevation_f'])

        A_line_i = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
        A_line_f = np.zeros((problem['N']+1,problem['dim'],problem['Nv']))
        if problem['vi'][i] == 0:
            for k in range(problem['dim']):
                A_line_i[0, k, i] = -alignedvec_i[k]
                A_line_i[2, k, i] = alignedvec_i[k]
            A.append(A_line_i.flatten())
            if problem['T']==0:
                A.append([0])
            b.append(0)
        if problem['vf'][i] == 0:
            for k in range(problem['dim']):
                A_line_f[-1, k, i] = -alignedvec_f[k]
                A_line_f[-3, k, i] = alignedvec_f[k]
            A.append(A_line_f.flatten())
            if problem['T']==0:
                A.append([0])
            b.append(0)

    if problem['T'] == 0:
        A_line = np.zeros((problem['N']+1)*problem['dim']*problem['Nv']+1)
        A_line[-1] = 1
        A.append(A_line)
        b.append(0)
    if len(A) != 0:
        return np.concatenate(A).reshape((len(b), -1)), np.array(b).reshape((-1, 1))
    else:
        return None, None


def getvec(v, azimuth, elevation=None):
    return v * np.array([np.cos(azimuth), np.sin(azimuth)] if elevation is None else [np.cos(azimuth)*np.cos(elevation), np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])


def ineqconstr(x, problem):
    """ Deals with nonlinear inequality constraints"""
    x, t_final = matrify(x, problem)
    c = []

    # derivs
    dx = bern.deriv(x, t_final)
    ddx = bern.deriv(dx, t_final)

    dxp = dx[:,0,:]
    dyp = dx[:,1,:]
    ddxp = ddx[:,0,:]
    ddyp = ddx[:,1,:]

    dx2dy2 = np.sum(bern.pow(dx[:,:2,:]), axis=1)

    # maximum speed
    c.append((problem['v_max']**2 - bern.degrelev(dx2dy2, problem['N']*10)).flatten())

    # maximum yaw rate
    dxddyddxdy = -bern.mul(dxp, ddyp) + bern.mul(ddxp, dyp)
    c.append(bern.degrelev( bern.add(problem['r_max']*dx2dy2,  dxddyddxdy), problem['N']*10 ).flatten())
    c.append(bern.degrelev( bern.add(problem['r_max']*dx2dy2, -dxddyddxdy), problem['N']*10 ).flatten())

    # maximum yaw acceleration
    dxddxdyddy = np.sum(bern.mul(dx[:, :2, :],ddx[:, :2, :]), axis=1)
    c.append(bern.degrelev(bern.add(problem['dr_max']**2 * dx2dy2, - 2 * dxddxdyddy), problem['N']*10).flatten())

    # inter vehicles
    c += [veh_coll_avoid(x[:, :problem['dim'], v1], x[:, :problem['dim'], v2], problem)
          for v1 in range(problem['Nv']) for v2 in range(v1 + 1, problem['Nv'])]

    # obstacles
    c += [obs.avoid(x[:, :2, veh]) for obs in problem['obstacles'] for veh in range(problem['Nv'])]
    return np.concatenate(c) if c else np.array([])


def rand_init_guess(problem):
    """Calculates a random initial guess"""
    return np.random.rand((problem['N']+1)*problem['dim']*problem['Nv'] + (1 if problem['T'] == 0 else 0))+1


def lin_init_guess(problem):
    if problem['T'] != 0:
        return np.linspace(problem['xi'].T, problem['xf'].T, problem['N']+1).flatten()
    else:
        return np.append(np.linspace(problem['xi'].T, problem['xf'].T, problem['N']+1).flatten(), 1)


def process_problem(problem_orig):
    """Returns a new dictionary with more fields that the rest of the functions need"""
    problem = problem_orig.copy()
    problem.setdefault('use_log_bar', False)
    problem.setdefault('use_sigma', True)
    problem.setdefault('T', 0)
    problem.setdefault('N', 20)
    problem = {**problem, **{
        # common parameters
        'DiffMat': bern.derivelevmat(problem['N'], 1),
        'Nv': problem['xi'].shape[0],
    }}
    problem.setdefault('obstacles_circles', [])
    problem.setdefault('obstacles_polygons', [])
    problem.setdefault('min_dist_obs', 0)
    problem.setdefault('min_dist_int_veh', 1)
    # noinspection PyTypeChecker
    problem = {**problem, **{
        'obstacles':
            [TOLCircle(c[:-1], c[-1], problem['min_dist_obs'])
             for c in problem['obstacles_circles']] +
            [TOLPolygon(m) for m in problem['obstacles_polygons']]
    }}
    problem.setdefault('plot_control_points', False)
    problem.setdefault('recover_xy', None)
    problem.setdefault('plot_boats', True)
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
        if problem['plot_boats']:
            points = bern.evalspace(x[:, :, i], t_final, (0, t_final, 10))
            for ti in range(10):
                ax.add_patch(plot_boat(points[ti, 1], points[ti, 0], np.pi / 2 - points[ti, 2], problem['boat_size']))
    for obs in problem['obstacles']:
        obs.plot(plot_inverted=True, ax=ax)
    plt.show()


def veh_coll_avoid(x1, x2, problem):
    """Calculates """
    return np.min(np.sqrt(np.sum((bern.degrelev(x1 - x2, problem['N']*10)) ** 2, axis=1))).flatten() - problem['min_dist_int_veh']**2
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
    def __init__(self, centre, rad, min_dist):
        self.centre = centre
        self.rad = rad
        self.min_dist = min_dist

    def avoid(self, poly):
        return np.sqrt(
            #np.min(np.sum((self.elev_mat @ (poly - self.centre)) ** 2, axis=1))).flatten() - self.rad - self.min_dist
            np.min(np.sum(bern.pow(bern.degrelev(poly - self.centre, 200)), axis=1))).flatten() - self.rad - self.min_dist
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

    #xin = problem.get('init_guess', rand_init_guess)(problem)
    xin = problem.get('init_guess', lin_init_guess)(problem)

    algorithm = {
        'method': 'SLSQP',
        'options': {'disp': True, 'ftol': 1e-02, 'maxiter': 5000},
    }

    constr = []

    Aeq, beq = lineqconstr(problem)
    #constr.append({'type': 'eq', 'fun': lambda x: (Aeq@x.reshape((-1,1))-beq).flatten(), 'jac': lambda x: Aeq.ravel() })
    constr.append({'type': 'eq', 'fun': lambda x: (Aeq@x.reshape((-1,1))-beq).flatten() })
    #print((Aeq@xin.reshape((-1, 1))-beq).flatten())

    A, b = linineqconstr(problem)
    if A is not None:
        constr.append({'type': 'ineq', 'fun': lambda x: (A@x.reshape((-1,1))-b).flatten() })

    if not problem['use_log_bar']:
        constr += [{'type': 'ineq', 'fun': lambda x: ineqconstr(x, problem)}]
        #print(ineqconstr(xin, problem))

    #print(cost_fun(xin, problem))

    lb = -np.inf * np.ones(((problem['N']+1),problem['dim'],problem['Nv']))
    ub = np.inf * np.ones(((problem['N']+1),problem['dim'],problem['Nv']))

    lb[0,:,:] = problem['xi'].T - .1
    ub[0,:,:] = problem['xi'].T + .1
    lb[-1,:,:] = problem['xf'].T -.1
    ub[-1,:,:] = problem['xf'].T + .1

    bnds = list(zip(lb.flatten(), ub.flatten()))

    if problem['T'] == 0:
        bnds.append((.1, np.inf))

    t = time()
    # noinspection PyTypeChecker
    res = minimize(cost_fun, xin, args=problem, method=algorithm['method'], bounds=bnds, constraints=constr, options=algorithm['options'])
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
