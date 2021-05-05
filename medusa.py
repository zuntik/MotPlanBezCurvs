from trajecoptim import run_problem, plot_xy, planner
from scipy.integrate import solve_ivp
import bernsteinlib as bern
import numpy as np
# from predefined_trajectories import circle_trajectory


def main():

    # Coefficients
    model_parameters = {
        'mass': 17.0,
        'i_z': 1,
        'x_dot_u': -20,
        'y_dot_v': -30,  # -1.3175
        'n_dot_r': -8.69,  # -0.5
        'x_u': -0.2,
        'y_v': -50,
        'n_r': -4.14,  # -0.1
        'x_uu': -25,
        'y_vv': -0.01,  # -101.2776
        'n_rr': -6.23,  # -21

        # Constants
        'fu': 0,
        'fv': 0,
        'fr': 0,
        'vcx': 0,
        'vcy': 0,
    }

    model_parameters |= {
        # masses
        'm_u': model_parameters['mass'] - model_parameters['x_dot_u'],
        'm_v': model_parameters['mass'] - model_parameters['y_dot_v'],
        'm_r': model_parameters['i_z'] - model_parameters['n_dot_r'],
        'm_uv': model_parameters['y_dot_v'] - model_parameters['x_dot_u'],
    }

    problem = {
        'T': 60,
        'xi': np.array([
            [0, 0, 0, 1, 0, 0],
            # [r*np.cos(0), r*np.sin(0), 0, 2*r*np.pi/150, 0, 0],
        ]),
        'xf': np.array([
            [30, 30, np.pi/2, 1, 0, 0],
            # [r*np.cos(0), r*np.sin(0), 0, 2*r*np.pi/150, 0, 0],
        ]),
        'state_bounds': [None, None, None, (-.1, 1.1), None, (-.74, .74)],
        'input_bounds': [(0, 25.9), (-.113, .113)],
        'N': 50,
    }

    #    problem = {
    #        'T': 15,
    #        'xi': np.array([
    #            [-10, 4, 0, 1, 0, 0],
    #            #            [-10, -4, 0, 1, 0, 0],
    #            #            [-10, 0, 0, 1, 0, 0],
    #        ]),
    #        'xf': np.array([
    #            [10, -1, 0, 1, 0, 0],
    #            #            [10, 1, 0, 1, 0, 0],
    #            #            [10, 0, 0, 1, 0, 0],
    #        ]),
    #        'statebounds': np.array([
    #            [-10000, -20000, -30000, -40000, -50000, -6000],
    #            [10000, 20000, 30000, 40000, 50000, 6000],
    #        ]),
    #        'input_bounds': np.array([[]]),
    #        'N': 40,
    #        # 'obstacles_circles': [[0, 0, 5]],
    #        'obstacles_circles': [],
    #        'obstacles_polygons': [],
    #        'min_dist_int_veh': 0.85,
    #        'min_dist_obs': 0,
    #    }

    problem |= {
        # common parameters
        'model_parameters': model_parameters,
        'num_inputs': 2,
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics,
        'recover_xy': recover_xy,
    }

    #    problem |= {
    #        # 'desiredpoints': circle_trajectory(np.linspace(0, problem['T'], problem['N']*40), problem['T'], r)
    #        'desiredpoints': circle_trajectory(np.linspace(0, problem['T'], 1000), problem['T'], r)
    #    }

    x_out, t_final, cost_final, elapsed_time = run_problem(problem)
    print('The final cost is ' + str(cost_final) + ' and the computation time was ' + str(elapsed_time))
    plot_xy(x_out, t_final, problem)


################################################################################
# functions
################################################################################
def recover_xy(x, t_final, problem):
    def ode_func(t, val, t_u, t_r, params):
        #  coefficients
        x_u = params['x_u']
        y_v = params['y_v']
        n_r = params['n_r']
        x_uu = params['x_uu']
        y_vv = params['y_vv']
        n_rr = params['n_rr']

        #  masses
        m_u = params['m_u']
        m_v = params['m_v']
        m_r = params['m_r']
        m_uv = params['m_uv']

        #  problem
        fu = params['fu']
        fv = params['fv']
        fr = params['fr']
        vcx = params['vcx']
        vcy = params['vcy']

        _, _, yaw, u, v, r = val

        # drag
        d_u = -x_u - x_uu * np.abs(u)
        d_v = -y_v - y_vv * np.abs(v)
        d_r = -n_r - n_rr * np.abs(r)

        dx = u * np.cos(yaw) - v * np.sin(yaw) + vcx
        dy = u * np.sin(yaw) + v * np.cos(yaw) + vcy
        dyaw = r
        du = 1 / m_u * (t_u(t) + m_v * v * r - d_u * u + fu)
        dv = 1 / m_v * (-m_u * u * r - d_v * v + fv)
        dr = 1 / m_r * (t_r(t) + m_uv * u * v - d_r * r + fr)

        return np.array([dx, dy, dyaw, du, dv, dr])

    def tau_u(t): return bern.eval(x[:, 6], t_final, t)

    def tau_r(t): return bern.eval(x[:, 7], t_final, t)

    odeargs = (tau_u, tau_r, problem['model_parameters'])
    sol = solve_ivp(ode_func, [0, problem['T']], x[0, :6], args=odeargs, dense_output=True, vectorized=True)

    return np.linspace(0, t_final, 1000), sol.sol(np.linspace(0, t_final, 1000))


def dynamics(x, t_final, problem):
    diff_mat = problem['DiffMat'] / t_final

    params = problem['model_parameters']
    #  coefficients
    x_u = params['x_u']
    y_v = params['y_v']
    n_r = params['n_r']
    x_uu = params['x_uu']
    y_vv = params['y_vv']
    n_rr = params['n_rr']

    #  masses
    m_u = params['m_u']
    m_v = params['m_v']
    m_r = params['m_r']
    m_uv = params['m_uv']

    #  problem
    fu = params['fu']
    fv = params['fv']
    fr = params['fr']
    vcx = params['vcx']
    vcy = params['vcy']

    # states
    xp, yp, yaw, u, v, r = x[:, 0:6].T
    # inputs
    tau_u, tau_r = x[:, 6:8].T

    # drag
    d_u = -x_u - x_uu * np.abs(u)
    d_v = -y_v - y_vv * np.abs(v)
    d_r = -n_r - n_rr * np.abs(r)

    return np.vstack((
        diff_mat @ xp - u * np.cos(yaw) + v * np.sin(yaw) + vcx,
        diff_mat @ yp - u * np.sin(yaw) - v * np.cos(yaw) + vcy,
        diff_mat @ yaw - r,
        diff_mat @ u - 1 / m_u * (tau_u + m_v * v * r - d_u * u + fu),
        diff_mat @ v - 1 / m_v * (-m_u * u * r - d_v * v + fv),
        diff_mat @ r - 1 / m_r * (tau_r + m_uv * u * v - d_r * r + fr),
    )).flatten()


def cost_fun_single(x, *_):
    return np.sum(x[:, 3]**2)
    # return np.sum(x[:, 6] ** 2) + np.sum(x[:, 7] ** 2)
    # return np.sum((problem['desiredpoints'] - (problem['EvalMat']@x)[:, :2])**2)


def medusa_planner(xi, xf, **problem):

    problem |= {
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics,
        'recover_xy': recover_xy,
    }

    return planner(xi, xf, **problem)


################################################################################
# run
################################################################################
if __name__ == "__main__":
    main()
