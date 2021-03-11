from trajecoptim import run_problem, plot_xy
from scipy.integrate import solve_ivp
import bernsteinlib as bern
import numpy as np
from predefined_trajectories import circle_trajectory


def main():

    # Coefficients
    modelparams = {
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

    modelparams = {**modelparams, **{
        # masses
        'm_u': modelparams['mass'] - modelparams['x_dot_u'],
        'm_v': modelparams['mass'] - modelparams['y_dot_v'],
        'm_r': modelparams['i_z'] - modelparams['n_dot_r'],
        'm_uv': modelparams['y_dot_v'] - modelparams['x_dot_u'],
    }}

    constants = {
        'T': 15,
        'xi': np.array([
            [0, 0, 0, 1, 0, 0],
            # [r*np.cos(0), r*np.sin(0), 0, 2*r*np.pi/150, 0, 0],
        ]),
        'xf': np.array([
            [0, 6, np.pi, 1, 0, 0],
            # [r*np.cos(0), r*np.sin(0), 0, 2*r*np.pi/150, 0, 0],
        ]),
        'statebounds': np.array([
            [-10000, -20000, -30000, -.1, -1000, -.74],
            [10000, 20000, 30000, 1.1, 1000, .74],
        ]),
        'inputbounds': np.array([
            [0, -0.113],
            [25.9, 0.113],
        ]),
        'N': 50,
    }

    #    constants = {
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
    #        'inputbounds': np.array([[]]),
    #        'N': 40,
    #        # 'obstacles_circles': [[0, 0, 5]],
    #        'obstacles_circles': [],
    #        'obstacles_polygons': [],
    #        'min_dist_int_veh': 0.85,
    #        'min_dist_obs': 0,
    #    }

    constants = {**constants, **{
        # common parameters
        'modelparams': modelparams,
        'numinputs': 2,
        # functions
        'costfun_single': costfun_single,
        'dynamics': dynamics,
        'recoverxy': recoverplot,
    }}

    #    constants = {**constants, **{
    #        # 'desiredpoints': circle_trajectory(np.linspace(0, constants['T'], constants['N']*40), constants['T'], r)
    #        'desiredpoints': circle_trajectory(np.linspace(0, constants['T'], 1000), constants['T'], r)
    #    }}

    res, elapsedtime, singtimes = run_problem(constants)
    print('The final cost is ' + str(res.fun))
    plot_xy(res, constants)


################################################################################
# functions
################################################################################
def recoverplot(x, constants):
    def odefunc(t, val, t_u, t_r, params):
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

        #  constants
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

    def tau_u(t): return bern.eval(x[:, 6], constants['T'], t)

    def tau_r(t): return bern.eval(x[:, 7], constants['T'], t)

    odeargs = (tau_u, tau_r, constants['modelparams'])
    sol = solve_ivp(odefunc, [0, constants['T']], x[0, :6], args=odeargs, dense_output=True, vectorized=True)

    return np.linspace(0, constants['T'], 1000), sol.sol(np.linspace(0, constants['T'], 1000))


def dynamics(x, constants):
    diffmat = constants['DiffMat']

    params = constants['modelparams']
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

    #  constants
    fu = params['fu']
    fv = params['fv']
    fr = params['fr']
    vcx = params['vcx']
    vcy = params['vcy']

    # states
    xp = x[:, 0]
    yp = x[:, 1]
    yaw = x[:, 2]
    u = x[:, 3]
    v = x[:, 4]
    r = x[:, 5]
    # inputs
    tau_u = x[:, 6]
    tau_r = x[:, 7]

    # drag
    d_u = -x_u - x_uu * np.abs(u)
    d_v = -y_v - y_vv * np.abs(v)
    d_r = -n_r - n_rr * np.abs(r)

    return np.vstack((
        diffmat @ xp - u * np.cos(yaw) + v * np.sin(yaw) + vcx,
        diffmat @ yp - u * np.sin(yaw) - v * np.cos(yaw) + vcy,
        diffmat @ yaw - r,
        diffmat @ u - 1 / m_u * (tau_u + m_v * v * r - d_u * u + fu),
        diffmat @ v - 1 / m_v * (-m_u * u * r - d_v * v + fv),
        diffmat @ r - 1 / m_r * (tau_r + m_uv * u * v - d_r * r + fr),
    )).flatten()


def costfun_single(x, *_):
    return np.sum(x[:, 6] ** 2) + np.sum(x[:, 7] ** 2)
    # return np.sum((constants['desiredpoints'] - (constants['EvalMat']@x)[:, :2])**2)


################################################################################
# run
################################################################################

if __name__ == "__main__":
    main()
