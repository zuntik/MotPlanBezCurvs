from trajecoptim import run_problem, plot_xy
from scipy.integrate import solve_ivp
import bernsteinlib as bern
import numpy as np


def main():

    # first illustrative example
    constants = {
        'T': 0,  # runtime
        'xi': np.array([[0, 0, 0, 1, 0]]),  # initial states
        'xf': np.array([[5, 5, np.pi / 2, 1, 0]]),  # final states
        'statebounds': np.array([[-10000, -20000, -30000, -40000, -50000], [10000, 20000, 30000, 40000, 50000]]),
        'N': 50,  # order of the polynomials
        'obstacles_circles': [[5, 0, 3]],  # n lines for n circles where columns are position x, position y, radius
    }

    #    constants = {
    #        'N': 30,
    #        'T': 15,
    #        'xi': np.array([[0, 5, 0, 1, 0], [5, 0, np.pi / 2, 1, 0]]),
    #        'xf': np.array([[10, 5, 0, 1, 0], [5, 10, np.pi / 2, 1, 0]]),
    #        'statebounds': None,
    #        'inputboudns': None,
    #        'obstacles_circles': [],
    #        'obstacles_polygons': [],
    #        'min_dist_int_veh': 3,
    #        'min_dist_obs': 0,
    #    }

    #    constants = {
    #        'N': 40,
    #        'T': 15,
    #        'xi': np.array([
    #            [-10, 4, 0, 1, 0],
    #            [-10, -4, 0, 1, 0],
    #            [-10, 0, 0, 1, 0],
    #        ]),
    #        'xf': np.array([
    #            [10, -1, 0, 1, 0],
    #            [10, 1, 0, 1, 0],
    #            [10, 0, 0, 1, 0],
    #        ]),
    #        'obstacles_circles': [[0, 0, 3]],
    #        'obstacles_polygons': [],
    #        'min_dist_obs': 0,
    #        'min_dist_int_veh': 0.9,
    #        'statebounds': None,  # np.array([[-20, -20, -10, -10, -10], [20, 20, 10, 10, 10]]),
    #        'inputbounds': None,
    #    }

    constants = {** constants, **{
        'uselogbar': False,
        'usesigma': True,
        # functions
        'costfun_single': costfun_single,
        'dynamics': dynamics5vars,
        'recoverxy': recoverplot,
    }}

    x_out, t_final, cost_final, elapsedtime, singtimes = run_problem(constants)
    print('The final cost is ' + str(cost_final))
    plot_xy(x_out, t_final, constants)


################################################################################
# functions
################################################################################
def recoverplot(x, _, constants):
    def odefunc(t, val, v, w):
        _, y, psi = val
        dx = v(t) * np.cos(psi)
        dy = v(t) * np.sin(psi)
        dpsi = w(t)
        return np.array([dx, dy, dpsi])

    def pol_v(t): return bern.eval(x[:, 3], constants['T'], t)

    def pol_w(t): return bern.eval(x[:, 4], constants['T'], t)

    sol = solve_ivp(odefunc, [0, constants['T']], x[0, :3], args=(pol_v, pol_w), dense_output=True, vectorized=True)

    return np.linspace(0, constants['T'], 1000), sol.sol(np.linspace(0, constants['T'], 1000))


def dynamics5vars(x, _, constants):
    """the vehicle dynamics"""
    diffmat = constants['DiffMat']
    xp, yp, psi, v, w = (x[:, i] for i in range(5))

    return np.vstack((
        diffmat @ xp - v * np.cos(psi),
        diffmat @ yp - v * np.sin(psi),
        diffmat @ psi - w,
    )).flatten()


def costfun_single(x, _, constants):
    """the running cost for a singular vehicle"""
    v = x[:, 3]
    w = x[:, 4]
    a = constants['DiffMat'] @ v
    # return np.sum((constants['ElevMat']@a)**2)+2*np.sum((constants['ElevMat']@w)**2)
    return np.sum(a ** 2) + 2 * np.sum(w ** 2)


################################################################################
# run
################################################################################

if __name__ == "__main__":
    main()
