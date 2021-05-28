from .trajecoptim import run_problem, plot_xy, planner
from scipy.integrate import solve_ivp
from . import bernsteinlib as bern
import numpy as np


def main():

    # first illustrative example
    problem = {
        'xi': np.array([[0, 0, 0, 0]]),  # initial states
        'xf': np.array([[5, 5, np.pi/2, 0]]),  # final states
        'v': .5,
        'N': 50,  # order of the polynomials
        # 'obstacles_circles': [[5, 0, 3]],  # n lines for n circles where columns are position x, position y, radius
        # 'state_bounds': [(-5, 10), (-5, 10), (-np.pi / 2, 2 * np.pi), (-5, 5)],
        'state_bounds': [(-5, 10), (-5, 10), (-np.pi / 2, 2 * np.pi), (-.5, .5)],
    }

    #    problem = {
    #        'N': 30,
    #        'xi': np.array([
    #            [0, 5, 0, 0],
    #            [5, 0, np.pi / 2, 0]
    #        ]),
    #        'xf': np.array([
    #            [10, 5, 0, 0],
    #            [5, 10, np.pi / 2, 0]
    #        ]),
    #        'min_dist_int_veh': 3,
    #    }

    #    problem = {
    #        'N': 20,
    #        'T': 15,
    #        'xi': np.array([
    #            [-10, 4, 0, 0],
    #            [-10, -4, 0, 0],
    #            [-10, 0, 0, 0],
    #            [0, -10, 0, 0],
    #        ]),
    #        'xf': np.array([
    #            [10, -3, 0, 0],
    #            [10, 3, 0, 0],
    #            [10, 0, 0, 0],
    #            [0, 10, 0, 0],
    #        ]),
    #        'obstacles_circles': [[0, 0, 3]],
    #        'min_dist_int_veh': 1,
    #    }

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics4vars,
        # 'recover_xy': recover_xy,
        # 'init_guess': init_guess,
    }}

    x_out, t_final, cost_final, elapsed_time = run_problem(problem)
    print('The final cost is ' + str(cost_final) + ' and the computation time was ' + str(elapsed_time))
    plot_xy(x_out, t_final, problem)


################################################################################
# functions
################################################################################
def recover_xy(x, t_final, problem):
    def ode_func(t, val, v, w):
        _, y, psi = val
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = w(t)
        return np.array([dx, dy, dpsi])

    def pol_w(t): return bern.eval(x[:, 4], t_final, t)

    sol = solve_ivp(ode_func, [0, t_final], x[0, :3], args=(problem['v'], pol_w), dense_output=True, vectorized=True)

    return np.linspace(0, t_final, 1000), sol.sol(np.linspace(0, t_final, 1000))


def dynamics4vars(x, t_final, problem):
    """the vehicle dynamics"""
    diff_mat = problem['DiffMat'] / t_final
    xp, yp, psi, w = x.T
    v = problem['v']
    return np.vstack((
        diff_mat @ xp - v * np.cos(psi),
        diff_mat @ yp - v * np.sin(psi),
        diff_mat @ psi - w,
    )).flatten()


def cost_fun_single(_, t_final, __):
    """the running cost for a singular vehicle"""
    return t_final


def dubinscar_no_v_planner(xi, xf, **problem):

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics4vars,
        'recover_xy': recover_xy,
    }}

    return planner(xi, xf, **problem)


if __name__ == "__main__":
    main()
