from .trajecoptim import run_problem, plot_xy, planner
from scipy.integrate import solve_ivp
from . import bernsteinlib as bern
import numpy as np


def main():

    # first illustrative example
    offset = np.array([1234, 185943, 0, 0, 0])
    problem = {
        'T': 10,  # runtime
        'xi': np.array([[0, 0, 0, 1, 0]])+ offset,  # initial states
        'xf': np.array([[5, 5, np.pi / 2, 1, 0]])+offset,  # final states
        'N': 20,  # order of the polynomials
        'obstacles_circles': [[5+offset[0], 0+offset[1], 3]],  # n lines for n circles where columns are position x, position y, radius
        'state_bounds': [None, None, None, (-1, 1), (-5, 5)]
    }

    #    problem = {
    #        'N': 30,
    #        'T': 15,
    #        'xi': np.array([
    #            [0, 5, 0, 1, 0],
    #            [5, 0, np.pi / 2, 1, 0]
    #        ]),
    #        'xf': np.array([
    #            [10, 5, 0, 1, 0],
    #            [5, 10, np.pi / 2, 1, 0]
    #        ]),
    #        'min_dist_int_veh': 2,
    #    }

    #    problem = {
    #        'N': 20,
    #        'T': 20,
    #        'xi': np.array([
    #            [-10, 4, 0, 1, 0],
    #            [-10, -4, 0, 1, 0],
    #            [-10, 0, 0, 1, 0],
    #            [0, -10, 0, 1, 0],
    #        ]),
    #        'xf': np.array([
    #            [10, -3, 0, 1, 0],
    #            [10, 3, 0, 1, 0],
    #            [10, 0, 0, 1, 0],
    #            [0, 10, 0, 1, 0],
    #        ]),
    #        'obstacles_circles': [[0, 0, 3]],
    #        'min_dist_int_veh': 1,
    #        'state_bounds': [None, None, None, (-.2, 2), (-2, 2)],
    #    }

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


if __name__ == "__main__":
    main()
