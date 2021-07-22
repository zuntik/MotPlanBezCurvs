from .trajecoptim import run_problem, plot_xy, planner
from scipy.integrate import solve_ivp
from . import bernsteinlib as bern
import numpy as np
import matplotlib.pyplot as plt


def main():

    #    problem = {
    #        'T': 40,  # runtime
    #        'xi': np.array([[0, 0, 0, .1, 0]]),  # initial states
    #        'xf': np.array([[10, -10, -np.pi / 2, .1, 0]]),  # final states
    #        'N': 20,  # order of the polynomials
    #        'obstacles_circles': [[5, 0, 3]],  # n lines for n circles where columns are position x, position y, radius
    #    }

    #    offset = np.array([491935.997, 4290797.518, 0, 0, 0])
    #    problem = {
    #        'T': 50,  # runtime
    #        'xi': np.array([[0, 0, 0, 0, 0]]) + offset,  # initial states
    #        'xf': np.array([[20, 20, np.pi/2, 0, 0]]) + offset,  # final states
    #        'N': 40,  # order of the polynomials
    #        'obstacles_circles': [[10+offset[0], 10+offset[1], 10]],  # n lines for n circles where columns are position x, position y, radius
    #    }

    problem = {
        'N': 40,
        'T': 40,
        'xi': np.array([
            [-10, 4, np.pi/2, .1, 0],
            [-10, -4, 0, 0.1, 0],
            [-10, 0, np.pi/4, .1, 0],
            [0, -10, 0, .1, 0],
        ]),
        'xf': np.array([
            [10, -3, 0, .1, 0],
            [10, 3, 0,  .1, 0],
            [10, 0, 0,  .1, 0],
            [0, 10, 0,  .1, 0],
        ]),
        #'obstacles_circles': [[0, 0, 2]],
        #'obstacles_circles': [[0, 0, 2], [-5, -5, 2], [-5, 5, 2.5]],
    }
    problem = {**problem, **{
        'input_bounds': [(-.1, .1), (-.5, .5)],
        'state_bounds': [None, None, None, (-.1, 1), (-.5, .5)],
    }}

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics,
        'num_inputs': 2,
        #'recover_xy': recover_xy,
    }}

    x_out, t_final, cost_final, elapsed_time = run_problem(problem)
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


def dynamics(x, t_final, problem):
    """the vehicle dynamics"""
    diff_mat = problem['DiffMat'] / t_final
    xp, yp, psi, v, w, dv, dw = x.T
    return np.vstack((
        diff_mat @ xp - v * np.cos(psi),
        diff_mat @ yp - v * np.sin(psi),
        diff_mat @ psi - w,
        diff_mat @ v - dv,
        diff_mat @ w - dw,
    )).flatten()


def cost_fun_single(x, t_final, problem):
    """the running cost for a singular vehicle"""
    if problem['T'] == 0:
        return t_final + np.sum((x[1:,:2]-x[:-1,:2])**2)
    v = x[:, 3]
    w = x[:, 4]
    #a = (problem['DiffMat'] / t_final) @ v
    a = x[:, 5]
    # return np.sum((problem['elev_mat']@a)**2)+2*np.sum((problem['ElevMat']@w)**2)
    return np.sum(a ** 2) + 2 * np.sum(w ** 2)


def dubinscar_planner(xi, xf, **problem):

    problem = {**problem, **{
        # functions
        'cost_fun_single': cost_fun_single,
        'dynamics': dynamics,
        'recover_xy': recover_xy,
    }}

    return planner(xi, xf, **problem)


if __name__ == "__main__":
    main()
