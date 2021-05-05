from diffeqpy import de
import numpy as np
import matplotlib.pyplot as plt


def main():

    # initial_x = [15, -5, np.pi / 2]
    initial_x = np.array([0, 0.5, np.pi / 2])
    t_final = 200

    dynamics = dynamics_dubinscar
    controller = controller_dubinscar
    reference = sine_wave

    prob = de.ODEProblem(controlled_system, initial_x, (0, t_final), p=(dynamics, controller, reference))
    sol = de.solve(prob)
    plt.plot(sol.t, sol.u)
    x_d = reference(np.linspace(0, t_final, 100))
    plt.plot(x_d[0], x_d[1])
    plt.show()


################################################################################
def sine_wave(t):
    return ([
        t,  # x position
        10 * np.sin(0.05 * t),  # y position
        1,  # dx
        0.5 * np.cos(0.05 * t),  # dy
    ])


def controlled_system(x, p, t):
    dynamics, controller, reference = p
    u = controller(t, reference, x)
    dx = dynamics(t, x, u)
    return dx


def dynamics_dubinscar(_, x, u):
    # a kinematics only model
    # This model is time independent, therefor, the "time" input is ignored
    ur = u[0]
    r = u[1]
    psi = x[2]
    d_x = [ur * np.cos(psi),  # dx
           ur * np.sin(psi),  # dy
           r]  # dyaw
    return d_x


def controller_dubinscar(t, reference, x):
    # produces the control law
    pd = np.array([reference(t)[0], reference(t)[1]]).reshape((-1, 1))
    pd_dot = np.array([reference(t)[2], reference(t)[3]]).reshape((-1, 1))
    p = x[:2].reshape((-1, 1))
    psi = x[2]

    # Setup constraint for the vehicle input (speed and heading rate)
    umin = 0
    umax = 2  # limit on the vehicle's speed
    rmin = -0.5
    rmax = 0.5  # limit on the vehicle's heading rate
    l_bound=np.array([[umin], [rmin]])
    u_bound=np.array([[umax], [rmax]])

    delta = -0.5
    Delta_i = np.array([[1], [-1 / delta]])
    epsilon = np.array([[delta], [0]])
    kx = .2
    ky = 0.05
    # Kk = np.array([[kx, 0], [0,  ky]])
    Kk = np.array([[kx], [ky]])
    RB = np.array([
        [np.cos(psi), np.sin(psi)],
        [-np.sin(psi), np.cos(psi)]
    ])[:, :, 0]
    e_pos = RB @ (p-pd) - epsilon

    u = Delta_i * (-Kk * e_pos + RB @ pd_dot)
    u = np.clip(u, l_bound, u_bound)
    return u


def example():
    def f(u,p,t):
        x, y, z = u
        sigma, rho, beta = p
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,8/3]
    prob = de.ODEProblem(f, u0, tspan, p)
    sol = de.solve(prob,saveat=0.01)

    plt.plot(sol.t,sol.u)
    plt.show()


if __name__ == "__main__":
    example()
    # main()
