from scipy.optimize import minimize
import bernsteinlib as bern
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from trajecoptim import plot_boat


def main():

    constants = setconstants()
    desiredpoints = xy_d(np.linspace(0, constants['T'], 1000), constants['T'])
    # xin = np.random.rand(constants['N']+1, 3).flatten()
    xin = xy_d(np.linspace(0, constants['T'], constants['N']+1), constants['T'])
    xin = np.concatenate((xin, np.pi/2+np.linspace(0, 2*np.pi, constants['N']+1).reshape((-1, 1))), axis=1)
    costbefore = itegralofdifference(xin, constants, desiredpoints)
    cons = [
        {'type': 'eq', 'fun': lambda x: eqconstr(x, constants)},
        {'type': 'ineq', 'fun': lambda x: ineqconstr(x, constants)}
    ]
    opts = {'disp': True, 'maxiter': 1000}
    # noinspection PyTypeChecker
    xout = minimize(lambda x: itegralofdifference(x, constants, desiredpoints), xin,
                    method='SLSQP', constraints=cons, options=opts)
    xout = xout.x.reshape((constants['N'] + 1, 3))
    costafter = itegralofdifference(xout, constants, desiredpoints)
    _, ax = plt.subplots()
    ax.plot(desiredpoints[:, 0], desiredpoints[:, 1])
    ax.axis('equal')
    _, ax = plt.subplots()
    ax.axis('equal')
    bern.plot(np.fliplr(xout[:, :2]), constants['T'], ax)
    points = bern.eval(xout, constants['T'], np.linspace(0, constants['T'], 10))
    for ti in range(10):
        ax.add_patch(plot_boat(points[ti, 1], points[ti, 0], np.pi / 2 - points[ti, 2], 3))
    t, xy = recoverplot(xout, constants)
    _, ax = plt.subplots()
    ax.axis('equal')
    ax.plot(xy[1, :], xy[0, :].T)
    _, ax = plt.subplots()
    _, v, _, _, _ = calcothers(xout, constants)
    bern.plot(v.reshape((-1, 1)), constants['T'], ax)

    _, ax = plt.subplots()
    def yaw(tt): return bern.eval(xout[:, 2].reshape((-1, 1)), constants['T'], tt)
    dx = bern.deriv(xout[:, 0].reshape((-1, 1)), constants['T'])
    dy = bern.deriv(xout[:, 0].reshape((-1, 1)), constants['T'])
    def speedangle(tt): return np.arctan2(bern.eval(dy, constants['T'], tt), bern.eval(dx, constants['T'], tt))
    def sideslip(tt): return yaw(tt) - speedangle(tt)
    ax.plot(np.linspace(0, constants['T'], 100), sideslip(np.linspace(0, constants['T'], 100)))

    plt.show()
    print('cost before: ' + str(costbefore) + '\n cost after: ' + str(costafter))
    print('eq constr before: ' + str(np.sum(eqconstr(xin, constants))))
    print('eq constr after: ' + str(np.sum(eqconstr(xout, constants))))


def xy_d(t, tf):
    # circle
    w = 2 * np.pi / tf
    return np.concatenate((10 * np.cos(w * t).reshape((-1, 1)), 10 * np.sin(w * t).reshape((-1, 1))), axis=1)


def itegralofdifference(x, constants, desiredpoints):
    x = x.reshape((constants['N']+1, 3))
    evalpoints = constants['EvalMat']@x
    return np.sum((desiredpoints - evalpoints[:, :2])**2)


def setconstants():
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

    #    bds = {
    #        'x': {'min': -10000, 'max': 1000},
    #        'y': {'min': -20000, 'max': 2000},
    #        'psi': {'min': -30000, 'max': 3000},
    #        'u': {'min': -40000, 'max': 4000},
    #        'v': {'min': -50000, 'max': 5000},
    #        'r': {'min': -60000, 'max': 6000},
    #        'tau_u': {'min': -70000, 'max': 7000},
    #        'tau_r': {'min': -80000, 'max': 8000}
    #    }
    bds = {
        'x': {'min': -10000, 'max': 1000},
        'y': {'min': -20000, 'max': 2000},
        'psi': {'min': -30000, 'max': 3000},
        'u': {'min': -.1, 'max': 1.1},
        'v': {'min': -50000, 'max': 5000},
        'r': {'min': -.74, 'max': .74},
        'tau_u': {'min': 0, 'max': 25.9},
        'tau_r': {'min': -.113, 'max': .113}
    }

    constants = {
        'T': 70,
        'bounds': bds,
        'statebounds': np.array([
            [bds['x']['min'], bds['y']['min'], bds['psi']['min'], bds['u']['min'], bds['x']['min']],
            [bds['x']['max'], bds['y']['max'], bds['psi']['max'], bds['u']['max'], bds['x']['max']],
        ]),
        'inputbounds': np.array([[]]),
        'N': 70,
    }

    constants |= {
        # common parameters
        'modelparams': modelparams,
    }

    constants |= {
        # common parameters
        'DiffMat': bern.derivelevmat(constants['N'], constants['T']),
        'ElevMat': bern.degrelevmat(constants['N'], constants['N'] * 10),
        'EvalMat': bern.evalmat(constants['N'], constants['T'], np.linspace(0, constants['T'], 1000)),
    }

    return constants


def ineqconstr(x, constants):
    x = x.reshape((-1, 3))
    u, v, r, tau_u, tau_r = calcothers(x, constants)
    return np.concatenate([
        u - constants['bounds']['u']['min'],
        constants['bounds']['u']['max'] - u,
        v - constants['bounds']['v']['min'],
        constants['bounds']['v']['max'] - v,
        tau_u - constants['bounds']['tau_u']['min'],
        constants['bounds']['tau_u']['max'] - tau_u,
        tau_r - constants['bounds']['tau_r']['min'],
        constants['bounds']['tau_r']['max'] - tau_r
    ]).flatten()


def eqconstr(x, constants):
    x = x.reshape((constants['N']+1, 3))
    diffmat = constants['DiffMat']
    params = constants['modelparams']
    #  coefficients
    y_v = params['y_v']
    y_vv = params['y_vv']

    #  masses
    m_u = params['m_u']
    m_v = params['m_v']

    #  constants
    fv = params['fv']
    vcx = params['vcx']
    vcy = params['vcy']

    # states
    xp = x[:, 0]
    yp = x[:, 1]
    yaw = x[:, 2]

    dx = diffmat @ xp
    dy = diffmat @ yp

    u = (dx - vcx) * np.cos(yaw) + (dy - vcy) * np.sin(yaw)
    v = -(dx - vcx) * np.sin(yaw) + (dy - vcy) * np.cos(yaw)
    r = diffmat @ yaw

    # drag
    d_v = -y_v - y_vv * np.abs(v)

    return np.vstack((
        diffmat @ v - 1 / m_v * (-m_u * u * r - d_v * v + fv),
    )).flatten()


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

    u_cp, v_cp, r_cp, tau_u_cp, tau_r_cp = calcothers(x, constants)

    xi = np.concatenate((x[0, :].flatten(), np.array([u_cp[0], v_cp[0], r_cp[0]])))
    def tau_u(t): return bern.eval(tau_u_cp, constants['T'], t)

    def tau_r(t): return bern.eval(tau_r_cp, constants['T'], t)

    odeargs = (tau_u, tau_r, constants['modelparams'])
    sol = solve_ivp(odefunc, [0, constants['T']], xi, args=odeargs, dense_output=True, vectorized=True)

    return np.linspace(0, constants['T'], 1000), sol.sol(np.linspace(0, constants['T'], 1000))


def calcothers(x, constants):
    diffmat = constants['DiffMat']
    params = constants['modelparams']
    #  coefficients
    x_u = params['x_u']
    # y_v = params['y_v']
    n_r = params['n_r']
    x_uu = params['x_uu']
    # y_vv = params['y_vv']
    n_rr = params['n_rr']

    #  masses
    m_u = params['m_u']
    m_v = params['m_v']
    m_r = params['m_r']
    m_uv = params['m_uv']

    #  constants
    fu = params['fu']
    # fv = params['fv']
    # fr = params['fr']
    vcx = params['vcx']
    vcy = params['vcy']

    # states
    xp = x[:, 0]
    yp = x[:, 1]
    yaw = x[:, 2]

    dx = diffmat @ xp
    dy = diffmat @ yp

    u = (dx - vcx) * np.cos(yaw) + (dy - vcy) * np.sin(yaw)
    v = -(dx - vcx) * np.sin(yaw) + (dy - vcy) * np.cos(yaw)
    r = diffmat @ yaw

    # drag
    d_u = -x_u - x_uu * np.abs(u)
    # d_v = -y_v - y_vv * np.abs(v)
    d_r = -n_r - n_rr * np.abs(r)

    tau_u = m_u * diffmat @ u - m_v * v * r + d_u * u - fu
    tau_r = m_r * diffmat @ r - m_uv * u * v + d_r * r

    return u, v, r, tau_u, tau_r


if __name__ == "__main__":
    main()
