import numpy as np

def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2

        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature `dy = derivs(yi)`
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float64)
    else:
        yout = np.zeros((len(t), Ny), np.float64)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]

def rk2(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 2nd-order Runge-Kutta (Ralston's method).
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float64)
    else:
        yout = np.zeros((len(t), Ny), np.float64)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        dt = t[i + 1] - t[i]
        dt2 = dt / 2.0

        y = yout[i]

        k1 = np.asarray(derivs(y))
        k2 = np.asarray(derivs(y + dt2 * k1))

        yout[i + 1] = y + dt * ((1/4) * k1 + (3/4) * k2)

    return yout[-1][:4]

def feuler(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using Euler's method.
    """
    yout = np.zeros((len(t), len(y0)))
    yout[0] = y0
    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        yout[i + 1] = yout[i] + dt * np.asarray(derivs(yout[i]))
    return yout[-1][:4]

def seuler(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using semi-implicit Euler's method.
    """
    yout = np.zeros((len(t), len(y0)))
    yout[0] = y0

    for i in np.arange(len(t) - 1):
        dt = t[i + 1] - t[i]

        y = yout[i]
        derivatives = derivs(y)

        acc = np.asarray(derivatives)[2:4]
        vel = y[2:4] + dt * acc
        pos = y[:2] + dt * vel

        torque = y[-1]  # keep input constant

        yout[i + 1] = np.concatenate([pos, vel, [torque]])

    return yout[-1][:4]

def ieuler(derivs, y0, t, tol=1e-8, max_iter=50):
    y0 = np.asarray(y0, dtype=np.float64)
    Ny = len(y0)
    yout = np.zeros((len(t), Ny), np.float64)
    yout[0] = y0

    torque = y0[-1]  # extract once, hold constant throughout

    def derivs4(y4):
        """Call derivs with the torque dimension re-attached, return only first 4 outputs."""
        return np.asarray(derivs(np.append(y4, torque)))[:4]

    def numerical_jacobian(f, y, eps=1e-6):
        n = len(y)
        J = np.zeros((n, n))
        f0 = f(y)
        for j in range(n):
            yp = y.copy()
            yp[j] += eps
            J[:, j] = (f(yp) - f0) / eps
        return J

    for i in np.arange(len(t) - 1):
        dt = t[i + 1] - t[i]
        yn = yout[i][:4]  # work only over physical state

        y_next = yn + dt * derivs4(yn)

        for _ in range(max_iter):
            f_next = derivs4(y_next)
            g = y_next - yn - dt * f_next

            if np.linalg.norm(g) < tol:
                break

            J_g = np.eye(4) - dt * numerical_jacobian(derivs4, y_next)
            delta = np.linalg.solve(J_g, g)
            y_next = y_next - delta

        yout[i + 1] = np.append(y_next, torque)  # re-attach torque for storage

    return yout[-1][:4]

def vverlet(derivs, y0, t):
    """
    Integrate using Velocity Verlet method.
    """
    y = np.array(y0[:4], dtype=np.float64)
    torque = y0[-1]

    def get_acc(state):
        # get accelerations (ddtheta1, ddtheta2) from derivs
        full = np.append(state, torque)
        d = np.asarray(derivs(full))
        return d[2:4]  # ddtheta1, ddtheta2

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]

        pos = y[:2]  # theta1, theta2
        vel = y[2:4]  # dtheta1, dtheta2

        a_curr = get_acc(y)

        # step position using current velocity AND acceleration
        new_pos = pos + vel * dt + 0.5 * a_curr * dt**2

        # get acceleration at new position
        new_state = np.concatenate([new_pos, vel])
        a_next = get_acc(new_state)

        # step velocity using AVERAGE of current and next acceleration
        new_vel = vel + 0.5 * (a_curr + a_next) * dt

        y = np.concatenate([new_pos, new_vel])

    return y[:4]