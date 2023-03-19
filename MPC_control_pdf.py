import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def mpc_control(x, y, pdf, u_prev, v_prev, T, Q, R, dt):
    # x: current position on x-axis
    # y: current position on y-axis
    # pdf: continous pdf of probability map
    # u_prev: previous control input (acceleration) on x-axis
    # v_prev: previous control input (acceleration) on y-axis
    # T: horizon length
    # Q: state weighting matrix
    # R: control weighting matrix

    u_min = 0.5  # min acceleration x direction
    u_max = 0.5  # max acceleration x direction
    v_min = 0.5  # min acceleration y direction
    v_max = 0.5  # max acceleration y direction

    # Initialize optimization variables
    N = T + 1  # number of time steps
    X = np.zeros((N, 2))  # state matrix (x-axis)
    Y = np.zeros((N, 2))  # state matrix (y-axis)
    U = np.zeros((N-1, 1))  # control input matrix (x-axis)
    V = np.zeros((N-1, 1))  # control input matrix (y-axis)

    # Set initial state and control input
    X[0] = x
    Y[0] = y
    U[0] = u_prev
    V[0] = v_prev

    # Define objective function and constraints
    def objective(UV):
        U = UV[:N-1]
        V = UV[N-1:]
        X[1:] = propagate_dynamics(X[:-1], U, N-1)
        Y[1:] = propagate_dynamics(Y[:-1], V, N-1)
        cost = compute_cost(X, Y, U, V, Q, R)
        return cost

    # Constraint 1
    def constraint1(UV):
        U = UV[:N-1]
        V = UV[N-1:]
        return np.concatenate((-U+u_max, -V+v_max))

    # Constraint 2
    def constraint2(UV):
        U = UV[:N-1]
        V = UV[N-1:]
        return np.concatenate((U+u_min, V+v_min))

    cons = ({'type':'ineq', 'fun': constraint1},{'type':'ineq', 'fun': constraint2})

    def propagate_dynamics(X, U, N):
        # Propagate system dynamics for N time steps
        for i in range(N-1):
            X[i+1] = X[i] + dt*np.array([X[i,1], U[i]])
        return X

    def compute_cost(X, Y, U, V, Q, R):
        # Compute cost based on states and control inputs
        cost = 0
        test = 0
        for i in range(N-1):
            cost += -pdf(X[i,0],Y[i,0])*Q
            cost += np.dot(U[i], U[i])*R
            cost += np.dot(V[i], V[i])*R
        return test + cost#[0,0]

    # Solve optimization problem
    result = minimize(objective, np.concatenate((U, V)), constraints=cons)
    UV_opt = result.x
    U_opt = UV_opt[:N-1]
    V_opt = UV_opt[N-1:]

    # Return the all open loop control inputs in the optimized sequence
    return U_opt, V_opt


