import sympy as sym
import numpy as np
from time import time

from ddp import DDPOptimizer
from cartpole_r import CartPoleREnv
from cartpole_r2 import CartPoleR2Env
from cartpole_og import CartPoleOGEnv

with_plots = False
try:
    import matplotlib.pyplot as plt

    with_plots = True
except ImportError:
    print("ERROR: matplotlib not found. Skipping plots")


# dynamics parameters
mp = 0.01
mc = 1.0
l = 0.25
# G = 9.80665
G = 9.8
# dt = 0.05
dt = 0.01
friccoeff = [0.5, 0.05]

# dynamics
def f(x, u, constrain=False):

    x_ = x[0]
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]
    F = sym.tanh(u[0]) if constrain else u[0]

    cos_theta = sym.cos(theta)
    sin_theta = sym.sin(theta)

    # Define dynamics model as per Razvan V. Florian's
    # "Correct equations for the dynamics of the cart-pole system".
    # Friction is neglected.

    # Eq. (23)
    temp = (F + mp * l * theta_dot**2 * sin_theta) / (mc + mp) #- x_dot*friccoeff[0]/mc
    numerator = G * sin_theta - cos_theta * temp
    denominator = l * (4.0 / 3.0 - mp * cos_theta**2 / (mc + mp))
    theta_dot_dot = numerator / denominator #- theta_dot*friccoeff[1]/mp

    # Eq. (24)
    x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

    # Deaugment state for dynamics.
    theta = sym.atan2(sin_theta, cos_theta)
    next_theta = theta + theta_dot * dt

    return sym.Matrix(
        [
            x_ + x_dot * dt,
            x_dot + x_dot_dot * dt,
            next_theta,
            theta_dot + theta_dot_dot * dt,
        ]
    )


# instantaneous cost
def g(x, u, x_goal):
    # error = x - x_goal
    # Q = np.eye(len(x))
    # # # Q[0, 0] = 5000
    # Q[0, 0] = 0
    # Q[1, 1] = 0.1
    # Q[2, 2] = 1.5
    # Q[3, 3] = 0.1
    # R = 15 * np.eye(len(u))
    R = 0.3 * np.eye(len(u))
    # a_cost = 0.25
    # ctrl_cost = 0.01*a_cost*a_cost*(sym.cosh(u[0]/a_cost)-1)
    # ctrl_cost = sym.Matrix([ctrl_cost])
    # print(type(error.T @ Q @ error))
    # print(type(sym.Matrix([ctrl_cost])))
    # return error.T @ Q @ error + ctrl_cost
    return u.T @ R @ u
    # return error.T @ Q @ error + u.T @ R @ u


# final cost
def h(x, x_goal):
    error = x - x_goal
    Q = 100 * np.eye(len(x))
    # Q[0, 0] = 5000
    Q[0, 0] = 100
    Q[1, 1] = 100
    Q[2, 2] = 15000
    Q[3, 3] = 300

    return error.T @ Q @ error


# trajectory parameters
max_iters = 400
N = 500  # trajectory points
Nx = 4  # state dimension
Nu = 1  # control dimesions

# starting state
x0 = np.array([0.0, 0.0, np.pi, 0.0])
# x0 = np.array([0.0, 0.0, np.sin(0.0), np.cos(0.0), 0.0])

# goal state we want to reach
x_goal = np.array([0.0, 0.0, 0.0, 0.0])

print("Starting state", x0)
print("Goal state", x_goal)

# # Create and run optimizer with random intialization
# print("Starting optimization")
env1 = CartPoleR2Env(init_state=x0.copy(), render_mode='human')
env1.reset()
for i in range(1):
    start_time = time()
    ddp = DDPOptimizer(
        Nx, Nu,
        f, g, h,
        max_iters=max_iters
    )
    X, U, X_hist, U_hist, J_hist = ddp.optimize(
        x0, x_goal, N=N,
        full_output=True
    )
    print(X)
    input()
    print("Finished optimization in {:.2f}s".format(time() - start_time))

    # input('Press "Enter" to start.. ..')
    for t in range(len(U)):
        env1.step(action=np.float32(U[t][0]))
        print(env1.state)
    x_0 = env1.state
    # env1.step(action=0.0)

# for i in range(10000):
    # action = 0.1*np.sin(i)
    # # if abs(env1.state[0]) < 1e-3:
    # #     action = 1
    # # else:
    # #     action = -int(env1.state[0]/abs(env1.state[0]))+1
    # #     # print(int(env1.state[0]/abs(env1.state[0])))
    # # if i < 10:
    # #     action = 2
    # # print(env1.state[0])
    # env1.step(action=action)
    # print(i)

# plot results
if with_plots:
    print("Plotting results")

    fig, ax = plt.subplots(3, 1, figsize=(4, 8))
    tt = np.linspace(0, dt * N, N)
    theta_sol = np.unwrap(np.arctan2(X[:, 2], X[:, 3]))
    theta_dot_sol = X[:, 3]

    ax[0].plot(theta_sol, theta_dot_sol)
    ax[0].set_xlabel(r"$\theta (rad)$")
    ax[0].set_ylabel(r"$\dot{\theta} (rad/s)$")
    ax[0].set_title("Phase Plot")
    ax[1].set_title("Control")
    ax[1].plot(tt, np.tanh(U))
    ax[1].set_xlabel("Time (s)")
    ax[2].plot(J_hist)
    ax[2].set_title("Trajectory cost")
    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    plt.savefig("ddp_cartpole.png")
