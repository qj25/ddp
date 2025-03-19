import sympy as sym
import numpy as np
from time import time

from ddp import DDPOptimizer

with_plots = False
try:
    import matplotlib.pyplot as plt

    with_plots = True
except ImportError:
    print("ERROR: matplotlib not found. Skipping plots")


# dynamics parameters
mp = 0.1
mc = 1.0
l = 1.0
G = 9.80665
dt = 0.05

# dynamics
def f(x, u, constrain=True):

    x_ = x[0]
    x_dot = x[1]
    sin_theta = x[2]
    cos_theta = x[3]
    theta_dot = x[4]
    F = sym.tanh(u[0]) if constrain else u[0]

    # Define dynamics model as per Razvan V. Florian's
    # "Correct equations for the dynamics of the cart-pole system".
    # Friction is neglected.

    # Eq. (23)
    temp = (F + mp * l * theta_dot**2 * sin_theta) / (mc + mp)
    numerator = G * sin_theta - cos_theta * temp
    denominator = l * (4.0 / 3.0 - mp * cos_theta**2 / (mc + mp))
    theta_dot_dot = numerator / denominator

    # Eq. (24)
    x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

    # Deaugment state for dynamics.
    theta = sym.atan2(sin_theta, cos_theta)
    next_theta = theta + theta_dot * dt

    return sym.Matrix(
        [
            x_ + x_dot * dt,
            x_dot + x_dot_dot * dt,
            sym.sin(next_theta),
            sym.cos(next_theta),
            theta_dot + theta_dot_dot * dt,
        ]
    )


# instantenious cost
def g(x, u, x_goal):
    error = x - x_goal
    Q = np.eye(len(x))
    Q[1, 1] = Q[4, 4] = 0.0
    R = 0.1 * np.eye(len(u))
    return error.T @ Q @ error + u.T @ R @ u


# termination cost
def h(x, x_goal):
    error = x - x_goal
    Q = 100 * np.eye(len(x))
    return error.T @ Q @ error


# trajectory parameters
N = 100  # trajectory points
Nx = 5  # state dimension
Nu = 1  # control dimesions

# starting state
x0 = np.array([0.0, 0.0, np.sin(np.pi), np.cos(np.pi), 0.0])

# goal state we want to reach
x_goal = np.array([0.0, 0.0, np.sin(0.0), np.cos(0.0), 0.0])

print("Starting state", x0)
print("Goal state", x_goal)


def render(state, params):
    # dynamics parameters
    mp = 0.1
    mc = 1.0
    l = 1.0
    G = 9.80665
    dt = 0.05

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError:
        raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gymnasium[classic_control]`"
        )

    screen_width = 600
    screen_height = 400
    screen = None
    clock = params[1]
    isopen = True
    render_mode = "human"
    screen = params[0]
    if screen is None:
        pygame.init()
        if render_mode == "human":
            pygame.display.init()
            screen = pygame.display.set_mode(
                (screen_width, screen_height)
            )
        else:  # mode == "rgb_array"
            screen = pygame.Surface((screen_width, screen_height))
    if clock is None:
        clock = pygame.time.Clock()

    x_threshold = 2.4
    world_width = x_threshold * 10
    scale = screen_width / world_width
    polewidth = 10.0
    polelen = scale * (2 * l)
    cartwidth = 50.0
    cartheight = 30.0

    if state is None:
        return None

    x = state

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    axleoffset = cartheight / 4.0
    cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    carty = 100  # TOP OF CART
    cart_coords = [(l, b), (l, t), (r, t), (r, b)]
    cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
    gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
    gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

    l, r, t, b = (
        -polewidth / 2,
        polewidth / 2,
        polelen - polewidth / 2,
        -polewidth / 2,
    )

    pole_coords = []
    for coord in [(l, b), (l, t), (r, t), (r, b)]:
        coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
        coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        pole_coords.append(coord)
    gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
    gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

    gfxdraw.aacircle(
        surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )
    gfxdraw.filled_circle(
        surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )

    gfxdraw.hline(surf, 0, screen_width, carty, (0, 0, 0))

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    if render_mode == "human":
        pygame.event.pump()
        clock.tick(metadata["render_fps"])
        pygame.display.flip()

    elif render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )
    
    return [screen, clock]
    
for i in range(2):
    # Create and run optimizer with random intialization
    print("Starting optimization")
    start_time = time()
    ddp = DDPOptimizer(Nx, Nu, f, g, h)
    X, U, X_hist, U_hist, J_hist = ddp.optimize(x0, x_goal, N=N, full_output=True)
    print("Finished optimization in {:.2f}s".format(time() - start_time))
    
    params = [None, None]
    for t in range(len(X)):
        params = render(X[t], params=params)
    x0 = X[-1]

# plot results
if with_plots:
    print("Plotting results")

    fig, ax = plt.subplots(3, 1, figsize=(4, 8))
    tt = np.linspace(0, dt * N, N)
    theta_sol = np.unwrap(np.arctan2(X[:, 2], X[:, 3]))
    theta_dot_sol = X[:, 4]

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
