import functools
from turtledemo.penrose import start

from IPython.display import display

import jax
import jax.numpy as jnp
import numpy as np

from constants import BEAD_RADIUS

np.seterr(invalid="ignore")

import matplotlib.pyplot as plt; plt.rcParams.update({'font.size': 20})
import matplotlib.collections
import matplotlib.transforms
from ipywidgets import interact, interactive
import time
from typing import Callable, NamedTuple
from constants import *

class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A
    f_u: jnp.array  # B

    def __call__(self, x, u, k=None):
        f_x, f_u = self
        return f_x @ x + f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class AffinePolicy(NamedTuple):
    l: jnp.array  # l
    l_x: jnp.array  # L

    def __call__(self, x, k=None):
        l, l_x = self
        return l + l_x @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class QuadraticCost(NamedTuple):
    c: jnp.array  # c
    c_x: jnp.array  # q
    c_u: jnp.array  # r
    c_xx: jnp.array  # Q
    c_uu: jnp.array  # R
    c_ux: jnp.array  # H.T

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros((c_xx.shape[:-2])),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx,
            c_uu,
            c_ux,
        )

    def __call__(self, x, u, k=None):
        c, c_x, c_u, c_xx, c_uu, c_ux = self
        return c + c_x @ x + c_u @ u + x @ c_xx @ x / 2 + u @ c_uu @ u / 2 + u @ c_ux @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class QuadraticStateCost(NamedTuple):
    v: jnp.array  # p (scalar)
    v_x: jnp.array  # p (vector)
    v_xx: jnp.array  # P

    @classmethod
    def from_pure_quadratic(cls, v_xx):
        return cls(
            jnp.zeros(v_xx.shape[:-2]),
            jnp.zeros(v_xx.shape[:-1]),
            v_xx,
        )

    def __call__(self, x, k=None):
        v, v_x, v_xx = self
        return v + v_x @ x + x @ v_xx @ x / 2 if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


def rollout_state_feedback_policy(dynamics, policy, x0, step_range, x_nom=None, u_nom=None):
    def scan_fn(x, k):
        u = policy(x, k) if x_nom is None else u_nom[k] + policy(x - x_nom[k], k)
        x1 = dynamics(x, u, k)
        #jax.debug.print("u = {dist}",dist=u)
        #jax.debug.print("x1 = {dist}", dist = x1)
        return (x1, (x1, u))

    xs, us = jax.lax.scan(scan_fn, x0, step_range)[1]
    return jnp.concatenate([x0[None], xs]), us


def riccati_step(
    current_step_dynamics: LinearDynamics,
    current_step_cost: QuadraticCost,
    next_state_value: QuadraticStateCost,
):
    f_x, f_u = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    v, v_x, v_xx = next_state_value

    q = c + v
    q_x = c_x + f_x.T @ v_x
    q_u = c_u + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x
    st=time.time()
    l = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)
    ed=time.time()

    #jax.debug.print("Time {dist}", dist=ed-st)
    current_state_value = QuadraticStateCost(
        q - l.T @ q_uu @ l / 2,
        q_x - l_x.T @ q_uu @ l,
        q_xx - l_x.T @ q_uu @ l_x,
    )
    current_step_optimal_policy = AffinePolicy(l, l_x)
    return current_state_value, current_step_optimal_policy


def ensure_positive_definite(a, eps=1e-3):
    w, v = jnp.linalg.eigh(a)
    return (v * jnp.maximum(w, eps)) @ v.T


class TotalCost(NamedTuple):
    running_cost: Callable
    terminal_cost: Callable

    def __call__(self, xs, us):
        step_range = jnp.arange(us.shape[0])
        return jnp.sum(jax.vmap(self.running_cost)(xs[:-1], us, step_range)) + self.terminal_cost(xs[-1])


class EulerIntegrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using the Euler method."""
    ode: Callable
    dt: float

    @jax.jit
    def __call__(self, x, u, k):
        return x + self.dt * self.ode(x, u)


class RK4Integrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    @jax.jit
    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


@jax.jit
# Run every single MPC loop (Every time we plan over the horizon)
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=MAX_ILQR_ITERATIONS, atol=1e-2):
    st=time.time()
    running_cost, terminal_cost = total_cost
    n, (N, m) = x0.shape[-1], u_guess.shape  # Initial state and control inputs
    step_range = jnp.arange(N) # 0, 1, 2, 3,..., N - 1

    xs, us = rollout_state_feedback_policy(dynamics, lambda x, k: u_guess[k], x0, step_range)
    j = total_cost(xs, us)

    def continuation_criterion(loop_vars):
        i, _, _, j_curr, j_prev = loop_vars
        #jax.debug.print(f"Current={j_curr} Prev={j_prev}")
        return (j_curr < j_prev - atol) & (i < maxiter)

    def ilqr_iteration(loop_vars):
        i, xs, us, j_curr, j_prev = loop_vars

        f_x, f_u = jax.vmap(jax.jacobian(dynamics, (0, 1)))(xs[:-1], us, step_range)

        # put A and stack a bunch of times to get f_x
        # same with B

        # f_x and f_u are just a bunch of A and B matrices
        #jax.debug.print("state = {dist}", dist = f_x)
        c = jax.vmap(running_cost)(xs[:-1], us, step_range)
        c_x, c_u = jax.vmap(jax.grad(running_cost, (0, 1)))(xs[:-1], us, step_range)
        (c_xx, c_xu), (c_ux, c_uu) = jax.vmap(jax.hessian(running_cost, (0, 1)))(xs[:-1], us, step_range)
        v, v_x, v_xx = terminal_cost(xs[-1]), jax.grad(terminal_cost)(xs[-1]), jax.hessian(terminal_cost)(xs[-1])

        # Ensure quadratic cost terms are positive definite.
        c_zz = jnp.block([[c_xx, c_xu], [c_ux, c_uu]])
        c_zz = jax.vmap(ensure_positive_definite)(c_zz)
        c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]
        v_xx = ensure_positive_definite(v_xx)

        linearized_dynamics = LinearDynamics(f_x, f_u)
        quadratized_running_cost = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quadratized_terminal_cost = QuadraticStateCost(v, v_x, v_xx)

        def scan_fn(next_state_value, current_step_dynamics_cost):
            current_step_dynamics, current_step_cost = current_step_dynamics_cost
            current_state_value, current_step_policy = riccati_step(
                current_step_dynamics,
                current_step_cost,
                next_state_value,
            )
            return current_state_value, current_step_policy

        policy = jax.lax.scan(scan_fn,
                              quadratized_terminal_cost, (linearized_dynamics, quadratized_running_cost),
                              reverse=True)[1] # 0.001sec

        def rollout_linesearch_policy(alpha):
            # Note that we roll out the true `dynamics`, not the `linearized_dynamics`!
            l, l_x = policy
            return rollout_state_feedback_policy(dynamics, AffinePolicy(alpha * l, l_x), x0, step_range, xs, us)

        # Backtracking line search (step sizes evaluated in parallel).

        all_xs, all_us = jax.vmap(rollout_linesearch_policy)(0.5**jnp.arange(16))
        js = jax.vmap(total_cost)(all_xs, all_us)
        a = jnp.argmin(js)
        j = js[a]
        xs = jnp.where(j < j_curr, all_xs[a], xs)
        us = jnp.where(j < j_curr, all_us[a], us)

        return i + 1, xs, us, jnp.minimum(j, j_curr), j_curr

    st = time.time()
    i, xs, us, j, _ = jax.lax.while_loop(continuation_criterion, ilqr_iteration, (0, xs, us, j, jnp.inf))

    return {
        "optimal_trajectory": (xs, us),
        "optimal_cost": j,
        "num_iterations": i,
    }

class RK4IntegratorObs(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    def __call__(self, x, dt):
        k1 = self.dt * self.ode(x, dt)
        k2 = self.dt * self.ode(x + k1 / 2, dt)
        k3 = self.dt * self.ode(x + k2 / 2, dt)
        k4 = self.dt * self.ode(x + k3, dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

class ContinuousTimeBeadDynamics(NamedTuple):
    def __call__(self, state, control):
        """
        Full Dynamics: mẍ + ẋβ + k(x - u) = η(t) + F(t)
        x : postion of bead
        u : postion of trap

        m = 0 : (negligible bead mass)
        β : damping coefficient (~10e-8)
        k : spring constant (~10e-5)
        η(t) = 0 : thermal forces on bead (negligible)
        F(t) = 0 : external forces on bead (negligible)

        Simplified dynamics: ẋβ + k(x - u) = 0

        Solve for ẋ: ẋ = k(-x + u) / β
        """

        x, y = state  # x and y positions of the bead
        u_x, u_y = control  # x and y positions of the trap
        b = 10e-8  # 10e-8
        k = 10e-5  # 10e-5
        return jnp.array([
            (-k * x + k * u_x) / b,
            (-k * y + k * u_y) / b
        ])


class ContinuousTimeObstacleDynamics(NamedTuple):
    def __call__(self, state, dt):
        """
        Full Dynamics: ẍ = -gamma/m * v + chi/m * N(0,1) + F_ext / m

        gamma = 6πηr : drag coefficient from Stoke's law for a spherical bead
          η : viscosity of fluid
          r : radius of bead

        chi = √((2 * gamma * K_B * T) / dt) : scaling constant
          K_B = 1.3806e-23 J/K : Boltzmann constant
          T : temperature of fluid

        N(0,1) : Normal random variable

        m : bead mass
        F_ext = 0 : untrapped bead

        Simplified dynamics: ẍ = -gamma/m * v + chi/m * N(0,1)
        """

        v_x, v_y = state  # x and y velocities of the bead

        r = 5e-6 / 2  # radius of bead (m)
        v = (4 / 3) * np.pi * (r ** 3)  # volume of bead (m^3)
        d = 1.05e6  # density of bead (g/m^3)
        m = v * d * 1e-3  # mass of bead (kg)

        nu = 0.89e-3  # viscosity of water at 20°C or 293.15K (Pa*s)
        T = 293.15  # room temperature (K)
        K_B = 1.3806e-23  # Boltzmann constant (J/K)
        mu, sigma = 0, 1  # mean and standard deviation for N(0,1)
        N_randx = np.random.normal(mu, sigma)
        N_randy = np.random.normal(mu, sigma)

        gamma = 6 * np.pi * nu * r  # drag coefficient
        chi = np.sqrt((2 * gamma * K_B * T) / dt)  # scaling constant

        return jnp.array([
            (-gamma / m) * v_x + (chi / m) * N_randx,
            (-gamma / m) * v_y + (chi / m) * N_randy
        ])


class Asteroid(NamedTuple):
    center: jnp.array
    radius: jnp.array
    velocity: jnp.array

    def update(self, kps, num_beads):
        #bead_centers = jnp.reshape(kps, (num_beads, 2))
        updated_radius = 10 * jnp.ones(num_beads)
        return self._replace(center=kps, radius=updated_radius)
    

class Environment(NamedTuple):
    asteroids: Asteroid
    obj_bead_radius: float
    bubble_radius: float
    bounds: jnp.array

    @classmethod
    def create(cls, num_beads, kpsarray, obj_bead_radius=5.0, bubble_radius=5.0, bounds=(640, 480)):
        bounds = jnp.array(bounds)
        #if num_beads > 0:
            #jax.debug.print(f"f{num_beads}___{len(kpsarray)}")
        return cls(
            Asteroid(
                jnp.reshape(kpsarray, (num_beads, 2)),  # np.random.rand(num_beads, 2) * bounds
                BEAD_RADIUS*jnp.ones(num_beads),
                jnp.zeros(num_beads),
            ), obj_bead_radius, bubble_radius, bounds)

    def update(self, kps, num_beads):
        updated_beads = self.asteroids.update(kps, num_beads)

        return self._replace(asteroids=updated_beads)

    def wrap_vector(self, vector):
        return (vector + self.bounds / 2) % self.bounds - self.bounds / 2


class RunningCost(NamedTuple):
    env: Environment
    dt: jnp.array
    obstacle_separation = 26
    def __call__(self, state, control, step):
        # NOTE: many parameters (gains, offsets) in this function could be lifted to fields of `RunningCost`, in which
        # case you could experiment with changing these parameters without incurring `jax.jit` recompilation.
        asteroids = self.env.asteroids
        separation = RunningCost.obstacle_separation
        #jax.debug.print("state = {dist}", dist = asteroids)
        #jax.debug.print("center = {dist}", dist = asteroids.center)
        separation_distance = jnp.linalg.norm(self.env.wrap_vector(state - asteroids.center), axis=-1) - asteroids.radius - self.env.obj_bead_radius
        #total_separation = 1e-5 * jnp.sum(separation_distance)**2
        collision_avoidance_penalty = jnp.sum(jnp.where(separation_distance > separation, 0, 1e2 * (self.obstacle_separation - separation_distance) ** 2))

        soft_avoidance_penalty = jnp.sum(jnp.where(separation_distance > 2 * separation, 0,1e2))

        #collision_penalty = jnp.sum(
        #    jnp.where(separation_distance > 2, 0, 1e5))
        u_x, u_y = control
        x_dist = 1e3 * (state[0] - u_x) ** 2 #1e3
        y_dist = 1e3 * (state[1] - u_y) ** 2

        max_move_x = jnp.maximum(jnp.abs(state[0] - u_x) - 2, 0)**2
        max_move_y = jnp.maximum(jnp.abs(state[1] - u_y) - 2, 0)**2

        max_move = (max_move_x + max_move_y)*1e3
        #min_move = jnp.linalg.norm(state - jnp.array(control))

        # how to allow for backtracking? balance cost of getting near object/ramming through it with cost of taking a longer route
        #minimize sum of distances away from beads
        #jax.debug.print(f"{step.val}")
        #
        return collision_avoidance_penalty + x_dist + y_dist + max_move


class MPCTerminalCost(NamedTuple):
    env: Environment
    goal_position: jnp.array

    @classmethod
    def create_ignoring_extra_args(cls, env, goal_position, *args, **kwargs):
        return cls(env, goal_position)

    def __call__(self, state):
        distance_to_goal = jnp.linalg.norm(state[:2] - self.goal_position)
        #goal_penalty = jnp.where(distance_to_goal > 50,  2 * (distance_to_goal - 50), 5e4 * distance_to_goal ** 2)
        goal_penalty = jnp.where(distance_to_goal > 25, 2 * (distance_to_goal - 25), distance_to_goal ** 2)
        #goal_penalty = distance_to_goal ** 2
        return 1e3 * goal_penalty
        #return 1000 * jnp.sum(jnp.square(state[:2] - self.goal_position))

# class RunningCost(NamedTuple):
#     env: Environment
#     dt: jnp.array
#
#     def __call__(self, state, control, step):
#         # NOTE: many parameters (gains, offsets) in this function could be lifted to fields of `RunningCost`, in which
#         # case you could experiment with changing these parameters without incurring `jax.jit` recompilation.
#         asteroids = self.env.asteroids
#         separation_distance = jnp.linalg.norm(self.env.wrap_vector(state - asteroids.center), axis=-1) - asteroids.radius - self.env.obj_bead_radius
#         total_separation = 1e-3 * jnp.sum(separation_distance**2)
#
#         #soft_avoidance_penalty = jnp.sum(jnp.where(separation_distance > 2*OBSTACLE_SEPARATION, 0,  1e2 * (OBSTACLE_SEPARATION - separation_distance) ** 2))
#
#         hard_avoidance_penalty = jnp.sum(jnp.where(separation_distance > 1.75*OBSTACLE_SEPARATION, 0, 1e3 * (OBSTACLE_SEPARATION - separation_distance) ** 2))
#
#         u_x, u_y = control
#         x_dist = 3e3 * (state[0] - u_x) ** 2 #1e3
#         y_dist = 3e3 * (state[1] - u_y) ** 2
#
#         #min_move = jnp.where(jnp.abs(state[0] - u_x) + jnp.abs(state[1] - u_y) < 1, 1e4, 0)
#
#         total_cost = x_dist + y_dist + hard_avoidance_penalty
#
#         #jax.debug.print("{}", total_cost)
#
#         return total_cost
#
#
# class MPCTerminalCost(NamedTuple):
#     env: Environment
#     goal_position: jnp.array
#
#     @classmethod
#     def create_ignoring_extra_args(cls, env, goal_position, *args, **kwargs):
#         return cls(env, goal_position)
#
#     def __call__(self, state):
#         distance_to_goal = jnp.linalg.norm(state[:2] - self.goal_position)
#         #goal_penalty = jnp.where(distance_to_goal > 50,  2 * (distance_to_goal - 50), 5e4 * distance_to_goal ** 2)
#         goal_penalty = jnp.where(distance_to_goal > 25, 2 * (distance_to_goal - 25), distance_to_goal ** 2)
#        # goal_penalty = distance_to_goal ** 2
#         #goal_penalty = jnp.where(distance_to_goal > 25, distance_to_goal ** 2, distance_to_goal ** 10)
#
#         return 1e4 * goal_penalty
#         #return 1000 * jnp.sum(jnp.square(state[:2] - self.goal_position))

# class MPCTerminalCost(NamedTuple):
#     env: Environment
#     goal_position: jnp.array
#
#     @classmethod
#     def create_ignoring_extra_args(cls, env, goal_position, *args, **kwargs):
#         return cls(env, goal_position)
#
#     def __call__(self, state):
#         distance_to_goal = jnp.linalg.norm(state[:2] - self.goal_position)
#
#         far_goal_penalty = jnp.where(distance_to_goal > 75, (distance_to_goal - 75), 0)
#
#         near_goal_penalty = jnp.where(distance_to_goal <= 75,
#                                       distance_to_goal ** 2,
#                                       0)
#
#         goal_penalty = far_goal_penalty + near_goal_penalty
#
#         return 1e3 * goal_penalty

class FullHorizonTerminalCost(NamedTuple):
    env: Environment
    goal_position: jnp.array

    @classmethod
    def create_ignoring_extra_args(cls, env, goal_position, *args, **kwargs):
        return cls(env, goal_position)

    def __call__(self, state):
        return 10000 * jnp.sum(jnp.square(state[:2] - self.goal_position))


def gen_initial_traj(start_state, goal_state, N):
    xs, ys = start_state
    xg, yg = goal_state

    x_traj = jnp.linspace(xs, xg, N)
    y_traj = jnp.linspace(ys, yg, N)

    traj = jnp.array([x_traj.T, y_traj.T])
    return traj


@functools.partial(jax.jit, static_argnames=["running_cost_type", "terminal_cost_type", "limited_sensing", "N", "dynamics"])
def policy(state, goal_position, u_guess, env, dynamics, running_cost_type, terminal_cost_type, empty_env, limited_sensing=False, N=20):
    solution = iterative_linear_quadratic_regulator(
        dynamics,
        TotalCost(
            running_cost_type(empty_env, dynamics.dt),
            terminal_cost_type.create_ignoring_extra_args(
                empty_env,
                goal_position,
                state[:2],
                empty_env.bubble_radius,
            ),
        ),
        state,
        u_guess,
    )
    solution = iterative_linear_quadratic_regulator(
        dynamics,
        TotalCost(
            running_cost_type(env, dynamics.dt),
            terminal_cost_type.create_ignoring_extra_args(
                env,
                goal_position,
                state[:2],
                env.bubble_radius,
            ),
        ),
        state,
        solution["optimal_trajectory"][1],
    )
    states, controls = solution["optimal_trajectory"]

    return solution


def simulate_mpc(start_state, goal_position, u_guess, env, dynamics, running_cost_type, terminal_cost_type, limited_sensing=False, N=20, T=1250):
    states = [start_state]
    controls = []
    plans = []
    for t in range(T):
        control, (mpc_states, mpc_controls) = policy(states[-1], goal_position, u_guess, env.at_time(t * dynamics.dt), dynamics,
                                                     running_cost_type, terminal_cost_type, limited_sensing, N)
        states.append(mpc_states[1])
        controls.append(control)
        plans.append(mpc_states)
        #print(mpc_states)
    states = np.array(states)
    controls = np.array(controls)
    plans = np.array(plans)

    return states, controls
