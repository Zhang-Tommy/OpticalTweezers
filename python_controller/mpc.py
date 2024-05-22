"""
MPC Controller Implementation
States: x,y,xdot,ydot
Input: x_control,y_control
Output: x_actual, y_actual
"""


"""
What do we need to have?
States: x,y from image processing. xdot, ydot from delayed calculation from two frames?

obstacles -> key_points (detected beads minus target beads)

Quadratic cost function 

Determine discrete numerical integration method (Euler, Runge Kutta)

Determine Problem parameters (Time to target, dt, start position, goal position)
"""


import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.collections
from typing import Callable, NamedTuple
import collections

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
        b = 10e-8
        k = 10e-5
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




# Linear dynamics inherets from NamedTuple providing immutability and field access
class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A
    f_u: jnp.array  # B

    def __call__(self, x, u, k=None):
        f_x, f_u = self
        return f_x @ x + f_u @ u if k is None else self[k](x, u)  # Ax + Bu

class EulerIntegrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using the Euler method."""
    ode: Callable
    dt: float

    @jax.jit
    def __call__(self, x, u):
        return x + self.dt * self.ode(x, u)

class RK4Integrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    #@jax.jit  # optimize using just-in-time compilation
    def __call__(self, x, u):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def controller(start_state, goal_position, u_guess, env, T, dt):
    """ Controller for obstacle avoidance and path planning """
    # What does controller need to know?
    # State of system (coordinates of all obstacles, trapped beads)
    # Outputs: control x,y for specific bead
    dynamics = RK4Integrator(ContinuousTimeBeadDynamics(), dt)
    states, controls = simulate_mpc(start_state, goal_position, u_guess, env, dynamics, RunningCost,
                                    FullHorizonTerminalCost, False, 20, T)
    return states


def simulator_mpc(sm):
    traps = list(sm.trapped_beads.keys())

    trap_parent.send(traps)
    if kp_child.poll():
        kps = kp_child.recv()

    num_beads = len(kps)

    x = float(round(kps[0][0]))
    y = float(round(kps[0][1]))

    sm.add_spot((int(x), int(y)))

    T = 7500

    start_state = jnp.array([x, y])
    print(start_state)
    goal_position = jnp.array([0, 0])

    u_guess = gen_intial_traj(start_state, goal_position, 20).T

    dt = 1e-6
    kpsarray = jnp.asarray(kps)
    # kpsarray = kpsarray.at[:, 1].set(480 - kpsarray[:, 1])

    # print(kpsarray)
    env = Environment.create(num_beads, kpsarray)

    states = controller(start_state, goal_position, u_guess, env, T, dt)

    for k in range(T - 1):
        traps = list(sm.trapped_beads.keys())

        if k % 10 == 1:
            trap_parent.send(traps)

        if kp_child.poll():
            kps = kp_child.recv()

        x_curr = int(states[k][0])
        y_curr = int(states[k][1])

        x_next = int(states[k + 1][0])
        y_next = int(states[k + 1][1])

        sm.move_trap((x_curr, y_curr), (x_next, y_next))

        time.sleep(0.01)