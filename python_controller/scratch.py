import matplotlib.pyplot as plt
import scipy
import numpy as np
import mpc


x0 = np.array([0, 0]).T

u0 = np.array([0, 0]).T

dynamics = mpc.ContinuousTimeObstacleDynamics()
dt = 1  # 100 pts/second
#mpc.EulerIntegrator(dynamics, 0.1)

T = 5000  # 50 pts

# 50pts 1 um/pt
states = [x0]
#np.zeros((2,1))
controls = []

for i in range(T):
    controls.append(np.array([5, i * 1e-6]).T)

for k in range(T):
    x_next = mpc.RK4IntegratorObs(dynamics, dt)(states[k], dt)
    states.append(x_next)

print(states)
print(controls)

states = np.array(states)
controls = np.array(controls)

# Plotting states
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(states[:, 0], label='Position (x)')
#plt.plot(states[:, 1], label='Position (u_x)')
plt.title('States Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Plotting controls
plt.subplot(2, 1, 2)
#plt.plot(controls[:, 1], label='Position (y)')
plt.plot(states[:, 1], label='Position (u_y)')
plt.title('Controls Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


































def discretize_circle(center, radius, num_points):
    pts = []
    x0, y0 = center

    for i in range(1, num_points + 1):
        xi = round(x0 + radius * (1 / np.cos(np.pi / num_points)) * np.cos((2 * i - 1) * (np.pi / num_points)), 2)
        yi = round(y0 + radius * (1 / np.cos(np.pi / num_points)) * np.sin((2 * i - 1) * (np.pi / num_points)), 2)

        pts.append((xi, yi))
    print(pts)
    return pts

def test_discretize_circle():
    points = discretize_circle((5,5),10, 12)

    hull = scipy.ConvexHull(np.array(points))

    for simplex in hull.simplices:
        plt.plot([points[simplex[0]][0], points[simplex[1]][0]], [points[simplex[0]][1], points[simplex[1]][1]], 'k-')

    print(hull.equations)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of 2D Point Outside Convex Hull')

    plt.legend()

    plt.grid(True)
    plt.show()