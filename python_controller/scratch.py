import matplotlib.pyplot as plt
import scipy
import numpy as np
import mpc


x0 = np.array([0, 0]).T

u0 = np.array([0, 0]).T

dynamics = mpc.ContinuousTimeObstacleDynamics()
# Steve Brunton has the solution

dt = 0.000001  # 100 pts/second

T = 500  # 50 pts

# 50pts 1 um/pt
velocity = [x0]
position = []

v_next = 0
x_next = 0
for k in range(T):
    a_next = mpc.RK4IntegratorObs(dynamics, dt)(velocity[k], dt)
    v_next = dt*a_next + v_next
    x_next = dt*v_next + x_next
    velocity.append(v_next)
    position.append(x_next)

velocity = np.array(velocity)
position = np.array(position)

plt.plot(np.arange(0, len(position)), position)
plt.plot(np.arange(0, len(velocity)), velocity)
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