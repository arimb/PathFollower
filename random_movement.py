import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import random
import statsmodels.api as sm

# Constants
DT = 0.1
MAX_SPEED = 2.0
MAX_STEER = np.radians(20)
WHEELBASE = 1.0
NUM_VEHICLES = 5

SENSOR_WEIGHT = 10
ODOMETRY_WEIGHT = 1


# --- Helper functions ---

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


# --- Vehicle dynamics and control functions ---

def update_pose(pose, v, omega, dt=DT):
    x, y, theta = pose
    v = clamp(v, -MAX_SPEED, MAX_SPEED)
    omega = clamp(omega, -MAX_STEER, MAX_STEER)
    return (
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + omega * dt
    )

def relative_pose(follower_pose, leader_pose):
    dx = leader_pose[0] - follower_pose[0]
    dy = leader_pose[1] - follower_pose[1]
    dtheta = leader_pose[2] - follower_pose[2]
    rot = np.array([
        [np.cos(-follower_pose[2]), -np.sin(-follower_pose[2])],
        [np.sin(-follower_pose[2]), np.cos(-follower_pose[2])]
    ])
    rel_pos = rot @ np.array([dx, dy])
    return np.array([rel_pos[0], rel_pos[1], dtheta])

def odometry_update(last_pose, v, omega, dt=DT):
    x, y, theta = last_pose
    return (
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + omega * dt
    )

# --- Drawing functions ---

def update_arrow(arrow, pose, length=1.5):
    x, y, theta = pose
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    arrow.set_positions((x, y), (x + dx, y + dy))



# --- Initialization ---

# Initial states
vehicle_poses = [np.array([-i * 3.0, -1.0, 0.0]) for i in range(NUM_VEHICLES)]
vehicle_poses[0] = np.array([0.0, 0.0, 0.0])  # Leader at origin

# Global pose estimates
# pose_estimates = [np.zeros(3) for _ in range(NUM_VEHICLES)]
pose_estimates = vehicle_poses - vehicle_poses[0]  # Initialize with relative positions


# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)

# Draw leader and follower arrows, lookahead markers, and pursuit arcs
vehicle_arrows = []
debug_markers = []
colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx in range(NUM_VEHICLES):
    debug_markers.append(ax.plot([], [], marker='x', color=colors[idx], markersize=6, alpha=0.5)[0])
    vehicle_arrows.append(FancyArrowPatch((0, 0), (1, 0), color=colors[idx], mutation_scale=15, arrowstyle='->'))
    ax.add_patch(vehicle_arrows[-1])


# --- Animation loop ---
def animate(i):
    global vehicle_poses, pose_estimates

    # --- Vehicle Movement ---
    for idx in range(NUM_VEHICLES):
        v = MAX_SPEED * random.uniform(0.0, 1.0)
        omega = MAX_STEER * random.uniform(-1.0, 1.0)
        vehicle_poses[idx] = update_pose(vehicle_poses[idx], v, omega)

    # --- Global Pose Estimate ---
    rel_poses = [relative_pose(vehicle_poses[idx], vehicle_poses[idx-1]) for idx in range(1, NUM_VEHICLES)]
    odometry_local = [odometry_update(pose_estimates[idx], v, omega) for idx in range(NUM_VEHICLES)]
    
    A = np.zeros((3*(2*NUM_VEHICLES-1), 3*NUM_VEHICLES))
    b = np.zeros((3*(2*NUM_VEHICLES-1), 1))
    w = np.zeros((3*(2*NUM_VEHICLES-1), 1))
    
    # First half of the matrix equation is from the sensor data, x_i - x_(i+1) = p_i
    for idx in range(NUM_VEHICLES-1):
        A[idx*3:(idx+1)*3, idx*3:(idx+1)*3] += np.eye(3)
        A[idx*3:(idx+1)*3, (idx+1)*3:(idx+2)*3] -= np.eye(3)
        b[idx*3:(idx+1)*3, 0] += rel_poses[idx]
        w[idx*3:(idx+1)*3, 0] += [SENSOR_WEIGHT] * 3
    
    # Second half of the matrix equation is from the odometry data, x_i = o_i
    for idx in range(NUM_VEHICLES):
        A[(NUM_VEHICLES-1+idx)*3:(NUM_VEHICLES+idx)*3, idx*3:(idx+1)*3] += np.eye(3)
        b[(NUM_VEHICLES-1+idx)*3:(NUM_VEHICLES+idx)*3, 0] += odometry_local[idx]
        w[(NUM_VEHICLES-1+idx)*3:(NUM_VEHICLES+idx)*3, 0] += [ODOMETRY_WEIGHT] * 3

    # Solve the linear system
    model = sm.WLS(b, A, weights=w)
    results = model.fit().params
    pose_estimates = [row.copy() for row in results.reshape((NUM_VEHICLES, 3))]
    
    # --- Update visualization ---
    for idx in range(NUM_VEHICLES):
        update_arrow(vehicle_arrows[idx], vehicle_poses[idx])
        debug_markers[idx].set_data([pose_estimates[idx][0]], [pose_estimates[idx][1]])
    
    return [*vehicle_arrows, *debug_markers]


ani = FuncAnimation(fig, animate, interval=100)
plt.show()
