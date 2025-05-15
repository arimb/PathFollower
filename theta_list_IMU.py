import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from collections import deque

# Constants
DT = 0.1  # seconds
MAX_LEADER_SPEED = 10.0  # m/s
MAX_SPEED = 2 * MAX_LEADER_SPEED  # m/s
MAX_LEADER_STEER = np.radians(5)  # delta
MAX_STEER = MAX_LEADER_STEER * 2  # delta
WHEELBASE = 2.0  # meters
GRAPH_LIMS = 150
JOYSTICK_DEADBAND = 0.1
NUM_VEHICLES = 5

FOLLOW_DIST = 10.0  # meters
FOLLOW_TIME = FOLLOW_DIST / MAX_LEADER_SPEED  # seconds
DIST_KP = 5  # (m/s)/m
ANGLE_KP = 1.5  # (delta)/rad
STEERING_KP = 0.5  # (delta)/rad

# Initialize Pygame for controller input
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# --- Helper functions ---

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

# --- Vehicle dynamics and control functions ---


def update_pose(pose, v, delta, dt=DT):
    x, y, theta = pose
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += v / WHEELBASE * np.tan(delta) * dt
    return np.array([x, y, theta])

def relative_pose(follower_pose, leader_pose):
    """Calculate the relative pose of the leader with respect to the follower.
    Args:
        follower_pose (np.array): The pose of the follower [x, y, theta].
        leader_pose (np.array): The pose of the leader [x, y, theta].
    Returns:
        np.array: The relative pose of the leader with respect to the follower [x_rel, y_rel, theta_rel]."""
    dx = leader_pose[0] - follower_pose[0]
    dy = leader_pose[1] - follower_pose[1]
    dtheta = leader_pose[2] - follower_pose[2]
    rot = np.array([
        [np.cos(-follower_pose[2]), -np.sin(-follower_pose[2])],
        [np.sin(-follower_pose[2]),  np.cos(-follower_pose[2])]
    ])
    rel_pos = rot @ np.array([dx, dy])
    return np.array([rel_pos[0], rel_pos[1], dtheta])


# --- Drawing functions ---

def update_arrow(arrow, pose, length=1.5):
    x, y, theta = pose
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    arrow.set_positions((x, y), (x + dx, y + dy))

def get_controller_input():
    pygame.event.pump()
    forward = (joystick.get_axis(5) + 1) / 2
    steer = -joystick.get_axis(0)
    return forward * MAX_LEADER_SPEED, steer * MAX_LEADER_STEER

# --- Initialization ---

# Initial states
vehicle_poses = [np.array([-i * 1.5 * FOLLOW_DIST, -1.0, 0.0]) for i in range(NUM_VEHICLES)]
vehicle_poses[0] = np.array([0.0, 0.0, 0.0])  # Leader at origin

# Commanded velocities
v_cmd = [0.0] * NUM_VEHICLES
delta_cmd = [0.0] * NUM_VEHICLES

# Queues for storing relative thetas
abs_theta_queues = []
for i in range(1, NUM_VEHICLES):
    abs_theta_queues.append(deque(maxlen=None))
    abs_theta_queues[-1].extend([vehicle_poses[0][2]] * int(FOLLOW_TIME / DT * i))


# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-GRAPH_LIMS, GRAPH_LIMS)
ax.set_ylim(-GRAPH_LIMS, GRAPH_LIMS)

# Draw leader and follower arrows, lookahead markers, and pursuit arcs
vehicle_arrows = []
colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx in range(NUM_VEHICLES):
    vehicle_arrows.append(FancyArrowPatch((0, 0), (1, 0), color=colors[idx], mutation_scale=15, arrowstyle='->'))
    ax.add_patch(vehicle_arrows[-1])

leader_trail = deque(maxlen = int(1.5 * FOLLOW_DIST * NUM_VEHICLES / MAX_LEADER_SPEED / DT))
leader_trail_line, = ax.plot([], [], 'k', lw=1.5, alpha=0.3)

# --- Animation loop ---
def animate(i):
    global vehicle_poses, abs_theta_queues, v_cmd, delta_cmd

    # Leader Control
    v_cmd[0], delta_cmd[0] = get_controller_input()

    # Simulate measurements
    rel_poses = np.array([relative_pose(vehicle_poses[idx], vehicle_poses[idx-1]) for idx in range(1, NUM_VEHICLES)])
    first_abs_heading = vehicle_poses[0][2]

    # --- Follower Control ---
    # From here on only use the simulated measurements
    
    for idx in range(1, NUM_VEHICLES):
        # Calculate the follower's relative pose w.r.t. the first vehicle, and its absolute heading
        follower_rel_to_first  = -np.sum(rel_poses[:idx], axis=0)
        follower_abs_heading = first_abs_heading + follower_rel_to_first[2]
        
        # Save queue of the first vehicle's absolute heading and pop the oldest (i.e. theta from before FOLLOW_TIME*idx)
        abs_theta_queues[idx-1].append(first_abs_heading)
        abs_theta_delay = abs_theta_queues[idx-1].popleft()
        
        # Proportional control on following distance for forward velocity
        dist_to_next = np.linalg.norm(rel_poses[idx-1][:2])
        v_cmd[idx] = clamp(v_cmd[0] + DIST_KP * (dist_to_next - FOLLOW_DIST), 0, MAX_SPEED)
        
        # Calculate turning speed as a combination of proportional control w.r.t. the delayed angle and a steering term to keep the follower aligned with the leader
        proportional = ANGLE_KP * (abs_theta_delay - follower_abs_heading)
        steering = STEERING_KP * np.arctan2(rel_poses[idx-1][1], rel_poses[idx-1][0])
        delta_cmd[idx] = clamp(proportional + steering, -MAX_STEER, MAX_STEER)

    # --- Update animation ---
    for idx in range(NUM_VEHICLES):
        vehicle_poses[idx] = update_pose(vehicle_poses[idx], v_cmd[idx], delta_cmd[idx])
        update_arrow(vehicle_arrows[idx], vehicle_poses[idx])
    
    leader_trail.append(vehicle_poses[0][:2])
    trail_array = np.array(leader_trail)
    leader_trail_line.set_data(trail_array[:, 0], trail_array[:, 1])

    return [*vehicle_arrows, leader_trail_line]



ani = FuncAnimation(fig, animate, interval=100)
plt.show()
