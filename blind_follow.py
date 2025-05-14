import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from collections import deque

# Constants
DT = 0.1  # seconds
MAX_LEADER_SPEED = 20.0  # m/s
MAX_SPEED = 2 * MAX_LEADER_SPEED  # m/s
MAX_STEER = np.radians(20)  # delta
WHEELBASE = 2.0  # meters
JOYSTICK_DEADBAND = 0.1
NUM_VEHICLES = 2

FOLLOW_DIST = 10.0  # meters
FOLLOW_TIME = 10.0  # seconds
ANGLE_KP = 0.5  # (delta)/deg
ANGLE_THRESHOLD = np.radians(30)  # delta
ANGLE_CORRECTION_FACTOR = 1.0  # (delta)/deg
DIST_KP = 0.5  # (m/s)/m

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
    return forward * MAX_LEADER_SPEED, steer * MAX_STEER

# --- Initialization ---

# Initial states
vehicle_poses = [np.array([-i * FOLLOW_DIST, -1.0, 0.0]) for i in range(NUM_VEHICLES)]
vehicle_poses[0] = np.array([0.0, 0.0, 0.0])  # Leader at origin

# Commanded velocities
v_cmd = [0.0] * NUM_VEHICLES
delta_cmd = [0.0] * NUM_VEHICLES

# Queues for storing relative thetas
theta_queues = [deque(maxlen=FOLLOW_TIME/DT).extend([0] * (FOLLOW_TIME / DT)) for _ in range(NUM_VEHICLES)]

# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)

# Draw leader and follower arrows, lookahead markers, and pursuit arcs
vehicle_arrows = []
debug_markers = []
colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx in range(NUM_VEHICLES):
    debug_markers.append(ax.plot([], [], marker='x', color=colors[idx], markersize=6, alpha=0.5)[0])
    vehicle_arrows.append(FancyArrowPatch((0, 0), (1, 0), color=colors[idx], mutation_scale=15, arrowstyle='->'))
    ax.add_patch(vehicle_arrows[-1])

leader_trail = []
MAX_TRAIL_POINTS = 1.5 * MAX_LEADER_SPEED * NUM_VEHICLES / DT
leader_trail_line, = ax.plot([], [], 'k', lw=1.5, alpha=0.3)

# --- Animation loop ---
def animate(i):
    global vehicle_poses

    # Leader Control
    v_cmd[0], delta_cmd[0] = get_controller_input()

    # Simulate measured relative poses
    rel_poses = [relative_pose(vehicle_poses[idx], vehicle_poses[idx-1]) for idx in range(1, NUM_VEHICLES)]

    # --- Follower Control ---
    
    for idx in range(1, NUM_VEHICLES):
        # Get the relative pose of the leader from the follower's perspective
        leader_rel_pose = np.sum(rel_poses[:idx-1], axis=0)
        
        # Proportional control on following distance for forward velocity
        dist_to_next = np.linalg.norm(rel_poses[idx-1][:2])
        v_cmd[idx] = clamp(v_cmd[0] + DIST_KP * (dist_to_next - FOLLOW_DIST), 0, MAX_SPEED)
        
        # Save queue of theta relative to leader and pop the oldest (i.e. theta from before FOLLOW_TIME)
        theta_queues[idx].append(leader_rel_pose[2])
        theta_delay = theta_queues[idx].popleft()
        
        # Calculate turning speed as a combination of proportional control w.r.t. the relative delayed angle and a correction term if the current relative angle is too large
        proportional = ANGLE_KP * (leader_rel_pose[2] - theta_delay)
        leader_diff = leader_rel_pose[2] - 
        correction = ANGLE_CORRECTION_FACTOR * (correction - ) if 
        delta_cmd[idx] = clamp(leader_rel_pose[2] - theta_delay, -ANGLE_THRESHOLD, ANGLE_THRESHOLD)

    # --- Update animation ---
    
    for idx in range(NUM_VEHICLES):
        vehicle_poses[idx] = update_pose(vehicle_poses[idx], v_cmd[idx], delta_cmd[idx])
        vehicle_arrows[idx].set_positions(vehicle_poses[idx][:2], vehicle_poses[idx][:2] + np.array([1, 0]) @ rot)
        debug_markers[idx].set_data(vehicle_poses[idx][0], vehicle_poses[idx][1])
    
    leader_trail.append(vehicle_poses[0][:2])
    if len(leader_trail) > MAX_TRAIL_POINTS:
        leader_trail.pop(0)
    trail_array = np.array(leader_trail)
    leader_trail_line.set_data(trail_array[:, 0], trail_array[:, 1])

    return [*vehicle_arrows, *debug_markers, leader_trail_line]



ani = FuncAnimation(fig, animate, interval=100)
plt.show()
