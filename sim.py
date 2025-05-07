import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# Constants
DT = 0.1
MAX_SPEED = 2.0
MAX_STEER = np.radians(30)
WHEELBASE = 1.0
DEADZONE = 0.1
MIN_FOLLOW_DIST = 2.0  # meters


# Initialize Pygame for controller input
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

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

def pure_pursuit_control(target, k=1.5):
    x, y, theta = target
    Ld = np.hypot(x, y)
    alpha = np.arctan2(y, x)
    delta = np.arctan2(2 * WHEELBASE * np.sin(alpha), Ld)
    return clamp(delta, -MAX_STEER, MAX_STEER)

def bezier_curve(p0, p1, p2, p3, num_points=50):
    t = np.linspace(0, 1, num_points)
    t = t[:, None]
    curve = (
        (1 - t)**3 * p0 +
        3 * (1 - t)**2 * t * p1 +
        3 * (1 - t) * t**2 * p2 +
        t**3 * p3
    )
    return curve

def update_arrow(arrow, pose, length=1.5):
    x, y, theta = pose
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    arrow.set_positions((x, y), (x + dx, y + dy))

# Initial states
leader_pose = np.array([0.0, 0.0, 0.0])
follower_pose = np.array([-5.0, -1.0, 0.0])

# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
leader_arrow = FancyArrowPatch((0, 0), (1, 0), color='red', mutation_scale=15, arrowstyle='->')
ax.add_patch(leader_arrow)
follower_arrow = FancyArrowPatch((0, 0), (1, 0), color='blue', mutation_scale=15, arrowstyle='->')
ax.add_patch(follower_arrow)
spline_line, = ax.plot([], [], 'g-', lw=1.5, label='Spline Trajectory')
leader_zone = plt.Circle((0, 0), MIN_FOLLOW_DIST, color='gray', fill=False, linestyle='--', alpha=0.3)
ax.add_patch(leader_zone)

def get_controller_input():
    pygame.event.pump()
    forward = -joystick.get_axis(1)
    forward = clamp(forward, 0, 1)

    left_trigger = (joystick.get_axis(5) + 1) / 2
    right_trigger = (joystick.get_axis(4) + 1) / 2

    steer = right_trigger - left_trigger
    steer = clamp(steer, -1, 1)

    return forward * MAX_SPEED, steer * MAX_STEER

def apply_deadzone(val, threshold=DEADZONE):
    return val if abs(val) > threshold else 0.0

def animate(i):
    global leader_pose, follower_pose

    # --- LEADER CONTROL ---
    v_leader, delta_leader = get_controller_input()
    leader_pose = update_pose(leader_pose, v_leader, delta_leader)

    # --- FOLLOWER CONTROL ---
    rel = relative_pose(follower_pose, leader_pose)
    dist_to_leader = np.hypot(rel[0], rel[1])

    if dist_to_leader > MIN_FOLLOW_DIST:
        delta_follower = pure_pursuit_control(rel)
        follower_pose = update_pose(follower_pose, MAX_SPEED * 0.9, delta_follower)
    # else: stay still

    # --- PLOTTING ---
    update_arrow(leader_arrow, leader_pose)
    update_arrow(follower_arrow, follower_pose)

    # Show target location (for reference)
    rot = np.array([
        [np.cos(follower_pose[2]), -np.sin(follower_pose[2])],
        [np.sin(follower_pose[2]),  np.cos(follower_pose[2])]
    ])
    target_global = rot @ rel[:2] + follower_pose[:2]

    # Spline visualization
    theta_goal = follower_pose[2] + rel[2]
    p0 = follower_pose[:2]
    p3 = target_global
    p1 = p0 + 1.5 * np.array([np.cos(follower_pose[2]), np.sin(follower_pose[2])])
    p2 = p3 - 1.5 * np.array([np.cos(theta_goal), np.sin(theta_goal)])
    curve = bezier_curve(p0, p1, p2, p3)
    spline_line.set_data(curve[:, 0], curve[:, 1])
    leader_zone.center = (leader_pose[0], leader_pose[1])

    return leader_arrow, follower_arrow, spline_line

ani = FuncAnimation(fig, animate, interval=100)
plt.show()
