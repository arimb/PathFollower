import pygame
import numpy as np
import matplotlib.pyplot as plt
from Tools.scripts.generate_re_casefix import alpha
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# Constants
DT = 0.1
MAX_LEADER_SPEED = 2.0
MAX_SPEED = 2 * MAX_LEADER_SPEED
MAX_STEER = np.radians(20)
WHEELBASE = 1.0
DEADZONE = 0.1
MIN_FOLLOW_DIST = 3.0
NUM_FOLLOWERS = 4

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

def calc_lookahead(rel, lookahead_dist=MIN_FOLLOW_DIST):
    lx = rel[0] + lookahead_dist * np.cos(rel[2])
    ly = rel[1] + lookahead_dist * np.sin(rel[2])
    return np.array([lx, ly, rel[2]])

def pure_pursuit_control(target):
    x, y, _ = target
    Ld = np.hypot(x, y)
    alpha = np.arctan2(y, x)
    delta = np.arctan2(2 * WHEELBASE * np.sin(alpha), Ld)
    return clamp(delta, -MAX_STEER, MAX_STEER)

def update_arrow(arrow, pose, length=1.5):
    x, y, theta = pose
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    arrow.set_positions((x, y), (x + dx, y + dy))

# Initial states
leader_pose = np.array([0.0, 0.0, 0.0])
follower_poses = [np.array([-(i+1) * MIN_FOLLOW_DIST, -1.0, 0.0]) for i in range(NUM_FOLLOWERS)]

# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)

# Leader zone
leader_zone = plt.Circle((0, 0), MIN_FOLLOW_DIST, color='gray', fill=False, linestyle='--', alpha=0.3)
ax.add_patch(leader_zone)

# Draw leader and follower arrows

follower_arrows = []
lookahead_markers = []
arc_lines = []
colors = ['blue', 'green', 'purple', 'orange']
for c in colors:
    arc_lines.append(ax.plot([], [], linestyle=':', color=c, lw=1.5, alpha=0.5)[0])
    lookahead_markers.append(ax.plot([], [], marker='x', color=c, markersize=6, alpha=0.5)[0])
    follower_arrows.append(FancyArrowPatch((0, 0), (1, 0), color=c, mutation_scale=15, arrowstyle='->'))
    ax.add_patch(follower_arrows[-1])

leader_arrow = FancyArrowPatch((0, 0), (1, 0), color='red', mutation_scale=15, arrowstyle='->')
ax.add_patch(leader_arrow)

def get_controller_input():
    pygame.event.pump()
    forward = (joystick.get_axis(5) + 1) / 2
    steer = -joystick.get_axis(0)
    return forward * MAX_LEADER_SPEED, steer * MAX_STEER

def animate(i):
    global leader_pose, follower_poses

    # --- Leader Control ---
    v_leader, delta_leader = get_controller_input()
    leader_pose = update_pose(leader_pose, v_leader, delta_leader)
    update_arrow(leader_arrow, leader_pose)
    leader_zone.center = (leader_pose[0], leader_pose[1])

    # Each follower follows its leader
    lead = leader_pose
    for idx, follower_pose in enumerate(follower_poses):
        rel = relative_pose(follower_pose, lead)
        dist = np.hypot(rel[0], rel[1])

        if dist > MIN_FOLLOW_DIST:
            lookahead = calc_lookahead(rel)
            delta = pure_pursuit_control(lookahead)
            speed = clamp((dist - MIN_FOLLOW_DIST), 0.0, MAX_SPEED)
            follower_pose = update_pose(follower_pose, speed, delta)
        else:
            lookahead = rel
            delta = 0

        rot = np.array([
            [np.cos(follower_pose[2]), -np.sin(follower_pose[2])],
            [np.sin(follower_pose[2]),  np.cos(follower_pose[2])]
        ])

        follower_poses[idx] = follower_pose
        update_arrow(follower_arrows[idx], follower_pose)
        lookahead_global = follower_pose[:2] + rot @ lookahead[:2]
        lookahead_markers[idx].set_data([lookahead_global[0]], [lookahead_global[1]])

        # --- Draw pursuit arc ---
        if dist > MIN_FOLLOW_DIST and abs(delta) > 1e-3:
            R = WHEELBASE / np.tan(delta)
            theta = follower_pose[2]
            x, y = follower_pose[0], follower_pose[1]

            # Determine turning center
            cx = x - R * np.sin(theta)
            cy = y + R * np.cos(theta)

            # Start and end angles relative to the turning center
            theta_start = np.arctan2(y - cy, x - cx)
            theta_end = np.arctan2(lookahead_global[1] - cy, lookahead_global[0] - cx)

            # Ensure correct arc direction
            if delta > 0:  # Turning left (CCW)
                if theta_end < theta_start:
                    theta_end += 2 * np.pi
            else:  # Turning right (CW)
                if theta_end > theta_start:
                    theta_end -= 2 * np.pi

            arc_theta = np.linspace(theta_start, theta_end, 100)
            arc_x = cx + abs(R) * np.cos(arc_theta)
            arc_y = cy + abs(R) * np.sin(arc_theta)
            arc_lines[idx].set_data(arc_x, arc_y)
        else:
            arc_lines[idx].set_data([], [])

        lead = follower_pose  # This follower becomes the next leader

    return [leader_arrow, *follower_arrows, *lookahead_markers, *arc_lines]


ani = FuncAnimation(fig, animate, interval=100)
plt.show()
