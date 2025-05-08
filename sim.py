import pygame
import numpy as np
import matplotlib.pyplot as plt
from Tools.scripts.generate_re_casefix import alpha
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from collections import deque

# Constants
DT = 0.1
MAX_LEADER_SPEED = 2.0
MAX_SPEED = 2 * MAX_LEADER_SPEED
MAX_STEER = np.radians(20)
WHEELBASE = 1.0
JOYSTICK_DEADBAND = 0.1
MIN_FOLLOW_DIST = 3.0
NUM_FOLLOWERS = 4

# Initialize Pygame for controller input
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# --- Helper functions ---

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

def draw_pursuit_arc(follower_pose, delta, target_global, arc_line):
    if abs(delta) < 1e-3:
        arc_line.set_data([], [])
        return

    R = WHEELBASE / np.tan(delta)
    theta = follower_pose[2]
    x, y = follower_pose[0], follower_pose[1]

    # Turning center
    cx = x - R * np.sin(theta)
    cy = y + R * np.cos(theta)

    # Start and end angles
    theta_start = np.arctan2(y - cy, x - cx)
    theta_end = np.arctan2(target_global[1] - cy, target_global[0] - cx)

    # Arc direction
    if delta > 0 and theta_end < theta_start:
        theta_end += 2 * np.pi
    elif delta < 0 and theta_end > theta_start:
        theta_end -= 2 * np.pi

    arc_theta = np.linspace(theta_start, theta_end, 100)
    arc_x = cx + abs(R) * np.cos(arc_theta)
    arc_y = cy + abs(R) * np.sin(arc_theta)
    arc_line.set_data(arc_x, arc_y)

# --- Initialization ---

# Initial states
leader_pose = np.array([0.0, 0.0, 0.0])
follower_poses = [np.array([-(i+1) * MIN_FOLLOW_DIST, -1.0, 0.0]) for i in range(NUM_FOLLOWERS)]

# Initialize target buffer
BUFFER_SIZE = 10
target_buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(NUM_FOLLOWERS)]

# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)

# Draw leader and follower arrows, lookahead markers, and pursuit arcs
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

leader_trail = []
MAX_TRAIL_POINTS = 150
leader_trail_line, = ax.plot([], [], 'k', lw=1.5, alpha=0.3)  # Black dotted line

# --- Animation loop ---
def animate(i):
    global leader_pose, follower_poses

    # --- Leader Control ---
    v_leader, delta_leader = get_controller_input()
    leader_pose = update_pose(leader_pose, v_leader, delta_leader)
    leader_trail.append(leader_pose[:2])
    if len(leader_trail) > MAX_TRAIL_POINTS:
        leader_trail.pop(0)

    update_arrow(leader_arrow, leader_pose)
    if leader_trail:
        trail_array = np.array(leader_trail)
        leader_trail_line.set_data(trail_array[:, 0], trail_array[:, 1])


    # Each follower follows its leader
    lead = leader_pose
    for idx, follower_pose in enumerate(follower_poses):

        # Follower logic
        rel = relative_pose(follower_pose, lead)
        dist = np.hypot(rel[0], rel[1])

        if dist > MIN_FOLLOW_DIST:
            lookahead = calc_lookahead(rel)
            target_buffers[idx].append(lookahead)

            # Use delayed target if buffer is full
            if len(target_buffers[idx]) == BUFFER_SIZE:
                target = target_buffers[idx][0]
            else:
                target = lookahead

            delta = pure_pursuit_control(target)
            speed = clamp((dist - MIN_FOLLOW_DIST), 0.0, MAX_SPEED)
            follower_pose = update_pose(follower_pose, speed, delta)
        else:
            target = rel
            delta = 0

        # Update animation
        rot = np.array([
            [np.cos(follower_pose[2]), -np.sin(follower_pose[2])],
            [np.sin(follower_pose[2]),  np.cos(follower_pose[2])]
        ])

        follower_poses[idx] = follower_pose
        update_arrow(follower_arrows[idx], follower_pose)
        target_global = follower_pose[:2] + rot @ target[:2]
        lookahead_markers[idx].set_data([target_global[0]], [target_global[1]])

        if dist > MIN_FOLLOW_DIST:
            draw_pursuit_arc(follower_pose, delta, target_global, arc_lines[idx])
        else:
            arc_lines[idx].set_data([], [])

        lead = follower_pose  # This follower becomes the next leader

    return [leader_arrow, *follower_arrows, *lookahead_markers, *arc_lines, leader_trail_line]



ani = FuncAnimation(fig, animate, interval=100)
plt.show()
