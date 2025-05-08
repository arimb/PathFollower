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

def estimate_global_displacements(rel_disp_t0, rel_disp_t1):
    """
    Estimate global displacements (x, y, theta) for N vehicles based on relative displacements
    between each vehicle and the one ahead, at two time steps.

    Parameters:
        rel_disp_t0: list of (dx, dy, dtheta) from each follower to its leader at time t0
        rel_disp_t1: same at time t1

    Returns:
        Nx3 array: estimated (dx, dy, dtheta) global displacement for each vehicle
                   (relative, up to global transform)
    """
    N = len(rel_disp_t0) + 1

    J = np.array([[0, -1], [1, 0]])  # 90 deg rotation matrix

    # Convert to relative positions at t0 and t1
    z_t0 = [np.array([dx, dy]) for dx, dy, _ in rel_disp_t0]
    z_t1 = [np.array([dx, dy]) for dx, dy, _ in rel_disp_t1]
    delta_z = [z1 - z0 for z0, z1 in zip(z_t0, z_t1)]

    delta_theta_rel = [theta1 - theta0 for (_, _, theta0), (_, _, theta1) in zip(rel_disp_t0, rel_disp_t1)]

    # Variables: 3 per vehicle (dx, dy, dtheta)
    num_vars = 3 * N
    A = []
    b = []

    for i in range(1, N):
        z0 = z_t0[i - 1]
        Jz = J @ z0

        # First row: x-component
        row = np.zeros(num_vars)
        row[3 * (i - 1) + 0] += 1  # dx_{i-1}
        row[3 * i + 0] += -1       # dx_i
        row[3 * i + 2] += Jz[0]    # dtheta_i (rotation of follower)
        A.append(row)
        b.append(delta_z[i - 1][0])

        # Second row: y-component
        row = np.zeros(num_vars)
        row[3 * (i - 1) + 1] += 1  # dy_{i-1}
        row[3 * i + 1] += -1       # dy_i
        row[3 * i + 2] += Jz[1]    # dtheta_i
        A.append(row)
        b.append(delta_z[i - 1][1])

        # Third row: theta (orientation change)
        row = np.zeros(num_vars)
        row[3 * (i - 1) + 2] += 1  # dtheta_{i-1}
        row[3 * i + 2] += -1       # dtheta_i
        A.append(row)
        b.append(delta_theta_rel[i - 1])

    A = np.array(A)
    b = np.array(b)

    # Solve least squares
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x.reshape((N, 3))


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

# Relative measurement history
estimated_global_poses = [leader_pose.copy()] + [pose.copy() for pose in follower_poses]
prev_rel_poses = None

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
    global leader_pose, follower_poses, prev_rel_poses, estimated_global_poses

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

    # --- Follower Control ---
    # Each follower follows its leader
    lead = leader_pose
    rel_poses = []
    for idx, follower_pose in enumerate(follower_poses):

        # Follower logic
        rel = relative_pose(follower_pose, lead)
        rel_poses.append(rel)
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

    # Estimate the global pose
    if prev_rel_poses is not None:
        displacements = estimate_global_displacements(prev_rel_poses, rel_poses)
        for j, (dx, dy, dtheta) in enumerate(displacements):
            theta = estimated_global_poses[j][2]
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            delta_global = rot @ np.array([dx, dy])
            estimated_global_poses[j][:2] += delta_global
            estimated_global_poses[j][2] += dtheta

            # Optional: Print error
            error = np.linalg.norm(estimated_global_poses[j][:2] - follower_poses[j][:2])
            print(f"Follower {j} error: {error:.2f}")

    prev_rel_poses = rel_poses

    return [leader_arrow, *follower_arrows, *lookahead_markers, *arc_lines, leader_trail_line]



ani = FuncAnimation(fig, animate, interval=100)
plt.show()
