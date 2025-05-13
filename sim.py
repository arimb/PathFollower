import pygame
import numpy as np
import matplotlib.pyplot as plt
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
NUM_VEHICLES = 2

# Initialize Pygame for controller input
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# --- Helper functions ---

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

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

def stanley_control(rel_target, current_theta, k=1.0):
    """
    Stanley controller for lateral control.
    
    Parameters:
        rel_target: np.array([x, y]) — target position in vehicle-relative coordinates.
        current_theta: float — current global orientation of the vehicle.
        k: float — control gain for cross-track error.

    Returns:
        delta: steering angle (radians)
    """
    # Cross-track error is the lateral displacement in vehicle frame
    cross_track_error = rel_target[1]

    # Heading error (in vehicle frame, assume target heading is along x-axis)
    heading_error = np.arctan2(rel_target[1], rel_target[0])

    # Avoid divide by zero
    L = np.hypot(rel_target[0], rel_target[1])
    if L < 1e-6:
        return 0.0

    # Stanley steering law
    delta = heading_error + np.arctan2(k * cross_track_error, L)

    # Clamp to max steering
    return clamp(delta, -MAX_STEER, MAX_STEER)

def pure_pursuit_path_control(relative_trajectory, current_pose, lookahead_distance, k_v=1.0, max_omega=np.pi/4):
    """
    Implements the Pure Pursuit algorithm to follow the leader's path.

    Inputs:
        relative_trajectory: list of (relative_x, relative_y, relative_theta) positions
        current_pose: current position of the follower (x, y, theta)
        lookahead_distance: distance ahead of the follower to look for a target point on the path
        k_v: Linear velocity gain (default is 1.0)
        max_omega: Maximum angular velocity (steering limit) (default is pi/4)

    Outputs:
        v: Linear velocity command
        omega: Angular velocity command (steering input)
    """
    
    # Find the lookahead point in the trajectory
    target_idx = None
    min_dist = float('inf')
    
    # Loop over the trajectory and find the point closest to the lookahead distance
    for i, (target_x, target_y, _) in enumerate(relative_trajectory):
        dist = np.linalg.norm([target_x - current_pose[0], target_y - current_pose[1]])
        if dist > lookahead_distance and dist < min_dist:
            min_dist = dist
            target_idx = i
    
    # If no valid target is found (i.e., the follower has reached the end of the trajectory), return zero velocities
    if target_idx is None:
        return 0.0, 0.0
    
    # Get the target lookahead point
    target_x, target_y, _ = relative_trajectory[target_idx]
    
    # Compute the lookahead angle
    dx = target_x - current_pose[0]
    dy = target_y - current_pose[1]
    alpha = np.arctan2(dy, dx) - current_pose[2]
    alpha = angle_wrap(alpha)  # Normalize the angle to [-pi, pi]
    
    # Calculate steering angle based on the lookahead geometry
    L = np.linalg.norm([dx, dy])  # Distance to the lookahead point
    if L == 0:
        omega = 0.0
    else:
        omega = 2 * np.sin(alpha) / L  # Pure Pursuit steering angle formula

    # Linear velocity (k_v scaling factor for speed)
    v = k_v * min(L, lookahead_distance)  # Limit the velocity to the lookahead distance

    # Limit omega to the max steering rate (for safety)
    omega = np.clip(omega, -max_omega, max_omega)
    
    return v, omega

def estimate_global_displacements(rel_disp_t1, rel_disp_t2, headings):
    """
    Estimate global linear displacements (positions) of N non-holonomic vehicles using relative measurements and headings.

    Parameters:
        rel_disp_t1: list of tuples (dx, dy, dtheta) for time t1
        rel_disp_t2: list of tuples (dx, dy, dtheta) for time t2
            - Each entry i is the relative pose of vehicle i with respect to i+1 in i+1's local frame
        headings: list of global headings (in radians) of each vehicle at time t1

    Returns:
        positions: Nx2 NumPy array of global displacement vectors (dx, dy) for each vehicle
    """
    N = len(headings)
    assert len(rel_disp_t1) == len(rel_disp_t2) == N - 1

    # Compute relative displacement change in each follower's local frame
    delta_rel = []
    for (dx1, dy1, _), (dx2, dy2, _) in zip(rel_disp_t1, rel_disp_t2):
        delta_rel.append((dx2 - dx1, dy2 - dy1))

    # Convert to global frame using follower headings
    b = []
    for i, (dx_local, dy_local) in enumerate(delta_rel):
        theta_follower = headings[i + 1]
        R = np.array([
            [np.cos(theta_follower), -np.sin(theta_follower)],
            [np.sin(theta_follower), np.cos(theta_follower)]
        ])
        delta_global = R @ np.array([dx_local, dy_local])
        b.append(delta_global)

    b = np.concatenate(b)  # shape (2*(N-1),)

    # Build H matrix
    H = np.zeros((2 * (N - 1), N))
    for i in range(N - 1):
        h_i = np.array([np.cos(headings[i]), np.sin(headings[i])])
        h_ip1 = np.array([np.cos(headings[i + 1]), np.sin(headings[i + 1])])
        H[2 * i:2 * i + 2, i] = h_i
        H[2 * i:2 * i + 2, i + 1] = -h_ip1

    # Solve least squares: H v = b
    v, _, _, _ = np.linalg.lstsq(H, b, rcond=None)

    # Compute global displacement vectors
    positions = np.array([
        v_i * np.array([np.cos(theta), np.sin(theta)])
        for v_i, theta in zip(v, headings)
    ])

    return positions


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
vehicle_poses = [np.array([-i * MIN_FOLLOW_DIST, -1.0, 0.0]) for i in range(NUM_VEHICLES)]
vehicle_poses[0] = np.array([0.0, 0.0, 0.0])  # Leader at origin

# Initialize target buffers for followers (excluding leader)
BUFFER_SIZE = 10
position_buffer = [deque(maxlen=BUFFER_SIZE) for _ in range(NUM_VEHICLES - 1)]

# Relative measurement history
estimated_global_poses = [pose.copy() for pose in vehicle_poses]
prev_rel_poses = None

# Visualization setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)

# Draw leader and follower arrows, lookahead markers, and pursuit arcs
vehicle_arrows = []
debug_markers = []
arc_lines = []
colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx in range(NUM_VEHICLES):
    if idx > 0:
        arc_lines.append(ax.plot([], [], linestyle=':', color=colors[idx], lw=1.5, alpha=0.5)[0])
    debug_markers.append(ax.plot([], [], marker='x', color=colors[idx], markersize=6, alpha=0.5)[0])
    vehicle_arrows.append(FancyArrowPatch((0, 0), (1, 0), color=colors[idx], mutation_scale=15, arrowstyle='->'))
    ax.add_patch(vehicle_arrows[-1])

leader_trail = []
MAX_TRAIL_POINTS = 150
leader_trail_line, = ax.plot([], [], 'k', lw=1.5, alpha=0.3)  # Black dotted line

# --- Animation loop ---
def animate(i):
    global vehicle_poses, prev_rel_poses, estimated_global_poses

    # --- Leader Control ---
    v_leader, delta_leader = get_controller_input()
    vehicle_poses[0] = update_pose(vehicle_poses[0], v_leader, delta_leader)

    leader_trail.append(vehicle_poses[0][:2])
    if len(leader_trail) > MAX_TRAIL_POINTS:
        leader_trail.pop(0)

    update_arrow(vehicle_arrows[0], vehicle_poses[0])
    if leader_trail:
        trail_array = np.array(leader_trail)
        leader_trail_line.set_data(trail_array[:, 0], trail_array[:, 1])

    # --- Follower Control ---
    # Each follower follows its leader
    rel_poses = []
    for idx in range(1, NUM_VEHICLES):
        lead = vehicle_poses[idx - 1]
        follower_pose = vehicle_poses[idx]

        # Follower logic
        rel = relative_pose(follower_pose, lead)
        rel_poses.append(rel)
        dist = np.hypot(rel[0], rel[1])

        if dist > MIN_FOLLOW_DIST:
            position_buffer[idx-1].append(rel[:2].copy())

            # Use delayed target if buffer is full
            # if len(position_buffer[idx-1]) == BUFFER_SIZE:
            buffered_pos = position_buffer[idx-1][0]
            target = calc_lookahead(np.append(buffered_pos, rel[2]))
            # else:
            #     target = calc_lookahead(rel)

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

        vehicle_poses[idx] = follower_pose
        update_arrow(vehicle_arrows[idx], follower_pose)
        target_global = follower_pose[:2] + rot @ target[:2]
        debug_markers[idx - 1].set_data([estimated_global_poses[idx][0]], [estimated_global_poses[idx][1]])

        if dist > MIN_FOLLOW_DIST:
            draw_pursuit_arc(follower_pose, delta, target_global, arc_lines[idx - 1])
        else:
            arc_lines[idx - 1].set_data([], [])

    # Estimate the global pose
    if prev_rel_poses is not None:
        displacements = estimate_global_displacements(prev_rel_poses, rel_poses, [pose[2] for pose in vehicle_poses])
        print(displacements)
        for j, (dx, dy) in enumerate(displacements):
            theta = vehicle_poses[j][2]
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            delta_global = rot @ np.array([dx, dy])
            estimated_global_poses[j][:2] += delta_global

            # error = np.linalg.norm(estimated_global_poses[j][:2] - vehicle_poses[j][:2])
            # print(f"Vehicle {j} error: {error:.2f}")

    prev_rel_poses = rel_poses

    return [*vehicle_arrows, *debug_markers, *arc_lines, leader_trail_line]



ani = FuncAnimation(fig, animate, interval=100)
plt.show()
