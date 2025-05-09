import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import pygame

# --- Constants ---
NUM_VEHICLES = 5
DT = 0.1  # seconds
DELAY_TIME = 2.0  # seconds
DELAY_STEPS = int(DELAY_TIME / DT)
MAX_SPEED = 2.0  # m/s
MAX_STEER = np.radians(30)  # radians/s
TRAIL_TIME = 10.0  # seconds
MAX_TRAIL_POINTS = int(TRAIL_TIME / DT)

# --- Helper Functions ---
def sinc(x):
    if abs(x) > 1e-6:
        return np.sin(x) / x
    else:
        return 1.0

# --- Pygame Joystick Setup ---
pygame.init()
pygame.joystick.init()
try:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
except pygame.error:
    print("No joystick found. Please connect a joystick and restart the program.")
    exit()

def get_controller_input():
    pygame.event.pump()
    forward = (joystick.get_axis(5) + 1) / 2  # Trigger usually from -1 to 1
    steer = -joystick.get_axis(0)
    return forward * MAX_SPEED, steer * MAX_STEER

# --- Motion Model ---
def update_pose(pose, v, omega, dt=DT):
    x, y, theta = pose
    return (
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + omega * dt
    )

def control_inputs_equation_30(x_rel, y_rel, alpha, v_tau, omega_tau, k1=1.0, k2=2.0, k3=3.0):
    omega = k2 * alpha + k3 * y_rel * v_tau * sinc(alpha) + omega_tau
    v = v_tau * np.cos(alpha) + k1 * x_rel
    return v, omega

# --- Drawing Functions ---

def update_arrow(arrow, pose, length=1.5):
    x, y, theta = pose
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    arrow.set_positions((x, y), (x + dx, y + dy))

def draw_leader_line(vehicle_poses):
    global leader_trail, leader_trail_line
    leader_trail.append(vehicle_poses[0][:2])
    if len(leader_trail) > MAX_TRAIL_POINTS:
        leader_trail.pop(0)

    update_arrow(vehicle_arrows[0], vehicle_poses[0])
    if leader_trail:
        trail_array = np.array(leader_trail)
        leader_trail_line.set_data(trail_array[:, 0], trail_array[:, 1])

# --- Initialize State ---
poses = [np.array([[-i * 2.0, 0.0, 0.0] for i in range(NUM_VEHICLES)])]
vels = [np.zeros((NUM_VEHICLES, 2))]

# --- Plot Setup ---
fig, ax = plt.subplots()
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')
ax.grid(True)

vehicle_arrows = []
colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx in range(NUM_VEHICLES):
    vehicle_arrows.append(FancyArrowPatch((0, 0), (1, 0), color=colors[idx], mutation_scale=15, arrowstyle='->'))
    ax.add_patch(vehicle_arrows[-1])

leader_trail = []
leader_trail_line, = ax.plot([], [], 'k', lw=1.5, alpha=0.3)  # Black dotted line


# --- Animation Function ---
def animate(i):
    global poses, vels

    t = len(poses)
    new_poses = poses[-1].copy()
    new_vels = vels[-1].copy()

    # Leader
    v, omega = get_controller_input()
    new_poses[0] = update_pose(new_poses[0], v, omega)
    new_vels[0] = [v, omega]

    # Followers
    for i in range(1, NUM_VEHICLES):
        if t - DELAY_STEPS < 0:
            continue
        lead_pose = poses[t - DELAY_STEPS][i - 1]
        lead_v, lead_omega = vels[t - DELAY_STEPS][i - 1]
        follower = new_poses[i]

        dx = lead_pose[0] - follower[0]
        dy = lead_pose[1] - follower[1]
        dtheta = lead_pose[2] - follower[2]

        cos_phi = np.cos(-follower[2])
        sin_phi = np.sin(-follower[2])
        x_rel = cos_phi * dx - sin_phi * dy
        y_rel = sin_phi * dx + cos_phi * dy
        alpha = dtheta

        v_i, omega_i = control_inputs_equation_30(x_rel, y_rel, alpha, lead_v, lead_omega)
        new_poses[i] = update_pose(follower, v_i, omega_i)
        new_vels[i] = [v_i, omega_i]

    # Append state
    poses.append(new_poses)
    vels.append(new_vels)

    # Redraw
    for idx, arrow in enumerate(vehicle_arrows):
        update_arrow(arrow, new_poses[idx])
    draw_leader_line(new_poses)

    return [*vehicle_arrows, leader_trail_line]

# --- Launch Animation ---
try:
    ani = FuncAnimation(fig, animate, interval=DT*1000, blit=True, cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    print("Animation stopped by user.")
finally:
    pygame.quit()
