import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import sys

def create_custom_grid(x_coords, y_coords):
    """
    Generate grid points based on custom x and y coordinates.

    Args:
        x_coords (list or numpy.ndarray): List of x-axis coordinates.
        y_coords (list or numpy.ndarray): List of y-axis coordinates.

    Returns:
        grid (numpy.ndarray): Array of grid points with shape 
                              (len(x_coords) * len(y_coords), 2).
    """
    # Create meshgrid from x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid = np.column_stack((X.flatten(), Y.flatten()))
    return grid


def update_vehicle_state(x, y, theta, v, delta, L, dt):
    """
    Update the vehicle position and heading angle based on the front wheel 
    steering angle and velocity (kinematic bicycle model).

    Args:
        x (float): Current x position of the vehicle.
        y (float): Current y position of the vehicle.
        theta (float): Current heading angle of the vehicle (radians).
        v (float): Vehicle velocity.
        delta (float): Front wheel steering angle (radians).
        L (float): Wheelbase length of the vehicle.
        dt (float): Time step.

    Returns:
        tuple:
            x_new (float): Updated x position.
            y_new (float): Updated y position.
            theta_new (float): Updated heading angle (radians).
    """
    # Compute state changes for the current time step
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dtheta = (v / L) * np.tan(delta) * dt  # assuming constant velocity v

    # Update vehicle position and heading
    x_new = x + dx
    y_new = y + dy
    theta_new = theta + dtheta

    return x_new, y_new, theta_new


def generate_x_coords(n, start=0.0, initial_interval=0.25, interval_diff=0.25):
    """
    Generate a sequence of x-coordinates with increasing spacing based on an 
    arithmetic progression.

    Args:
        n (int): Number of points to generate.
        start (float): Starting x-coordinate value.
        initial_interval (float): The first interval between points.
        interval_diff (float): Increment added to each subsequent interval.

    Returns:
        list: A list of generated x-coordinates.
    """
    x_coords = [start]
    current_interval = initial_interval
    for i in range(1, n):
        next_coord = x_coords[-1] + current_interval
        x_coords.append(next_coord)
        current_interval += interval_diff
    return x_coords


def generate_y_coords(n, start=0.025, initial_interval=0.02, interval_diff=0.02):
    """
    Generate a sequence of y-coordinates spreading outward from the center (0).
    The spacing between points increases according to an arithmetic progression.

    Args:
        n (int): Number of points to generate.
        start (float): Starting offset for the first coordinate.
        initial_interval (float): The first interval value.
        interval_diff (float): Increment added to each subsequent interval.

    Returns:
        list: A list of generated y-coordinates.
    """
    half_n = n // 2
    intervals = [initial_interval + i * interval_diff for i in range(half_n)]
    intervals[0] = start

    right_side = [sum(intervals[:i+1]) for i in range(half_n)]
    left_side = [-x for x in reversed(right_side)]

    if n % 2 == 1:
        return left_side + [0] + right_side
    else:
        return left_side + right_side

# Parameter settings
nx = 21   # Number of grid points along the x-axis
ny = 21   # Number of grid points along the y-axis
xmin = 0.0   # Minimum value of x-axis
xmax = 10.0  # Maximum value of x-axis
ymin = -0.5  # Minimum value of y-axis
ymax = 0.5   # Maximum value of y-axis

n_points = 61
x_coords = generate_x_coords(n_points, start=float(0.0), initial_interval=float(0.25), interval_diff=float(0.0))
x_coords = [round(coord, 3) for coord in x_coords]

y_coords = generate_y_coords(n_points, start=0.025, initial_interval=0.025, interval_diff=0)
y_coords = [round(coord, 4) for coord in y_coords]

print(f"[INFO] Generated {len(x_coords)} x-coordinates, range: {x_coords[0]} ~ {x_coords[-1]}")
print(f"[INFO] Generated {len(y_coords)} y-coordinates, range: {y_coords[0]} ~ {y_coords[-1]}")

grid_points = create_custom_grid(x_coords, y_coords)

# Create a set of linear and angular velocities
v_values = np.linspace(0, 32, 320*2+1)       # Linear velocity range [0, 32] m/s
deg_values = np.linspace(-49, 49, 980*4+1)   # Steering angle range [-49, 49] degrees
L = 2.765    # Vehicle wheelbase (m)
dt = 0.01    # Time step (s)
steps = 50   # Number of simulation steps

all_trajectories = []
all_trajectories_yaw = []

# Generate dense anchor trajectories
for v in tqdm(v_values, desc="Generating anchors"):
    for deg in deg_values:
        # Initialize parameters
        x_init = 0.0      # Initial x position
        y_init = 0.0      # Initial y position
        theta_init = 0.0  # Initial heading angle (radians)
        delta = np.radians(deg)  # Front wheel steering angle (converted to radians)

        # Initialize state
        x, y, theta = x_init, y_init, theta_init

        # Record vehicle states for visualization (optional)
        # x_history, y_history, theta_history = [x], [y], [theta]

        trajectory = []
        trajectory.append([x, y, theta])

        # Iteratively update vehicle state
        for _ in range(steps):
            x, y, theta = update_vehicle_state(x, y, theta, v, delta, L, dt)
            trajectory.append([x, y, theta])
        
        trajectory = np.array(trajectory)[::10]
        all_trajectories.append(trajectory[:, :2])
        all_trajectories_yaw.append(trajectory[1:2, 2])

anchor_file_path = "./data/traj_anchor_05s_dense.npy"
yaw_file_path = "./data/traj_anchor_yaw_05s_dense.npy"

np.save(anchor_file_path,all_trajectories)
np.save(yaw_file_path,all_trajectories_yaw)

all_trajectories = np.load(anchor_file_path)
all_trajectories_yaw = np.load(yaw_file_path)

# Match grid points to the nearest trajectory points
match_traj = []
match_traj_yaw = []

# Initialize mask matrix to mark uniquely matched grid points
match_mask = np.zeros_like(grid_points[:, 0], dtype=bool)

# Initialize arrays for matched trajectory indices and distances
matched_indices = -np.ones(len(all_trajectories), dtype=int)  # Index of matched trajectory for each grid point
matched_distances = np.full(len(all_trajectories), np.inf)    # Minimum distance to the matched trajectory

# for grid_index, grid_point in enumerate(grid_points):
for grid_index, grid_point in tqdm(enumerate(grid_points), 
                                  total=len(grid_points), 
                                  desc="Anchor traj matching"):
    distances = np.linalg.norm(all_trajectories[0:,5,:] - grid_point, axis=1)
    nearest_index = np.argmin(distances)
    min_distance = np.min(distances)  
    min_index = np.argmin(distances) 

    if min_distance < matched_distances[min_index]:
        if matched_indices[min_index] != -1: 
            old_grid_index = matched_indices[min_index]
            match_mask[old_grid_index] = False 
        matched_distances[min_index] = min_distance 
        matched_indices[min_index] = grid_index
        match_mask[grid_index] = True
    else:
        match_mask[grid_index] = False

    match_traj.append(all_trajectories[nearest_index])
    match_traj_yaw.append(all_trajectories_yaw[nearest_index])

match_traj = np.array(match_traj)
match_traj_yaw = np.array(match_traj_yaw)
match_mask = match_mask[..., np.newaxis]

match_traj_temp = match_traj.reshape(n_points,n_points,6,2)
match_traj_yaw_temp = match_traj_yaw.reshape(n_points,n_points)
match_mask_temp = match_mask.reshape(n_points,n_points)

boundary = {
            0: 30,
            1: 30,
            2: 28,
            3: 26,
            4: 22,
            5: 17,
            6: 10,
            7: 1,
            }

for i in range(0,8):
    b = boundary[i]
    match_mask_temp[i,:b] = False
    match_mask_temp[i,n_points-b:] = False
    match_traj_yaw_temp[i,:b] = match_traj_yaw_temp[i,b]
    match_traj_yaw_temp[i,n_points-b:] = match_traj_yaw_temp[i,n_points-b-1]
    match_traj_temp[i,:b] = match_traj_temp[i,b]
    match_traj_temp[i,n_points-b:] = match_traj_temp[i,n_points-b-1]

anchor_match_file_path = "./data/traj_anchor_05s_3721.npy"
yaw_match_file_path = "./data/traj_anchor_yaw_05s_3721.npy"
mask_match_file_path = "./data/traj_anchor_mask_05s_3721.npy"

np.save(anchor_match_file_path,match_traj_temp.reshape(-1,6,2))
np.save(yaw_match_file_path,match_traj_yaw_temp.reshape(-1,1))
np.save(mask_match_file_path,match_mask_temp.reshape(-1,1))