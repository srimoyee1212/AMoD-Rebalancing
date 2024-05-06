import numpy as np
from data_preprocessing import taxi_data

# Define the number of stations
num_stations = len(taxi_data["PULocationID"].unique())

# Define the maximum number of vehicles
max_vehicles = 100  # Adjust as needed based on your dataset

# Define the state space dimension
state_dim = 2 * num_stations

# Define the action space dimension
action_dim = num_stations

# Define the maximum number of actions per vehicle
max_actions_per_vehicle = 1  # Adjust as needed based on your requirements

# Define the state space bounds
state_bounds = np.array([[0, max_vehicles]] * num_stations)

# Define the action space bounds
action_bounds = np.array([[0, 1]] * num_stations)  # Binary action space (stay or move)

# Print information about the state and action spaces
print("State space dimension:", state_dim)
print("Action space dimension:", action_dim)
print("State space bounds:", state_bounds)
print("Action space bounds:", action_bounds)
