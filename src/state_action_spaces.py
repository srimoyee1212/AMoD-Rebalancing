import numpy as np

class StateActionSpace:
    def __init__(self, taxi_data):
        self.taxi_data = taxi_data

        # Define the number of stations
        self.num_stations = len(taxi_data["PULocationID"].unique())

        # Define the maximum number of vehicles
        self.max_vehicles = 100  # Adjust as needed based on your dataset

        # Define the state space dimension
        self.state_dim = 2 * self.num_stations

        # Define the action space dimension
        self.action_dim = self.num_stations

        # Define the maximum number of actions per vehicle
        self.max_actions_per_vehicle = 1  # Adjust as needed based on your requirements

        # Define the state space bounds
        self.state_bounds = np.array([[0, self.max_vehicles]] * self.num_stations)

        # Define the action space bounds
        self.action_bounds = np.array([[0, 1]] * self.num_stations)  # Binary action space (stay or move)

        # Print information about the state and action spaces
        print("State space dimension:", self.state_dim)
        print("Action space dimension:", self.action_dim)
        print("State space bounds:", self.state_bounds)
        print("Action space bounds:", self.action_bounds)
