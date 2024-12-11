# this is the optimization based agent

import logging
from gymnasium import spaces
from abc import ABC, abstractmethod
from ttrl_agent.configuration import Configurable

from pomdp_solver import *

logger = logging.getLogger(__name__)


class POMCPAgent(Configurable, ABC):
    def __init__(self, env, map, num_particles=10, max_depth=5, num_sims=20):
        self.env = env
        self.map = map
        self.num_particles = num_particles
        self.max_depth=max_depth
        self.num_sims=num_sims
        # Access the config safely
        if hasattr(env, "get_wrapper_attr"):
            # Use `get_wrapper_attr` to fetch the config
            config = env.get_wrapper_attr("config")
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "config"):
            # Fallback to `unwrapped.config` if `get_wrapper_attr` is unavailable
            config = env.unwrapped.config
        self.target_speeds = config["action"]["target_speeds"]
        self.vel_range = [self.target_speeds[1], self.target_speeds[3]]
        self.acc_range = [-2.0, 2.0]
        self.dt = 1 / config["simulation_frequency"]
        self.xrange = config["observation"]["features_range"]["x"]
        self.yrange = config["observation"]["features_range"]["y"]
        self.vxrange = config["observation"]["features_range"]["vx"]
        self.vyrange = config["observation"]["features_range"]["vy"]

    def reset(self, state):
        vehicles_data = self.convert_obs_to_dict(state)
        self.init_map_id = []
        for i in range(len(vehicles_data)):
            x = vehicles_data[i]["x"]
            y = vehicles_data[i]["y"]
            cos_h = vehicles_data[i]["cos_h"]
            sin_h = vehicles_data[i]["sin_h"]
            _, r = self.map.convert_to_topo_position((x,y,np.arctan2(sin_h, cos_h)))
            self.init_map_id.append(r)

        # update map table based on observation
        self.trim_map = self.map.trim_map(self.init_map_id)
        self.trim_map.draw_tree("topology_graph_trimed_"+str(self.init_map_id)+".png")
        # Initialize components from pomdp_core
        self.transition_model = TransitionModel(self.trim_map, self.dt)
        self.observation_model = ObservationModel(self.trim_map, 0.1)
        self.reward_model = RewardModel(self.trim_map, self.dt, self.vel_range, self.acc_range)
        self.policy_model = PolicyModel(self.dt, self.target_speeds[2])

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
                      np.ndarray
        :return: a, an action to perform, where a is a int
                 ACTIONS_LONGI = {0: "STOP", 1: "SLOWER", 2: "IDLE", 3: "FASTER"}
        """

        vehicles_data = self.convert_obs_to_dict(state)
        ego = vehicles_data[0]
        ego_position = (ego["x"], ego["y"], np.arctan2(ego["sin_h"], ego["cos_h"]))
        ego_s, ego_r = self.trim_map.convert_to_topo_position(ego_position)
        ego_v = (ego["vx"]**2 + ego["vy"]**2)**0.5
        initial_state_data = [(ego_s, ego_v, 0.0, ego_r)]

        for i in range(1, len(vehicles_data)):
            x = vehicles_data[i]["x"]
            y = vehicles_data[i]["y"]
            vx = vehicles_data[i]["vx"]
            vy = vehicles_data[i]["vy"]
            cos_h = vehicles_data[i]["cos_h"]
            sin_h = vehicles_data[i]["sin_h"]
            s, r = self.trim_map.convert_to_topo_position((x,y,np.arctan2(sin_h, cos_h)))
            initial_state_data.append((s, (vx**2 + vy**2)**0.5, 0.0, r))
        # find the initial state based on map and obs
        initial_state = State(initial_state_data)
        # Initialize belief with particles around the initial state
        belief = Belief([initial_state] * self.num_particles, transition_model=self.transition_model)
        # Initialize POMCPOW Solver
        solver = POMCPOWSolver(
            belief=belief,
            transition_model=self.transition_model,
            observation_model=self.observation_model,
            reward_model=self.reward_model,
            policy_model=self.policy_model,
            max_depth=self.max_depth,
            num_sims=self.num_sims,
        )
        # convert to a value in self.env.action_space based on self.target_speeds
        acc_action = solver.plan()
        # Calculate the new speed
        target_speed = max(0.0, ego_v + acc_action.data * self.dt * 10) # POMCPOW is targeted on max_depth=5
        print(f"target acc: {acc_action.data:.2f} target speed: {target_speed:.4f}")
        # Find the key corresponding to the target speed closest to the new speed
        best_key = None
        min_diff = float('inf')  # Set to a very large value initially

        # Iterate through target_speeds dictionary
        for key, value in enumerate(self.target_speeds):
            diff = abs(target_speed - value) # Calculate the difference
            if diff < min_diff:
                min_diff = diff  # Update the smallest difference
                best_key = key  # Update the best target key

        if min_diff > 1.0: # buffer zone: 1.0 [m/s]
            best_key = int(min(3, max(0, best_key + np.sign(target_speed - self.target_speeds[best_key]))))

        # Visualize the step and save the result
        self.visualize_step(vehicles_data, best_key, target_speed, save_path="step_visualization_"+str(ego)+".png")
        print(f"save image for {ego}")

        # return action key
        return best_key

    def set_directory(self, directory):
        self.directory = directory

    def convert_obs_to_dict(self, obs_array):
        """
            Convert env's observation (numpy array) into a dictionary of dictionaries.

        :param obs_array: numpy array from the observe() method.
        :return: dict of dicts where each key is 'vehicle_i' and the value is a dict of features.
        """
        features = self.env.config["observation"]["features"]
        vehicles_data = []
        def reverse_normalize(value, feature_range):
            """
            Reverse normalization for a given value and feature range.

            :param value: Normalized value (e.g., in [-1, 1]).
            :param feature_range: Tuple (min_value, max_value) of the original range.
            :return: Original value in the feature range.
            """
            min_val, max_val = feature_range
            return (value + 1) * (max_val - min_val) / 2 + min_val

        for i, row in enumerate(obs_array):
            # Map each row to a dictionary with feature names as keys.
            vehicle_dict = {feature: row[j] for j, feature in enumerate(features)}
            if vehicle_dict["presence"] > 0.5:
                # Reverse normalization for specific features
                if "x" in vehicle_dict:
                    vehicle_dict["x"] = reverse_normalize(vehicle_dict["x"], self.xrange)
                if "y" in vehicle_dict:
                    vehicle_dict["y"] = reverse_normalize(vehicle_dict["y"], self.yrange)
                if "vx" in vehicle_dict:
                    vehicle_dict["vx"] = reverse_normalize(vehicle_dict["vx"], self.vxrange)
                if "vy" in vehicle_dict:
                    vehicle_dict["vy"] = reverse_normalize(vehicle_dict["vy"], self.vyrange)

                # Add this dictionary to the vehicles_data list
                vehicles_data.append(vehicle_dict)
        return vehicles_data

    def visualize_step(self, vehicles_data, ego_action, ego_target_speed, save_path=None):
        """
        Visualize the current step with vehicle positions and actions.

        :param vehicles_data: List of dictionaries containing vehicle state data.
        :param ego_action: The action chosen for the ego vehicle.
        :param ego_target_speed: The target speed for the ego vehicle.
        :param save_path: Optional path to save the visualization image.
        """
        plt.figure(figsize=(8, 8))
        plt.title("POMDP Step Visualization")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        # Set plot limits
        plt.xlim(-60, 60)
        plt.ylim(-60, 60)

        # Plot other vehicles
        for vehicle in vehicles_data[1:]:
            plt.scatter(vehicle["x"], vehicle["y"], color="blue", label="Other Vehicles" if vehicle == vehicles_data[1] else None)
            plt.arrow(vehicle["x"], vehicle["y"], vehicle["vx"], vehicle["vy"], color="blue", head_width=0.5)

        # Plot ego vehicle
        ego = vehicles_data[0]
        plt.scatter(ego["x"], ego["y"], color="red", label="Ego Vehicle")
        plt.arrow(ego["x"], ego["y"], ego["vx"], ego["vy"], color="red", head_width=0.5, label="Ego Velocity")
        ego_v = (ego["vx"]**2 + ego["vy"]**2)**0.5

        # Annotate the ego action and target speed
        plt.text(ego["x"], ego["y"] + 1, f"Action: {ego_action}\nTarget Speed: {ego_target_speed:.2f}\nSpeed: {ego_v:.2f}", color="red")

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()
