# this is the optimization based agent
# TODO: modify based on the implementation of topo_solver class

import logging
from gymnasium import spaces
from abc import ABC, abstractmethod
from ttrl_agent.configuration import Configurable

from topo_solver import *

logger = logging.getLogger(__name__)


class TopoAgent(Configurable, ABC):
    def __init__(self,
                 env,
                 map,
                 num_particles=10,
                 max_depth=5,
                 num_sims=20):
        self.env = env
        self.map = map
        self.num_particles = num_particles
        self.max_depth=max_depth,
        self.num_sims=num_sims,

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

        # Initialize components from pomdp_core
        self.transition_model = TransitionModel(self.trim_map)
        self.observation_model = ObservationModel(self.trim_map, 0.1)
        self.reward_model = RewardModel(self.trim_map)
        self.policy_model = PolicyModel()

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
                      np.ndarray
        :return: [a0, a1, a2...], a sequence of actions to perform, where a is a int
                 ACTIONS_LONGI = {0: "STOP", 1: "SLOWER", 2: "IDLE", 3: "FASTER"}
        """

        vehicles_data = self.convert_obs_to_dict(state)
        ego = vehicles_data[0]
        ego_position = (ego["x"], ego["y"], np.arctan2(ego["sin_h"], ego["cos_h"]))
        ego_s, ego_r = self.map.convert_to_topo_position_with_reference(ego_position, self.init_map_id[0])
        initial_state_data = [(ego_s, (ego["vx"]**2 + ego["vy"]**2)**0.5, 0.0, ego_r)]

        for i in range(1, len(vehicles_data)):
            x = vehicles_data[i]["x"]
            y = vehicles_data[i]["y"]
            vx = vehicles_data[i]["vx"]
            vy = vehicles_data[i]["vy"]
            cos_h = vehicles_data[i]["cos_h"]
            sin_h = vehicles_data[i]["sin_h"]
            s, r = self.map.convert_to_topo_position_with_reference((x,y,np.arctan2(sin_h, cos_h)), self.init_map_id[i])
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
            max_depth=5,
            num_sims=20
        )
        # return action
        return solver.plan()

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
        for i, row in enumerate(obs_array):
            # Map each row to a dictionary with feature names as keys.
            vehicle_dict = {feature: row[j] for j, feature in enumerate(features)}
            # Assign this dictionary to a vehicle key.
            vehicles_data.append(vehicle_dict)
        return vehicles_data
