# this is the optimization based agent

import logging
from gymnasium import spaces
from abc import ABC, abstractmethod
from ttrl_agent.configuration import Configurable

from pomdp_solver import *

logger = logging.getLogger(__name__)


class POMCPAgent(Configurable, ABC):
    def __init__(self, env, map, config=None):
        self.env = env
        self.map = map
        assert isinstance(env.action_space, spaces.Discrete) or isinstance(env.action_space, spaces.Tuple), \
            "Only compatible with Discrete action spaces."
        self.previous_state = None
        # TODO: init agent model here
        self.steps = 0

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
                      np.ndarray
        :return: [a0, a1, a2...], a sequence of actions to perform, where a is a int
                 ACTIONS_LONGI = {0: "STOP", 1: "SLOWER", 2: "IDLE", 3: "FASTER"}
        """
        self.previous_state = state
        vehicles_data = self.convert_obs_to_dict(state)
        ego = vehicles_data["vehicle_0"]
        # TODO: find an action based on observation
        for vehicle_info in vehicles_data:
            x = vehicle_info["x"]
            y = vehicle_info["y"]
            vx = vehicle_info["vx"]
            vy = vehicle_info["vy"]
            cos_h = vehicle_info["cos_h"]
            sin_h = vehicle_info["sin_h"]

        # return action

    def set_directory(self, directory):
        self.directory = directory

    def convert_obs_to_dict(self, obs_array):
        """
            Convert the observation output (numpy array) into a dictionary of dictionaries.

        :param obs_array: numpy array from the observe() method.
        :return: dict of dicts where each key is 'vehicle_i' and the value is a dict of features.
        """
        features = self.env.config["observation"]["features"]
        vehicles_data = {}
        for i, row in enumerate(obs_array):
            # Map each row to a dictionary with feature names as keys.
            vehicle_dict = {feature: row[j] for j, feature in enumerate(features)}
            # Assign this dictionary to a vehicle key.
            vehicles_data[f'vehicle_{i}'] = vehicle_dict

        return vehicles_data
