"""This POMDP comparison class is the classic formulation of intersection crossing problem

States: [S0,S1,...,Sk], math:`S\subseteq[\mathbb{R},\mathbb{R},\mathbb{R},\mathbb{N}]^(k+1)`
Actions: a0, math:`A\subseteq[\mathbb{R}`
Rewards:
    - Crush reward
    - velocity reward
    - acceleration reward
Observations: [o0,o1,...,ok], math:`S\subseteq[\mathbb{R},\mathbb{R},\mathbb{R},\mathbb{R}]^(k+1)`

"""

import random
import numpy as np
from scipy.stats import norm

TSTEP = 0.1

class TopoMap:
    """
    Map consist of waypoints, each waypoint is a section of lane in the intersection area
    There is no conflicting point inside a waypoint
    """
    def __init__(self):
        self.waypoints = {} # Dictionary to map int to an array of waypoints, each being (x, y, yaw)
        self.topology = {}  # Dictionary to map int to a list of next waypoint IDs
        self.conflict = {}  # Dictionary to map int to int

    def add_waypoints(self, waypoint_id, points):
        """
        Adds a series of waypoints (each with x, y, yaw) to the map.

        Args:
            waypoint_id (int): Unique identifier for the waypoint sequence.
            points (list of tuples): List of tuples, each containing (x, y, yaw) coordinates.
        """
        # Convert points to a NumPy array if it's not already
        self.waypoints[waypoint_id] = np.array(points)
        if waypoint_id not in self.topology:
            self.topology[waypoint_id] = []

    def add_connection(self, from_id, to_id):
        """
        Adds a topological connection from one waypoint sequence to another.

        Args:
            from_id (int): ID of the waypoint sequence from which the connection starts.
            to_id (int): ID of the waypoint sequence to which the connection goes.
        """
        if from_id in self.waypoints and to_id in self.waypoints:
            self.topology[from_id].append(to_id)
        else:
            raise ValueError("Both waypoint sequences must exist in the map to create a connection.")
    def add_confliction(self, conflict_id1, conflict_id2):
        """
        Adds a confliction between two waypoints.

        Args:
            conflict_id1 (int): ID of the waypoint sequence.
            conflict_id2 (int): ID of the waypoint sequence.
        """
        self.conflict[conflict_id1] = conflict_id2
        self.conflict[conflict_id2] = conflict_id1

    def get_waypoints(self, waypoint_id):
        """
        Retrieves a series of waypoints for a given waypoint ID.

        Args:
            waypoint_id (int): Unique identifier for the waypoint sequence.

        Returns:
            np.ndarray: Array with shape (n, 3) where each row is (x, y, yaw), or None if not found.
        """
        return self.waypoints.get(waypoint_id, None)

    def get_next_waypoints(self, waypoint_id):
        """
        Retrieves the next connected waypoint sequences for a given waypoint sequence ID.

        Args:
            waypoint_id (int): ID of the current waypoint sequence.

        Returns:
            list: List of next waypoint sequence IDs, or an empty list if none are found.
        """
        if waypoint_id not in self.topology:
            print(f"Warning: waypoint_id {waypoint_id} not found in topology.")
            return []

        return self.topology[waypoint_id]

    def find_waypoint_by_length(self, waypoint_id, distance_from_end):
        """
        Finds the waypoint at a given distance from the end of a path.

        Args:
            waypoint_id (int): ID of the waypoint sequence to search.
            distance_from_end (float): Distance along the path from the end.

        Returns:
            np.ndarray: The (x, y, yaw) of the found waypoint, or None if waypoint_id is not found.
        """
        waypoints = self.waypoints.get(waypoint_id)
        if waypoints is None:
            return None  # Return None if waypoint_id is not found

        # Calculate cumulative distance from the end
        cumulative_distance = 0.0
        for i in range(len(waypoints) - 1, 0, -1):
            # Distance between consecutive waypoints
            segment_length = np.linalg.norm(waypoints[i][:2] - waypoints[i - 1][:2])
            cumulative_distance += segment_length
            if cumulative_distance >= distance_from_end:
                return waypoints[i]  # Return waypoint at this cumulative distance
        return waypoints[0]  # Return start if distance exceeds total path length

    def find_length_by_waypoint(self, waypoint_id):
        """
        Finds the total length of a waypoint path defined by waypoint_id.

        Args:
            waypoint_id (int): ID of the waypoint sequence to search.

        Returns:
            float: Total path length. Returns 0 if waypoint_id not found or has fewer than 2 points.
        """
        waypoints = self.waypoints.get(waypoint_id)
        if waypoints is None or len(waypoints) < 2:
            return 0.0  # Return 0 if no waypoints are found or there aren't enough points to measure length

        length = 0.0
        for i in range(1, len(waypoints)):
            # Calculate the Euclidean distance between consecutive waypoints
            segment_length = np.linalg.norm(waypoints[i][:2] - waypoints[i - 1][:2])
            length += segment_length

        return length

    def __str__(self):
        # String representation for easy visualization
        return f"TopoMap(waypoints={self.waypoints}, topology={self.topology})"


# State space: {[s,v,a,r]}
class State():
    def __init__(self, data, terminate=False):
        """
        data (list of tuple): Each tuple contains (double, double, double, int) representing [s, v, a, r].
        terminate (bool): Indicates if it's a terminate state.
        """
        # Ensure that `data` is a list of tuples with the correct structure
        if not isinstance(data, list) or not all(isinstance(t, tuple) and len(t) == 4 for t in data):
            raise ValueError("Data must be a list of tuples, each containing (double, double, double, int).")

        self.data = data
        self.terminate = terminate

    def __hash__(self):
        # Convert data to a hashable tuple structure
        return hash(tuple(self.data))

    def __eq__(self, other):
        if isinstance(other, State):
            # Compare lists of tuples element-wise
            return self.data == other.data
        return False

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"State(data={self.data})"


# Action space: a0
class Action():
    """The action is a single velocity value."""

    def __init__(self, data):
        """
        Initializes an action with a single velocity.

        Args:
            data (float): A single velocity as a double.
        """
        # Convert data to a float
        self.data = float(data)

    def __hash__(self):
        # Hash directly with the float value
        return hash(self.data)

    def __eq__(self, other):
        if isinstance(other, Action):
            return abs(self.data - other.data) < 0.01
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Action(data={self.data})"



# Observation space: {[x,y,vx,vy]}
class Observation():
    def __init__(self, data):
        """
        Initializes an observation with a list of tuples.

        Args:
            data (list of tuple): Each tuple should contain (double, double, double, double) representing [x, y, vx, vy].
        """
        # Ensure data is a list of tuples with the correct structure
        if not isinstance(data, list) or not all(isinstance(t, tuple) and len(t) == 4 for t in data):
            raise ValueError("Data must be a list of tuples, each containing (double, double, double, double).")
        self.data = data

    def __hash__(self):
        # Convert data to a hashable tuple structure
        return hash(tuple(self.data))

    def __eq__(self, other):
        if isinstance(other, Observation):
            if len(self.data) != len(other.data):
                return False
            # Compare lists of tuples element-wise with tolerances
            for (x1, y1, vx1, vy1), (x2, y2, vx2, vy2) in zip(self.data, other.data):
                if not (abs(x1 - x2) < 0.1 and abs(y1 - y2) < 0.1 and
                        abs(vx1 - vx2) < 0.1 and abs(vy1 - vy2) < 0.1):
                    return False
            return True
        return False


    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Observation(data={self.data})"


# Observation model
class ObservationModel():
    """
    The ObservationModel class defines the observation dynamics of a POMDP, determining
    how likely a particular observation is given the current state and an action. It also
    generates observations from the next state for simulation purposes.

    Attributes:
        map (TopoMap): The map used for finding waypoints and paths for observations.
        noise (float): Noise parameter to introduce randomness in the observation sampling.

    Methods:
        probability(observation, next_state, action):
            Computes the probability of observing `observation` given `next_state` and `action`.
            The probability sconsiders the angular difference and distance between the observation
            and the next state's corresponding waypoint.

        sample(next_state, action, argmax=False):
            Generates a sample observation based on `next_state` and `action`. If `argmax` is
            True, the most likely observation is returned without added noise.

        argmax(next_state, action):
            Returns the most likely observation for a given `next_state` and `action` by
            calling `sample` with `argmax=True`.
    """
    def __init__(self, map=None, noise_level=0.1):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided
        self.noise_level = noise_level  # Noise parameter to control randomness in sampling

    def probability(self, observation, next_state, action):
        # print("observation prob") # debug
        p = []
        k = len(observation.data)
        for i in range(k):
            waypoint_id = next_state.data[i][3]
            point = self.map.find_waypoint_by_length(waypoint_id, next_state.data[i][0])
            v_angle = np.arctan2(observation.data[i][3], observation.data[i][2])
            f1 = (v_angle - point[2] + np.pi) % (2 * np.pi) - np.pi
            f2 = f2 = np.linalg.norm(np.array(observation.data[i][:2]) - point[:2])
            p1 = norm.pdf(f1, loc=0, scale=0.8)
            p2 = norm.pdf(f2, loc=0, scale=4.0)
            p.append(p1 * p2)
        p = np.array(p)
        pa = np.exp(np.sum(np.log(p)))
        pb = norm.pdf(next_state.data[0][2] - action.data, loc=0, scale=0.5)
        return pa * pb

    def sample(self, next_state, action, argmax=False):
        data = []
        k = len(next_state.data)
        for i in range(k):
            # Assuming `waypoint_id` is stored in `next_state[i][3]`
            waypoint_id = next_state.data[i][3]

            # Assuming `distance_from_end` is stored in `next_state[i][0]`
            distance_from_end = next_state.data[i][0]

            # Find the waypoint based on the given distance from the end
            waypoint = self.map.find_waypoint_by_length(waypoint_id, distance_from_end)

            # Return an empty observation if the waypoint does not exist
            if waypoint is None:
                return Observation([])

            x, y, yaw = waypoint
            # Introduce Gaussian noise to the position
            if i != 0:
                x += norm.rvs(scale=self.noise_level * 2)
                y += norm.rvs(scale=self.noise_level * 2)
            # Determine velocity magnitude
            v_magnitude = next_state.data[i][1]
            if not argmax and i != 0:
                v_magnitude += norm.rvs(scale=self.noise_level)  # Small noise for velocity magnitude

            # Calculate the velocity components based on yaw
            vx = v_magnitude * np.cos(yaw)  # X-component of velocity
            vy = v_magnitude * np.sin(yaw)  # Y-component of velocity

            # Append the (x, y, vx, vy) tuple to the data list
            data.append((x, y, vx, vy))

        # debug
        # print("Sampling Observation for state:", next_state)
        # print("Generated observation data:", data)

        return Observation(data)

    def argmax(self, next_state, action):
        """Returns the most likely observation"""
        return self.sample(next_state, action, argmax=True)


# Transition Model
class TransitionModel():
    """
    The TransitionModel class defines the transition dynamics of a POMDP, describing
    how the state evolves given an action. This model determines the probability of
    transitioning from the current state to a next state based on the ego vehicle's
    behavior and potential conflicts with other vehicles.

    Attributes:
        map (TopoMap): The map used to retrieve waypoints, paths, and distances.

    Methods:
        probability(next_state, state, action):
            Calculates the probability of transitioning from `state` to `next_state`
            given an `action`. The probability considers the ego vehicle's change in
            velocity and potential conflicts with other vehicles based on Time-To-Collision (TTC).

        sample(state, action):
            Generates a sample of the next state from the current `state` when an
            `action` is applied. The next state is computed using kinematic equations
            and path updates for each vehicle.

        argmax(state, action):
            Returns the most likely `next_state` based on the given `state` and `action`.
            This method currently returns a sample state that represents the most probable
            outcome according to the defined transition dynamics.

        Note:
            rewrite the transition model when changing environment. If the transition model is define
            in here, it means that the environment has an explicit formulation.

    """
    def __init__(self, map=None):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided

    def probability(self, next_state, state, action):
        """probability of conflicting vehicle's reaction toward ego"""
        # ego
        pa = norm.pdf(action.data - next_state.data[0][2], loc=0, scale=1.0)

        # confliction
        p = []
        k = len(state.data)
        ego_list = [state.data[0][3]] + self.map.get_next_waypoints(state.data[0][3]) # TODO: extend to a certain distance
        TTC = []
        for id in ego_list:
            len_to_end = state.data[0][0] if id == state.data[0][3] else state.data[0][0] + self.map.find_length_by_waypoint(id)
            if action.data == 0:
                ttc = len_to_end / state.data[0][1]
            else:
                ttc = (np.sqrt(2 * action.data * len_to_end + state.data[0][1]**2) - state.data[0][1]) / action.data
            # deal with the case the vehicle decc to zero
            if state.data[0][1] + action.data * ttc < 0.0:
                ttc = float('inf')  # Set TTC to infinity if velocity is zero
            TTC.append((id, ttc))

        for i in range(1, k):
            conflict_waypoint_id = self.map.conflict.get(state.data[i][3])
             # Check if there is a conflict and it exists in ego_list
            if conflict_waypoint_id in ego_list:
                # Find corresponding TTC for the conflict waypoint in ego's TTC list
                ego_ttc = next((ttc for waypoint_id, ttc in TTC if waypoint_id == conflict_waypoint_id), None)

                # If TTC is not found, skip this conflict
                if ego_ttc is None:
                    continue

                 # Calculate TTC for the conflicting vehicle
                conflict_len_to_end = state.data[i][0]
                conflict_acc = state.data[i][2]
                # Estimate a TTC for the conflicting vehicle
                if conflict_acc == 0:
                    conflict_ttc = conflict_len_to_end / state.data[i][1]
                else:
                    conflict_ttc = (np.sqrt(2 * conflict_acc * conflict_len_to_end + state.data[i][1]**2) - state.data[i][1]) / conflict_acc
                # deal with the case the vehicle decc to zero
                if state.data[0][1] + action.data * ttc < 0.0:
                    ttc = float('inf')  # Set TTC to infinity if velocity is zero
                # Calculate probability based on TTC difference
                ttc_diff = np.abs(ego_ttc - conflict_ttc)
                # Calculate probability based on TTC difference, handling inf case
                if np.isinf(ego_ttc) or np.isinf(conflict_ttc):
                    p_i = 1.0  # No imminent collision, set p_i to 1
                else:
                    ttc_diff = np.abs(ego_ttc - conflict_ttc)
                    p_i = norm.pdf(ttc_diff, loc=0, scale=2.0)
                p.append(p_i)

        p = np.array(p)
        pb = np.exp(np.sum(np.log(p)))
        return pa * pb

    def sample(self, state, action):
        # Debugging: check if it actually generates randomized states
        next_state_candidates = []
        k = len(state.data)
        random_cnt = 3

        # print(f"[DEBUG] Generating {random_cnt} randomized configurations for next states")

        # Generate multiple action configurations
        for config_idx in range(random_cnt):
            # Fix the ego vehicle's action, randomize actions for conflicting vehicles
            actions = [action.data] + [random.uniform(-2.0, 2.0) for _ in range(1, k)]
            # print(f"[DEBUG] Configuration {config_idx + 1}: Actions = {actions}")

            # Generate the next state based on this configuration
            candidate_state_data = []
            next_terminal = state.terminate

            for i in range(k):
                # Calculate new position based on current velocity
                new_s = state.data[i][0] - state.data[i][1] * TSTEP
                if new_s < 0.0:  # Check if vehicle reaches end of path segment
                    next_list = self.map.get_next_waypoints(state.data[i][3])
                    if not next_list:  # No more waypoints
                        new_r = state.data[i][3]
                        new_s = 0.0
                        if i == 0:
                            next_terminal = True  # Mark as terminal if ego vehicle
                    else:
                        new_r = random.choice(next_list)
                        new_s += self.map.find_length_by_waypoint(new_r)  # Wrap to next waypoint
                else:
                    new_r = state.data[i][3]

                # Set acceleration for ego and conflicting vehicles
                new_a = actions[i]
                new_v = state.data[i][1] + new_a * TSTEP

                # Append data for this vehicle
                data = (new_s, new_v, new_a, new_r)
                candidate_state_data.append(data)
                # print(f"[DEBUG] Vehicle {i} -> New Position: {new_s}, New Velocity: {new_v}, Acceleration: {new_a}, Waypoint: {new_r}")

            # Calculate the probability for this candidate state
            candidate_state = State(candidate_state_data, next_terminal)
            prob = self.probability(candidate_state, state, action)
            # print(f"[DEBUG] Probability of Configuration {config_idx + 1}: {prob}")

            # Append candidate state with its probability
            next_state_candidates.append((candidate_state, prob))

        # Select the candidate state with the highest probability
        best_state, best_prob = max(next_state_candidates, key=lambda x: x[1])
        # print(f"[DEBUG] Selected State with Highest Probability: {best_prob}")

        return best_state

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)


# Reward Model
class RewardModel():
    """
    The RewardModel class calculates the reward for transitioning from a given state
    to a next state when an action is taken. The reward considers various factors
    such as collision risk, speed alignment, and comfort (acceleration).

    Attributes:
        map (TopoMap): The map used to find waypoints and assess distances.
        d_safe (float): The safe distance threshold for collision risk.
        speed_limit (list of float): The minimum and maximum speed limits.
        acceleration_limit (list of float): The allowable acceleration range.
        K1 (float): The weight for the collision risk reward.
        K2 (float): The weight for the velocity alignment reward.
        K3 (float): The weight for the acceleration comfort reward.
    """
    def __init__(self, map=None):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided
        self.d_safe = 3.0
        self.speed_limit = [4.0, 12.0]
        self.acceleration_limit = [-2.0, 2.0]
        self.K1 = 20.0 # collision reward
        self.K2 = 10.0 # velocity reward
        self.K3 = 10.0 # acceleration reward
    def sample(self, state, action, next_state):
        # deterministic
        if state.terminate:
            return 100  # reach target and terminated
        R1 = [1]
        R2 = [norm.pdf(state.data[0][1] + action.data * TSTEP, loc=8.0, scale=4.0)]
        R3 = [norm.pdf(action.data, loc=0.0, scale=2.0)]
        k = len(state.data)
        ego_point = self.map.find_waypoint_by_length(state.data[0][3], state.data[0][0])
        ego_list = [state.data[0][3]] + self.map.get_next_waypoints(state.data[0][3])
        for i in range(1, k):
            conflict_waypoint_id = self.map.conflict.get(state.data[i][3])
             # Check if there is a conflict and it exists in ego_list
            if conflict_waypoint_id in ego_list:
                pointi = self.map.find_waypoint_by_length(state.data[i][3], state.data[0][0])
                di = np.linalg.norm(ego_point[:2] - pointi[:2])
                r1 = 1 / (1 + np.exp(2 * (self.d_safe - di))) - 0.5
                r2 = norm.pdf(state.data[i][1], loc=8.0, scale=4.0)
                r3 = norm.pdf(state.data[i][1], loc=0.0, scale=2.0)
                R1.append(r1)
                R2.append(r2)
                R3.append(r3)
        reward = self.K1 * np.prod(R1) ** (1 / len(R1)) + self.K2 * np.prod(R2) ** (1 / len(R2)) + self.K3 * np.prod(R3) ** (1 / len(R3))
        # print("reward: ", reward)
        return reward

# Policy Model
class PolicyModel():
    """The policy should favor 1. keep speed (v) 2. comfort (a)"""

    ACTIONS = [Action(s) for s in {-2.0, -1.0, 0.0, 1.0, 2.0}]

    def sample(self, state):
        # print("policy sample")  # debug
        action_probabilities = self._calculate_action_probabilities(state)
        # print("Action Probabilities:", action_probabilities)  # Print the calculated probabilities

        action_probabilities /= action_probabilities.sum()
        chosen_action = np.random.choice(self.ACTIONS, p=action_probabilities)
        # print("Chosen Action:", chosen_action)  # Print the chosen action based on probabilities

        return chosen_action


    def _calculate_action_probabilities(self, state):
        """
        Calculates the probabilities for actions that favor:
        1. Keeping speed (v) close to 8.
        2. Ensuring comfort (small acceleration/deceleration).

        Returns:
            A list of probabilities for each action in ACTIONS.
        """
        v0 = state.data[0][1]
        a0 = state.data[0][2]
        av = np.clip((8.0 - v0) / (10 * TSTEP), -2, 2)
        action_probabilities = []
        for action in self.ACTIONS:
            p1 = norm.pdf(action.data, loc=a0, scale=2.0)
            p2 = norm.pdf(action.data, loc=av, scale=4.0)
            action_probabilities.append(p1 + p2)

        # Normalize to create a probability distribution
        action_probabilities = np.array(action_probabilities)
        return action_probabilities

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS
