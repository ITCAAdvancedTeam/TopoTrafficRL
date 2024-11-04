"""This POMDP comparison class is modified based on the rocksample problem: project_ws/venv/lib64/python3.10/site-packages/pomdp_py/problems/rocksample/rocksample_problem.py.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

States: [S0,S1,...,Sk], math:`S\subseteq[\mathbb{R},\mathbb{R},\mathbb{R},\mathbb{N}]^(k+1)`
Actions: a0, math:`A\subseteq[\mathbb{R}`
Rewards:
    - Crush reward
    - velocity reward
    - acceleration reward
Observations: [o0,o1,...,ok], math:`S\subseteq[\mathbb{R},\mathbb{R},\mathbb{R}]^(k+1)`

Note that in this example, it is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more
complicated examples.)
"""

import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
from scipy.stats import norm
import sys
import copy

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
        return self.topology.get(waypoint_id, [])

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
class State(pomdp_py.State):
    def __init__(self, data, terminal=False):
        """
        data (list of tuple): Each tuple contains (double, double, double, int) representing [s, v, a, r].
        terminal (bool): Indicates if it's a terminal state.
        """
        # Ensure that `data` is a list of tuples with the correct structure
        if not isinstance(data, list) or not all(isinstance(t, tuple) and len(t) == 4 for t in data):
            raise ValueError("Data must be a list of tuples, each containing (double, double, double, int).")

        self.data = data
        self.terminal = terminal

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
class Action:
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
            return self.data == other.data
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Action(data={self.data})"



# Observation space: {[x,y,vx,vy,a]}
class Observation:
    def __init__(self, data):
        """
        Initializes an observation with a list of tuples.

        Args:
            data (list of tuple): Each tuple should contain (double, double, double, double, double) representing [x, y, vx, vy, a].
        """
        # Ensure data is a list of tuples with the correct structure
        if not isinstance(data, list) or not all(isinstance(t, tuple) and len(t) == 5 for t in data):
            raise ValueError("Data must be a list of tuples, each containing (double, double, double, double, double).")
        self.data = data

    def __hash__(self):
        # Convert data to a hashable tuple structure
        return hash(tuple(self.data))

    def __eq__(self, other):
        if isinstance(other, Observation):
            # Compare lists of tuples element-wise
            return self.data == other.data
        return False

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Observation(data={self.data})"



# Observation model
class ObservationModel(pomdp_py.ObservationModel):
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
            The probability considers the angular difference and distance between the observation
            and the next state's corresponding waypoint.

        sample(next_state, action, argmax=False):
            Generates a sample observation based on `next_state` and `action`. If `argmax` is
            True, the most likely observation is returned without added noise.

        argmax(next_state, action):
            Returns the most likely observation for a given `next_state` and `action` by
            calling `sample` with `argmax=True`.
    """
    def __init__(self, map=None, noise=0.15):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided
        self.noise = noise  # Noise parameter to control randomness in sampling

    def probability(self, observation, next_state, action):
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
        pb = norm.pdf(observation.data[0][4] - action.data, loc=0, scale=1.0)
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
            # Determine velocity magnitude: use `action.data` if `i == 0`, else use `next_state[i][2]`
            v_magnitude = action.data if i == 0 else next_state.data[i][2]

            # Calculate the velocity components based on yaw
            vx = v_magnitude * np.cos(yaw)  # X-component of velocity
            vy = v_magnitude * np.sin(yaw)  # Y-component of velocity

            # Calculate acceleration with some random noise around `next_state[i][2]`
            if argmax:
                acceleration = next_state.data[i][2]
            else:
                acceleration = norm.rvs(loc=next_state.data[i][2], scale=0.1)

            # Append the (x, y, vx, vy, acceleration) tuple to the data list
            data.append((x, y, vx, vy, acceleration))

        data = np.array(data)
        return Observation(data)

    def argmax(self, next_state, action):
        """Returns the most likely observation"""
        return self.sample(next_state, action, argmax=True)


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
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
    """
    def __init__(self, map=None):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided

    def probability(self, next_state, state, action):
        """probability of conflicting vehicle's reaction toward ego"""
        # ego
        pa = norm.pdf(state.data[0][2] - next_state.data[0][2], loc=0, scale=1.0)

        # confliction
        p = []
        k = len(state.data)
        ego_list = [state.data[0][3]] + self.map.get_next_waypoints(state.data[0][3])
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
        next_state = []
        k = len(state.data)
        for i in range(k):
            # Create the data tuple according to stype
            new_s = state.data[i][0] - state.data[i][1] * TSTEP
            if new_s < 0.0:
                new_r = random.choice(self.map.get_next_waypoints(state.data[i][3]))
                new_s = self.map.find_length_by_waypoint(new_r) + new_s
            else:
                new_r = state.data[i][3]
            new_v = state.data[i][1] + state.data[i][2] * TSTEP
            new_a = state.data[i][2]
            data = (new_s, new_v, new_a, new_r)
            next_state.append(data)
        next_state = np.array(next_state, dtype=stype)
        return State(next_state)

    def argmax(self, state, action):
        """Returns the most likely next state"""
        return self.sample(state, action)


# Reward Model
class RewardModel(pomdp_py.RewardModel):
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
        if state.terminal:
            return 0  # terminated. No reward
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
        return self.K1 * np.prod(R1) ** (1 / len(R1)) + self.K2 * np.prod(R2) ** (1 / len(R2)) + self.K3 * np.prod(R3) ** (1 / len(R3))


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """The policy should favor 1. keep speed (v) 2. comfort (a)"""

    ACTIONS = [Action(s) for s in {-2.0, -1.0, 0.0, 1.0, 2.0}]

    def sample(self, state):
        action_probabilities = self._calculate_action_probabilities(state)
        # Select an action based on these weighted probabilities
        chosen_action = np.random.choice(self.ACTIONS, p=action_probabilities)
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
            p1 = norm.pdf(action, loc=a0, scale=2.0)
            p2 = norm.pdf(action, loc=av, scale=4.0)
            action_probabilities.append(p1 + p2)

        # Normalize to create a probability distribution
        action_probabilities = np.array(action_probabilities)
        return action_probabilities

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


# Problem definition
class IntersectionProblem(pomdp_py.POMDP):
    """
    In fact, creating a IntersectionProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief, map):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(map=map),
            ObservationModel(map=map, noise=obs_noise),
            RewardModel(map=map),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(map=map), RewardModel(map=map))
        super().__init__(agent, env, name="IntersectionProblem")

    def print_state(self):
        string = "\n______ID______\n"
        print(string)


def test_planner(intersection_problem, planner, nsteps=3, discount=0.95):
    """TODO"""
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        print("==== Step %d ====" % (i + 1))
        action = planner.plan(intersection_problem.agent)
        # pomdp_py.visual.visualize_pouct_search_tree(intersection_problem.agent.tree,
        #                                             max_depth=5, anonymize=False)

        true_state = copy.deepcopy(intersection_problem.env.state)
        env_reward = intersection_problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(intersection_problem.env.state)

        real_observation = intersection_problem.env.provide_observation(
            intersection_problem.agent.observation_model, action
        )
        intersection_problem.agent.update_history(action, real_observation)
        planner.update(intersection_problem.agent, action, real_observation)
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        intersection_problem.print_state()

        if intersection_problem.in_exit_area(intersection_problem.env.state.position):
            break
    return total_reward, total_discounted_reward


def init_particles_belief(num_particles, init_state):
    """ TODO """
    num_particles = 200
    particles = []
    for _ in range(num_particles):
        particles.append(State(init_state, False))
    init_belief = pomdp_py.Particles(particles)
    return init_belief


def interpolate_line_with_yaw(start_point, end_point, num_points=20):
    """
    Interpolates a line segment between two points with yaw values and returns a list of (x, y, yaw) tuples.

    Parameters:
    start_point (tuple): Coordinates and yaw of the start point (x1, y1, yaw1).
    end_point (tuple): Coordinates and yaw of the end point (x2, y2, yaw2).
    num_points (int): Number of interpolated points. Default is 20.

    Returns:
    list: A list of tuples, each containing (x, y, yaw).
    """
    # Extract start and end coordinates and yaw
    x1, y1, yaw1 = start_point
    x2, y2, yaw2 = end_point

    # Generate interpolated x, y, and yaw values
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)
    yaw_values = np.linspace(yaw1, yaw2, num_points)

    # Combine x, y, and yaw into a list of tuples
    interpolated_points = [(x, y, yaw) for x, y, yaw in zip(x_values, y_values, yaw_values)]

    return interpolated_points

def simple_no_left_4_way_intersection():
    map = TopoMap()
    l = 2.0 # half lane width

    in1 = interpolate_line_with_yaw([6*l, 1*l, np.pi], [4*l, 1*l, np.pi])
    in2 = interpolate_line_with_yaw([-1*l, 6*l, -np.pi/2], [-1*l, 4*l, -np.pi/2])
    in3 = interpolate_line_with_yaw([-6*l, -1*l, 0], [-4*l, -1*l, 0])
    in4 = interpolate_line_with_yaw([1*l, -6*l, np.pi/2], [1*l, -4*l, np.pi/2])

    in5 = interpolate_line_with_yaw([4*l, 1*l, np.pi], [1*l, 1*l, np.pi])
    in6 = interpolate_line_with_yaw([-1*l, 4*l, -np.pi/2], [-1*l, 1*l, -np.pi/2])
    in7 = interpolate_line_with_yaw([-4*l, -1*l, 0], [-1*l, -1*l, 0])
    in8 = interpolate_line_with_yaw([1*l, -4*l, np.pi/2], [1*l, -1*l, np.pi/2])

    mid1 = interpolate_line_with_yaw([1*l, 1*l, np.pi], [-1*l, 1*l, np.pi])
    mid2 = interpolate_line_with_yaw([-1*l, 1*l, -np.pi/2], [-1*l, -1*l, -np.pi/2])
    mid3 = interpolate_line_with_yaw([-1*l, -1*l, 0], [1*l, -1*l, 0])
    mid4 = interpolate_line_with_yaw([1*l, -1*l, np.pi/2], [1*l, 1*l, np.pi/2])

    out1 = interpolate_line_with_yaw([1*l, -1*l, 0], [4*l, -1*l, 0])
    out2 = interpolate_line_with_yaw([1*l, 1*l, np.pi/2], [1*l, 4*l, np.pi/2])
    out3 = interpolate_line_with_yaw([-1*l, 1*l, np.pi], [-4*l, 1*l, np.pi])
    out4 = interpolate_line_with_yaw([-1*l, -1*l, -np.pi/2], [-1*l, -4*l, -np.pi/2])

    out5 = interpolate_line_with_yaw([4*l, -1*l, 0], [6*l, -1*l, 0])
    out6 = interpolate_line_with_yaw([1*l, 4*l, np.pi/2], [1*l, 6*l, np.pi/2])
    out7 = interpolate_line_with_yaw([-4*l, 1*l, np.pi], [-6*l, 1*l, np.pi])
    out8 = interpolate_line_with_yaw([-1*l, -4*l, -np.pi/2], [-1*l, -6*l, -np.pi/2])

    turn1 = interpolate_line_with_yaw([4*l, 1*l, np.pi], [1*l, 4*l, np.pi/2])
    turn2 = interpolate_line_with_yaw([-1*l, 4*l, -np.pi/2], [-4*l, 1*l, -np.pi])
    turn3 = interpolate_line_with_yaw([-4*l, -1*l, 0], [-1*l, -4*l, -np.pi/2])
    turn4 = interpolate_line_with_yaw([1*l, -4*l, np.pi/2], [4*l, -1*l, 0])

    map.add_waypoints(0, in1)
    map.add_waypoints(1, in5)
    map.add_connection(0, 1)
    map.add_waypoints(2, turn1)
    map.add_connection(0, 2)

    map.add_waypoints(3, in2)
    map.add_waypoints(4, in6)
    map.add_connection(3, 4)
    map.add_waypoints(5, turn2)
    map.add_connection(3, 5)

    map.add_waypoints(6, in3)
    map.add_waypoints(7, in7)
    map.add_connection(6, 7)
    map.add_waypoints(8, turn3)
    map.add_connection(6, 8)

    map.add_waypoints(9, in4)
    map.add_waypoints(10, in8)
    map.add_connection(9, 10)
    map.add_waypoints(11, turn4)
    map.add_connection(9, 11)

    map.add_waypoints(12, mid1)
    map.add_connection(12, 1)
    map.add_waypoints(13, mid2)
    map.add_connection(13, 4)
    map.add_waypoints(14, mid3)
    map.add_connection(14, 7)
    map.add_waypoints(15, mid4)
    map.add_connection(15, 10)
    map.add_confliction(1, 15)
    map.add_confliction(4, 12)
    map.add_confliction(7, 13)
    map.add_confliction(10, 14)

    map.add_waypoints(16, out1)
    map.add_connection(14, 16)
    map.add_confliction(16, 11)
    map.add_waypoints(17, out2)
    map.add_connection(15, 17)
    map.add_confliction(17, 2)
    map.add_waypoints(18, out3)
    map.add_connection(12, 18)
    map.add_confliction(18, 5)
    map.add_waypoints(19, out4)
    map.add_connection(13, 19)
    map.add_confliction(19, 8)

    map.add_waypoints(20, out5)
    map.add_connection(11, 20)
    map.add_connection(16, 20)
    map.add_waypoints(21, out6)
    map.add_connection(2, 21)
    map.add_connection(17, 21)
    map.add_waypoints(22, out7)
    map.add_connection(5, 22)
    map.add_connection(18, 22)
    map.add_waypoints(23, out8)
    map.add_connection(8, 23)
    map.add_connection(19, 23)

    return map


def main():
    # TODO: create a init_belief and init_true_state
    obs_noise = 0.15
    map = simple_no_left_4_way_intersection()
    s0 = map.find_length_by_waypoint(9)
    s1 = map.find_length_by_waypoint(0)
    init_true_state_data = [(s0, 8.0, 0.0, 9), (s1, 8.0, 0.0, 0)]
    init_true_state = State(init_true_state_data)
    init_belief = init_particles_belief(num_particles=200, init_state=init_true_state)

    problem = IntersectionProblem(obs_noise, init_true_state, init_belief, map)

    print("** Testing POMCP **")
    problem.agent.set_belief(
        pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True
    )
    pomcp = pomdp_py.POMCP(
        max_depth=3,
        discount_factor=0.95,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=problem.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    tt, ttd = test_planner(problem, pomcp, nsteps=100, discount=0.95)


if __name__ == "__main__":
    main()
