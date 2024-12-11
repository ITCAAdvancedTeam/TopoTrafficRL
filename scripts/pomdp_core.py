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
import networkx as nx
import matplotlib.pyplot as plt


class TopoMap:
    """
    Map consist of waypoints, each waypoint is a section of lane in the intersection area
    There is no conflicting point inside a waypoint
    """
    def __init__(self):
        self.waypoints = {} # Dictionary to map int to an array of waypoints, each being (x, y, yaw)
        self.topology = {}  # Dictionary to map int to a list of next waypoint IDs
        self.conflict = {}  # Dictionary to map int to a list of conflicting waypoint IDs
        self.conflict_point = {} # Dictionary to map (int, int) to (float, float)

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
        # add itself into conflict
        self.conflict[waypoint_id] = [waypoint_id]
        self.conflict_point[(waypoint_id, waypoint_id)] = (0.0, 0.0)

    def add_connection(self, from_id, to_id):
        """
        Adds a topological connection from one waypoint sequence to another.

        Args:
            from_id (int): ID of the waypoint sequence from which the connection starts.
            to_id (int): ID of the waypoint sequence to which the connection goes.
        """
        if from_id in self.waypoints and to_id in self.waypoints:
            self.topology[from_id].append(to_id)
            # add next into conflict
            self.conflict[from_id].append(to_id)
            self.conflict_point[(from_id, to_id)] = (self.find_length_by_waypoint(to_id), 0.0)
        else:
            raise ValueError("Both waypoint sequences must exist in the map to create a connection.")

    def add_confliction(self, conflict_id1, conflict_id2):
        """
        Adds a confliction between two waypoints and assigns the conflict_point as
        the lengths to the end of the respective waypoint paths.

        Args:
            conflict_id1 (int): ID of the first waypoint sequence.
            conflict_id2 (int): ID of the second waypoint sequence.
        """
        # Avoid duplicate work
        if (conflict_id1, conflict_id2) in self.conflict_point:
            return

        # Find the closest points between the two waypoint arrays
        waypoints1 = self.waypoints[conflict_id1]
        waypoints2 = self.waypoints[conflict_id2]

        min_distance = float('inf')
        closest_index1 = None
        closest_index2 = None

        for idx1, point1 in enumerate(waypoints1):
            for idx2, point2 in enumerate(waypoints2):
                # Calculate Euclidean distance
                distance = np.linalg.norm(point1[:2] - point2[:2])  # Compare (x, y) only
                if distance < min_distance:
                    min_distance = distance
                    closest_index1 = idx1
                    closest_index2 = idx2
        if min_distance < 1.0:
            # Add conflict entries for both waypoints
            self.conflict[conflict_id1].append(conflict_id2)
            self.conflict[conflict_id2].append(conflict_id1)

            # Calculate lengths along the path to the end for both waypoints
            def calculate_path_length(waypoints, start_index):
                """Helper function to calculate path length from start_index to the end."""
                length = 0.0
                for i in range(start_index, len(waypoints) - 1):
                    length += np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2])
                return length

            length_to_end1 = calculate_path_length(waypoints1, closest_index1)
            length_to_end2 = calculate_path_length(waypoints2, closest_index2)

            # Store the conflict_point as the lengths to the end along the path
            self.conflict_point[(conflict_id1, conflict_id2)] = (length_to_end1, length_to_end2)
            self.conflict_point[(conflict_id2, conflict_id1)] = (length_to_end2, length_to_end1)

    def find_all_conflict(self):
        """
        Examines all waypoint pairs for conflicts using the add_confliction method.

        Iterates through all waypoint pairs and identifies conflicts between them.
        """
        waypoint_ids = list(self.waypoints.keys())
        for i in range(len(waypoint_ids)):
            for j in range(i + 1, len(waypoint_ids)):
                self.add_confliction(waypoint_ids[i], waypoint_ids[j])

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

    def convert_to_topo_position(self, position):
        """
        Finds the closest waypoint and its position on that waypoint.

        Args:
            position (tuple): A tuple (x, y, yaw) representing the position to check.

        Returns:
            tuple: A tuple (s, r), where:
                - s (float): Length to the end of the closest waypoint.
                - r (int): The ID of the closest waypoint.
        """
        def wrap_to_pi(angle):
            """ Wraps an angle in radians to the range [-π, π]. """
            return (angle + np.pi) % (2 * np.pi) - np.pi

        x, y, yaw = position  # Extract x, y, yaw from the position
        closest_waypoint_id = None
        closest_distance = float('inf')
        closest_index = None
        candidates = []  # List to store candidates within 0.5 distance

        # Iterate through all waypoints to find the closest
        for waypoint_id, points in self.waypoints.items():
            for idx, point in enumerate(points):
                # Calculate Euclidean distance
                distance = np.linalg.norm(np.array([x, y]) - point[:2])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_waypoint_id = waypoint_id
                    closest_index = idx
                # Add to candidates if distance is within 0.5
                if distance < 0.5:
                    candidates.append((waypoint_id, idx, point, distance))
                    break
        # If there are candidates, refine selection based on yaw difference
        if candidates:
            closest_yaw_diff = float('inf')
            best_candidate = None
            for waypoint_id, idx, point, distance in candidates:
                yaw_diff = np.abs(wrap_to_pi(yaw - point[2]))
                if yaw_diff < closest_yaw_diff:
                    closest_yaw_diff = yaw_diff
                    closest_waypoint_id = waypoint_id
                    closest_index = idx

        if closest_waypoint_id is None:
            raise ValueError("No suitable waypoint found in the map.")

        # Calculate length (s) to the end of the closest waypoint
        waypoints = self.waypoints[closest_waypoint_id]
        s = 0.0
        for i in range(closest_index, len(waypoints) - 1):
            segment_length = np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2])
            s += segment_length

        return s, closest_waypoint_id

    def convert_to_topo_position_with_reference(self, position, reference_id):
        """
        Finds the closest waypoint starting from a reference waypoint and traversing connected waypoints.

        Args:
            position (tuple): A tuple (x, y, yaw) representing the position to check.
            reference_id (int): The ID of the reference waypoint to start the search from.

        Returns:
            tuple: A tuple (s, r), where:
                - s (float): Length to the end of the closest waypoint.
                - r (int): The ID of the closest waypoint.
        """
        def wrap_to_pi(angle):
            """ Wraps an angle in radians to the range [-π, π]. """
            return (angle + np.pi) % (2 * np.pi) - np.pi

        x, y, yaw = position  # Extract x, y, yaw from the position
        closest_waypoint_id = None
        closest_distance = float('inf')
        closest_index = None
        candidates = []  # List to store candidates within 0.5 distance

        # Queue for BFS-like traversal
        queue = [reference_id]
        visited = set()

        # Traverse connected waypoints starting from the reference ID
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            # Process waypoints for the current ID
            points = self.waypoints.get(current_id)
            if points is None:
                continue

            for idx, point in enumerate(points):
                # Calculate Euclidean distance
                distance = np.linalg.norm(np.array([x, y]) - point[:2])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_waypoint_id = current_id
                    closest_index = idx
                # Add to candidates if distance is within 0.5
                if distance < 0.5:
                    candidates.append((current_id, idx, point, distance))
                    break

            # Add connected waypoints to the queue
            queue.extend(self.topology.get(current_id, []))

        # If there are candidates, refine selection based on yaw difference
        if candidates:
            closest_yaw_diff = float('inf')
            best_candidate = None
            for waypoint_id, idx, point, distance in candidates:
                yaw_diff = np.abs(wrap_to_pi(yaw - point[2]))
                if yaw_diff < closest_yaw_diff:
                    closest_yaw_diff = yaw_diff
                    closest_waypoint_id = waypoint_id
                    closest_index = idx

        if closest_waypoint_id is None:
            raise ValueError("No suitable waypoint found starting from the reference waypoint.")

        # Calculate length (s) to the end of the closest waypoint
        waypoints = self.waypoints[closest_waypoint_id]
        s = 0.0
        for i in range(closest_index, len(waypoints) - 1):
            segment_length = np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2])
            s += segment_length

        return s, closest_waypoint_id

    def trim_map(self, ids):
        """
        Creates a new TopoMap instance containing only waypoints and connections
        reachable from the given IDs.

        Args:
            ids (list or set): A list or set of starting waypoint IDs.

        Returns:
            TopoMap: A new TopoMap instance with trimmed waypoints, topology, and conflicts.
        """
        # Use a set to track reachable waypoints
        reachable = set(ids)

        # Use a queue to perform BFS traversal
        queue = list(ids)

        # Perform BFS to find all reachable waypoints
        while queue:
            current_id = queue.pop(0)
            # Extend to all connected waypoints in topology
            for next_id in self.topology.get(current_id, []):
                if next_id not in reachable:
                    queue.append(next_id)
                    reachable.add(next_id)

        # Ensure all input IDs are still reachable
        if not set(ids).issubset(reachable):
            raise ValueError("Some of the provided IDs are not reachable in the current map.")

        # Create a new TopoMap instance
        new_topomap = TopoMap()

        # Add reachable waypoints to the new TopoMap
        new_topomap.waypoints = {wp_id: points for wp_id, points in self.waypoints.items() if wp_id in reachable}
        new_topomap.topology = {wp_id: [conn for conn in connections if conn in reachable]
                                for wp_id, connections in self.topology.items() if wp_id in reachable}
        new_topomap.conflict = {wp_id: [conflict_id for conflict_id in conflicts if conflict_id in reachable]
                            for wp_id, conflicts in self.conflict.items() if wp_id in reachable}

        return new_topomap

    def draw_tree(self, filename="topology_graph.png"):
        """
        Visualizes the topology of the waypoints based on their coordinates and saves it to a file.

        Args:
            filename (str): The file name or path to save the graph image.
        """
        G = nx.DiGraph()  # Create a directed graph

        # Define positions for the nodes based on waypoint coordinates
        pos = {}  # Dictionary to store positions for the graph
        for waypoint_id, points in self.waypoints.items():
            if len(points) > 0:
                # Use the mean of all (x, y) positions in the waypoints for node position
                mean_x = np.mean(points[:, 0])
                mean_y = np.mean(points[:, 1])
                pos[waypoint_id] = (mean_x, mean_y)
                G.add_node(waypoint_id)  # Add the waypoint ID as a node

        # Add edges based on the topology dictionary
        for wp_id, next_ids in self.topology.items():
            for next_id in next_ids:
                if wp_id in pos and next_id in pos:  # Ensure both waypoints exist in pos
                    G.add_edge(wp_id, next_id)

        # Draw the graph using actual coordinates
        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos, with_labels=True, node_color='skyblue', node_size=200,
            edge_color='k', linewidths=1, font_size=12
        )

        # Add conflict points as text annotations
        for (wp_id1, wp_id2), (dist1, dist2) in self.conflict_point.items():
            # Calculate positions of conflict points based on distances to the end of the waypoints
            conflict_pos1 = self.find_waypoint_by_length(wp_id1, dist1)
            conflict_pos2 = self.find_waypoint_by_length(wp_id2, dist2)

            if conflict_pos1 is not None and conflict_pos2 is not None:
                # Take the midpoint of the conflict positions for annotation
                conflict_x = (conflict_pos1[0] + conflict_pos2[0]) / 2
                conflict_y = (conflict_pos1[1] + conflict_pos2[1]) / 2

                # Add text annotation for the conflict
                plt.text(
                    conflict_x, conflict_y,
                    f"Conflict {wp_id1}-{wp_id2}",
                    fontsize=8, color='red', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.3)
                )

        # Add waypoint labels
        labels = {key: f"{key}" for key in pos.keys()}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color="black")

        plt.title("Waypoint Topology with Actual Coordinates")
        plt.savefig(filename)  # Save the figure to a file
        plt.close()  # Close the plot to avoid displaying it

    def __str__(self):
        # String representation for easy visualization
        return f"TopoMap(waypoints={self.waypoints}, topology={self.topology})"


TSTEP = 0.1


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
        Initializes an action with a single velocity, clamping the input to [-2, 2]
        and rounding to the closest integer.

        Args:
            data (float or int): acceleration.
        """
        # Round to the nearest integer and clamp to the range [-2, 2]
        self.data = max(-2, min(2, round(data)))

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
            The probability considers the angular difference and distance between the observation
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
    def __init__(self, map=None, dt=TSTEP):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided
        self.dt = dt

    def probability(self, next_state, state, action):
        """probability of conflicting vehicle's reaction toward ego"""
        # ego
        pa = norm.pdf(action.data - next_state.data[0][2], loc=0, scale=1.0)

        # confliction
        p = []
        k = len(state.data)
        ego_list = [state.data[0][3]] + self.map.get_next_waypoints(state.data[0][3])

        for i in range(1, k):
            conflict_waypoint_id_list = self.map.conflict.get(state.data[i][3])
            # Check if there is a conflict and it exists in ego_list
            for conflict_waypoint_id in conflict_waypoint_id_list:
                if conflict_waypoint_id not in ego_list:
                    continue
                if (conflict_waypoint_id, state.data[i][3]) in self.map.conflict_point:
                    ego_dist, vel_dist = self.map.conflict_point[(conflict_waypoint_id, state.data[i][3])]
                    if conflict_waypoint_id == state.data[0][3]:
                        ego_dist = state.data[0][0] - ego_dist
                    else:
                        ego_dist = state.data[0][0] - ego_dist + self.map.find_length_by_waypoint(conflict_waypoint_id)
                    vel_dist = state.data[i][0] - vel_dist
                    if vel_dist * ego_dist < 0: # if pass conflict point
                        continue
                    if 2 * state.data[i][2] * vel_dist + state.data[i][1]**2 < 0 or np.abs(state.data[i][1]) < 0.001: # if vehicle stop before conflict point
                        continue
                    if 2 * state.data[0][2] * ego_dist + state.data[0][1]**2 < 0 or np.abs(state.data[0][1]) < 0.001: # if ego stop before conflict point
                        continue

                    # Find corresponding TTC for the conflict waypoint in ego's TTC list
                    if action.data == 0:
                        ego_ttc = ego_dist / state.data[0][1]
                    else:
                        ego_ttc = (np.sqrt(2 * action.data * ego_dist + state.data[0][1]**2) - state.data[0][1]) / action.data
                    if state.data[i][2] == 0:
                        vel_ttc = vel_dist / state.data[i][1]
                    else:
                        vel_ttc = (np.sqrt(2 * state.data[i][2] * vel_dist + state.data[i][1]**2) - state.data[i][1]) / state.data[i][2]

                    # Calculate probability based on TTC difference, handling inf case
                    if np.isinf(ego_ttc) or np.isinf(vel_ttc):
                        p_i = 1.0  # No imminent collision, set p_i to 1
                    else:
                        ttc_diff = np.abs(ego_ttc - vel_ttc)
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
            actions = [action.data] + [random.choice([-2, -1, 0, 1, 2]) for _ in range(1, k)]
            # print(f"[DEBUG] Configuration {config_idx + 1}: Actions = {actions}")

            # Generate the next state based on this configuration
            candidate_state_data = []
            next_terminal = state.terminate

            for i in range(k):
                # Calculate new position based on current velocity
                new_s = state.data[i][0] - state.data[i][1] * self.dt
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
                new_v = state.data[i][1] + new_a * self.dt

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
        vel_limit (list of float): The minimum and maximum speed limits.
        acc_limit (list of float): The allowable acceleration range.
        K1 (float): The weight for the collision risk reward.
        K2 (float): The weight for the velocity alignment reward.
        K3 (float): The weight for the acceleration comfort reward.
    """
    def __init__(self, map=None, dt=TSTEP, vel_range=[4.0, 14.0], acc_range=[-2.0, 2.0]):
        self.map = map if map is not None else TopoMap()  # Default to TopoMap if no map is provided
        self.d_safe = 3.0
        self.vel_limit = vel_range
        self.acc_limit = acc_range
        self.K1 = 10.0 # collision reward
        self.K2 = 15.0 # velocity reward
        self.K3 = 15.0 # acceleration reward
        self.dt = dt
    def sample(self, state, action, next_state):
        # deterministic
        if state.terminate:
            return 100  # reach target and terminated

        speed_mean = (self.vel_limit[0] + self.vel_limit[1]) / 2  # Midpoint of speed range
        speed_std = (self.vel_limit[1] - self.vel_limit[0]) / 2  # Approximation of scale
        accel_mean = (self.acc_limit[0] + self.acc_limit[1]) / 2  # Midpoint of acceleration range
        accel_std = (self.acc_limit[1] - self.acc_limit[0]) / 2  # Approximation of scale

        R1 = [1]
        R2 = [norm.pdf(state.data[0][1] + action.data * self.dt, loc=speed_mean, scale=speed_std)]
        R3 = [norm.pdf(action.data, loc=accel_mean, scale=accel_std)]
        k = len(state.data)
        ego_point = self.map.find_waypoint_by_length(state.data[0][3], state.data[0][0])
        ego_list = [state.data[0][3]] + self.map.get_next_waypoints(state.data[0][3])
        for i in range(1, k):
            conflict_waypoint_id = self.map.conflict.get(state.data[i][3])
             # Check if there is a conflict and it exists in ego_list
            if conflict_waypoint_id in ego_list:
                pointi = self.map.find_waypoint_by_length(state.data[i][3], state.data[i][0])
                di = np.linalg.norm(ego_point[:2] - pointi[:2])
                r1 = 1 / (1 + np.exp(2 * (self.d_safe - di))) - 0.5
                r2 = norm.pdf(state.data[i][1], loc=speed_mean, scale=speed_std)
                r3 = norm.pdf(state.data[i][2], loc=accel_mean, scale=accel_std)
                R1.append(r1)
                R2.append(r2)
                R3.append(r3)
        reward = self.K1 * np.prod(R1) ** (1 / len(R1)) + self.K2 * np.prod(R2) ** (1 / len(R2)) + self.K3 * np.prod(R3) ** (1 / len(R3))

        formatted_data = ", ".join([f"[{s:.2f}, {v:.2f}, {a:.2f}, {r}]" for s, v, a, r in state.data])
        R1_str = ", ".join([f"{r:.4f}" for r in R1])
        R2_str = ", ".join([f"{r:.4f}" for r in R2])
        R3_str = ", ".join([f"{r:.4f}" for r in R3])
        print(f" Particle: {formatted_data}")
        print(f"  reward: {reward:.4f}, R1: {R1_str}, R2: {R2_str}, R3: {R3_str}")

        # print(f"reward: R1 = {np.prod(R1) ** (1 / len(R1))}, R2 = {np.prod(R2) ** (1 / len(R2))}, R3 = {np.prod(R3) ** (1 / len(R3))}")
        return reward

# Policy Model
class PolicyModel():
    """The policy should favor 1. keep speed (v) 2. comfort (a)"""
    ACTIONS = [Action(s) for s in {-2.0, -1.0, 0.0, 1.0, 2.0}]
    def __init__(self, dt=TSTEP, target_speed=9.0):
        self.dt = dt
        self.target_speed = target_speed

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
        1. Keeping speed (v) close to target_speed.
        2. Ensuring comfort (small acceleration/deceleration).

        Returns:
            A list of probabilities for each action in ACTIONS.
        """
        v0 = state.data[0][1]
        a0 = state.data[0][2]
        av = np.clip((self.target_speed - v0) / (10 * self.dt), -2, 2)
        action_probabilities = []
        for action in self.ACTIONS:
            p1 = norm.pdf(action.data, loc=a0, scale=2.0)
            p2 = norm.pdf(action.data, loc=av, scale=2.0)
            action_probabilities.append(p1 + p2)

        # Normalize to create a probability distribution
        action_probabilities = np.array(action_probabilities)
        return action_probabilities

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS

# Helper function
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
