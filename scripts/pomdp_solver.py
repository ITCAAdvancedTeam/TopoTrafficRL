"""
This POMDP solver file is designed to address the intersection crossing problem
by using a particle-based Monte Carlo Planning with Progressive Widening (POMCPOW) solver.
It leverages models implemented in `pomdp_core.py`, which define the transition, observation,
and reward dynamics. The solver uses continuous states and actions to plan in an environment
where state uncertainty is managed by particles.

Classes:
    - POMCPOWSolver: The primary solver that plans actions based on belief particles.
    - Belief: Maintains the particle-based belief distribution and updates it with new observations.

Functions:
    - main: Initializes the problem components and runs a few steps of the solver for testing.
"""

import random
import numpy as np
from pomdp_core import *


class TreeNode:
    """Represents a node in the MCTS tree for the POMCPOW solver."""
    def __init__(self, belief, parent=None):
        self.belief = belief
        self.children = {}  # Maps actions to child nodes
        self.visit_count = 0
        self.value = 0.0
        self.parent = parent

    def update(self, reward):
        """Updates the node's value and visit count based on observed reward."""
        self.visit_count += 1
        self.value += (reward - self.value) / self.visit_count


class POMCPOWSolver:
    def __init__(self, belief, transition_model, observation_model, reward_model, policy_model, max_depth=3, num_sims=100):
        self.belief = belief
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.max_depth = max_depth
        self.num_sims = num_sims
        self.exploration_constant = 1.0  # UCB exploration constant

    def plan(self):
        root = TreeNode(self.belief)
        for _ in range(self.num_sims):
            self.simulate(root, depth=self.max_depth)
        return self.select_best_action(root)

    def progressive_widening(self, node):
        """Returns a progressively expanded set of actions in continuous space."""
        num_actions = int(np.log(self.num_sims) + 1)  # Determine number of actions to explore
        actions = [self.sample_continuous_action() for _ in range(num_actions)]
        return actions

    def sample_continuous_action(self):
        """Samples a continuous action value (e.g., acceleration) within a specified range."""
        action_value = random.uniform(-2.0, 2.0)  # Adjust bounds as needed
        return Action(action_value)

    def select_action(self, node):
        """Selects an action based on UCB score, considering exploration vs. exploitation."""
        def ucb_score(child_node):
            exploration = self.exploration_constant * np.sqrt(np.log(node.visit_count + 1) / (child_node.visit_count + 1))
            return child_node.value + exploration

        return max(node.children.keys(), key=lambda action: ucb_score(node.children[action]))

    def simulate(self, node, depth):
        if depth == 0:
            return 0

        # Choose action based on UCB
        actions = self.progressive_widening(node)
        action = self.select_action(node) if node.children else random.choice(actions)

        # Expand node if this action hasn't been tried yet
        if action not in node.children:
            next_belief = self.belief.update(action, self.observation_model.sample(self.belief.particles[0], action), self.observation_model)
            child_node = TreeNode(next_belief, parent=node)
            node.children[action] = child_node
            reward = self.rollout(next_belief, depth - 1)
            node.update(reward)
            return reward

        # Otherwise, simulate recursively down the tree
        next_node = node.children[action]
        next_state = self.transition_model.sample(self.belief.particles[0], action)
        observation = self.observation_model.sample(next_state, action)
        self.belief.update(action, observation, self.observation_model)

        immediate_reward = self.reward_model.sample(self.belief.particles[0], action, next_state)
        future_reward = self.simulate(next_node, depth - 1)

        total_reward = immediate_reward + 0.95 * future_reward
        node.update(total_reward)
        return total_reward

    def rollout(self, belief, depth):
        """Simulates a random rollout to estimate reward for unexplored nodes."""
        if depth == 0:
            return 0

        action = self.policy_model.rollout(belief)  # Use the policy model's rollout
        next_state = self.transition_model.sample(belief.particles[0], action)
        immediate_reward = self.reward_model.sample(belief.particles[0], action, next_state)
        return immediate_reward + 0.95 * self.rollout(belief, depth - 1)

    def select_best_action(self, root):
        """Selects the best action from the root node based on visit counts."""
        best_action = max(root.children, key=lambda action: root.children[action].visit_count)
        return best_action


class Belief:
    """Represents the belief state with particle-based filtering."""
    def __init__(self, num_particles, initial_state, transition_model):
        self.particles = [initial_state] * num_particles
        self.transition_model = transition_model

    def update(self, action, observation, observation_model):
        new_particles = []
        for particle in self.particles:
            # Sample next state based on transition model and weight by observation likelihood
            next_particle = self.transition_model.sample(particle, action)
            obs_prob = observation_model.probability(observation, next_particle, action)
            if obs_prob > 0:
                new_particles.append(next_particle)

        # Resample particles if none remain after filtering
        if not new_particles:
            new_particles = [self.transition_model.sample(random.choice(self.particles), action) for _ in range(len(self.particles))]
        self.particles = new_particles


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
