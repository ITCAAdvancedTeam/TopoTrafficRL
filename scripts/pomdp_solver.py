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
import copy

class TreeNode:
    """Represents a node in the MCTS tree for the POMCPOW solver."""
    def __init__(self, belief, parent=None):
        self.belief = copy.deepcopy(belief)
        self.children = {}  # Maps actions to child nodes
        self.visit_count = 0
        self.value = 0.0
        self.parent = parent
        self.observation_count = {}

    def update(self, reward):
        """Updates the node's value and visit count based on observed reward."""
        self.visit_count += 1
        self.value += (reward - self.value) / self.visit_count
        # print(f'[DEBUG] Updating TreeNode with reward: {reward} --> new visit_count: {self.visit_count}, new value: {self.value}')

    def print_tree(self, level=0):
        """
        Recursively prints the tree structure, showing each node's visit count and value.

        Args:
            level (int): The depth level of the current node, used for indentation.
        """
        indent = " + " * level  # Adjust indent for clarity
        print(f"{indent}Node(level={level}, visit_count={self.visit_count}, value={self.value})")

        for i, particle in enumerate(self.belief.particles[:3]):  # Print only the first few particles for brevity
            print(f"{indent}  Particle {i}: {particle}")

        # Recursively print each child node with its associated action
        for action, child in self.children.items():
            print(f"{indent}  Action: {action}")
            if child:
                child.print_tree(level + 1)
            else:
                print(f"{indent}    Node(level={level + 1}, visit_count=0, value=0.0) - Placeholder for unexpanded action")


class POMCPOWSolver:
    def __init__(self, belief, transition_model, observation_model, reward_model, policy_model, max_depth=3, num_sims=100):
        self.init_belief = belief
        self.init_action = Action(belief.particles[0].data[0][2])
        self.belief = copy.deepcopy(self.init_belief)
        self.action = copy.deepcopy(self.init_action)
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.policy_model = policy_model
        self.max_depth = max_depth
        self.num_sims = num_sims
        self.exploration_constant = 1.0  # UCB exploration constant
        self.ka = 1.0  # Controls the rate of action expansion
        self.alpha_a = 0.5  # Governs widening dependence on visits
        self.root = TreeNode(copy.deepcopy(belief))


    def plan(self):
        # print(f'[DEBUG] Starting plan method')

        # Run simulations, resetting to the initial root each time
        for _ in range(self.num_sims):
            # print("-----------------------------------------------------------")
            # self.root.print_tree()
            # print("-----------------------------------------------------------")
            current_node = self.root  # Start each simulation from the initial root
            self.belief = self.init_belief
            self.action = self.init_action
            self.simulate(current_node, depth=self.max_depth)

        # Select the best action from the root node after simulations
        return self.select_best_action(self.root)

    def action_progressive_widening(self, node):
        """
        Progressive widening for action selection based on visit counts.

        Args:
            node (TreeNode): The current tree node for which actions are being widened.

        Returns:
            Action: The selected action after applying progressive widening.
        """
        action_count = len(node.children)  # Number of actions currently expanded
        widening_threshold = int(self.ka * (node.visit_count ** self.alpha_a))
        new_action = None

        # Add new actions if below the widening threshold
        if action_count <= widening_threshold:
            new_action = self.sample_continuous_action()
            while new_action in node.children:  # Ensure uniqueness
                new_action = self.sample_continuous_action()
            # print(f"  add new action: {new_action}")

        # Select the action with the highest UCB score
        return new_action

    def select_ucb_action(self, node):
        """
        Selects the action with the highest UCB score among the node's children.

        Args:
            node (TreeNode): The current node from which to select the best action based on UCB.

        Returns:
            Action: The action with the highest UCB score.
        """
        best_action = None
        best_ucb_score = float('-inf')
        # print(f" ucb analysis starts for node with action {self.belief.particles[0].data[0][2]}")
        for action, child_node in node.children.items():
            # Calculate UCB score for the child node
            exploitation = child_node.value
            exploration = self.exploration_constant * np.sqrt(np.log(node.visit_count + 1) / (child_node.visit_count + 1))
            ucb_score = exploitation + exploration
            # print(f" node visit count {node.visit_count}, child node visit count {child_node.visit_count}, child node value {child_node.value} ")

            # Update the best action based on the UCB score
            if ucb_score > best_ucb_score:
                best_ucb_score = ucb_score
                best_action = action

        return best_action

    def sample_continuous_action(self):
        """Samples a continuous action value (e.g., acceleration) within a specified range."""
        action_value = random.uniform(-2.0, 2.0)  # Adjust bounds as needed
        return Action(action_value)

    def simulate(self, node, depth):
        # indent = " + " * (4 - depth)
        # print(f'{indent} simulate - Depth: {depth}')
        if depth == 0:
            return 0

        # Choose action based on progressive widening if needed
        new_action = self.action_progressive_widening(node)
        if (new_action != None):
            # Sample the next state and observation
            new_next_states = []
            observations = []
            for particle in self.belief.particles:
                next_state = self.transition_model.sample(particle, new_action)
                new_next_states.append(next_state)

                observation = self.observation_model.sample(next_state, new_action)
                observations.append(observation)
            # print(f"{indent} new action: {new_action}")

            # Update belief based on the action and observation
            new_belief = self.belief.update(new_action, observations, self.observation_model)
            node.children[new_action] = TreeNode(new_belief)  # Add new action node
            self.action = new_action
        else:
            self.action = self.select_ucb_action(node)

        # Calculate immediate reward as an average over all particles
        immediate_rewards = []
        for particle, next_state in zip(self.belief.particles, node.children[self.action].belief.particles):
            reward = self.reward_model.sample(particle, self.action, next_state)
            immediate_rewards.append(reward)
        immediate_reward = np.mean(immediate_rewards)

        self.belief = node.children[self.action].belief

        # print(f"{indent} action: {self.action}")

        # Simulate recursively down the tree
        future_reward = self.simulate(node.children[self.action], depth - 1)

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
    """Represents the belief state with particle-based filtering and weighting."""
    def __init__(self, particles, transition_model):
        self.particles = particles
        self.transition_model = transition_model

    def update(self, action, observations, observation_model):
        new_particles = []
        for particle, observation in zip(self.particles, observations):
            next_particle = self.transition_model.sample(particle, action)
            obs_prob = observation_model.probability(observation, next_particle, action)
            if obs_prob > 0:
                new_particles.append((next_particle, obs_prob))

        if not new_particles:
            new_particles = [(self.transition_model.sample(random.choice(self.particles), action), 1) for _ in range(len(self.particles))]

        weights = np.array([w for _, w in new_particles])
        weights /= weights.sum()

        # Create and return a new Belief instance
        random_particles = random.choices([p for p, _ in new_particles], weights=weights, k=len(self.particles))
        return Belief(random_particles, self.transition_model)
