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

# Solver class: POMCPOWSolver
class POMCPOWSolver:
    def __init__(self, belief, transition_model, observation_model, reward_model, max_depth=3, num_sims=100):
        self.belief = belief
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.max_depth = max_depth
        self.num_sims = num_sims

    def plan(self):
        best_action = None
        best_value = float('-inf')
        actions = self.progressive_widening()  # Actions sampled with progressive widening

        for action in actions:
            value = self.simulate(self.belief, action, depth=self.max_depth)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def progressive_widening(self):
        """Progressively selects a subset of actions to sample in continuous space."""
        num_actions = int(np.log(self.num_sims) + 1)  # Adjust based on exploration needs
        actions = [self.sample_continuous_action() for _ in range(num_actions)]
        return actions

    def sample_continuous_action(self):
        """Samples a continuous action value (acceleration) from a range, e.g., [-2, 2]."""
        action_value = random.uniform(-2.0, 2.0)  # Adjust bounds as needed
        return Action(action_value)

    def simulate(self, belief, action, depth):
        if depth == 0:
            return 0

        # Sample state from belief
        state = random.choice(belief.particles)

        # Sample next state and observation
        next_state = self.transition_model.sample(state, action)
        observation = self.observation_model.sample(next_state, action)

        # Update belief with new action and observation
        belief.update(action, observation, self.observation_model)

        # Calculate immediate reward
        immediate_reward = self.reward_model.sample(state, action, next_state)

        # Recursive simulation to estimate future rewards
        next_action = self.plan()  # Recurse or use rollout policy for quicker evaluation
        future_reward = self.simulate(belief, next_action, depth - 1)

        return immediate_reward + 0.95 * future_reward


# Belief class with particle-based representation
class Belief:
    def __init__(self, num_particles, initial_state, transition_model):
        self.particles = [initial_state] * num_particles
        self.transition_model = transition_model

    def update(self, action, observation, observation_model):
        new_particles = []
        for particle in self.particles:
            # Sample a next state from the transition model and weight by observation likelihood
            next_particle = self.transition_model.sample(particle, action)
            obs_prob = observation_model.probability(observation, next_particle, action)

            if obs_prob > 0:
                new_particles.append(next_particle)

        # Handle particle deprivation by resampling if necessary
        if not new_particles:
            new_particles = [self.transition_model.sample(random.choice(self.particles), action) for _ in range(len(self.particles))]

        self.particles = new_particles


# Main function to test the solver
def main():
    # Initialize components from pomdp_core
    transition_model = TransitionModel()
    observation_model = ObservationModel()
    reward_model = RewardModel()
    policy_model = PolicyModel()  # Not directly used here but part of the POMDP structure

    # Define the initial state
    initial_state = State([(0.0, 8.0, 0.0, 1)])  # Example values: (position=0, velocity=8, acceleration=0, lane_id=1)
    num_particles = 100  # Number of particles in the belief

    # Initialize belief with particles around the initial state
    belief = Belief(num_particles=num_particles, initial_state=initial_state, transition_model=transition_model)

    # Initialize POMCPOW Solver
    solver = POMCPOWSolver(
        belief=belief,
        transition_model=transition_model,
        observation_model=observation_model,
        reward_model=reward_model,
        max_depth=3,
        num_sims=100
    )

    # Run a test plan loop
    for step in range(5):  # Run for a few steps to observe behavior
        print(f"Step {step}")
        action = solver.plan()
        print(f"Chosen Action: {action.data}")

        # Simulate the environment response (observation) for this example
        next_state = transition_model.sample(initial_state, action)
        observation = observation_model.sample(next_state, action)

        print(f"Observation: {observation.data}")

        # Update initial state for next loop iteration
        initial_state = next_state

        print(f"Updated State: {initial_state.data}")
        print("------")

if __name__ == "__main__":
    main()
