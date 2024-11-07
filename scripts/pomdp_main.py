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
from pomdp_solver import *

map = simple_no_left_4_way_intersection()
# Initialize components from pomdp_core
transition_model = TransitionModel(map)
observation_model = ObservationModel(map, 0.1)
reward_model = RewardModel(map)
policy_model = PolicyModel()

# Define the initial state
s0 = map.find_length_by_waypoint(9)
s1 = map.find_length_by_waypoint(0)
initial_state = [(s0, 8.0, 0.0, 9), (s1, 8.0, 0.0, 0)]
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
