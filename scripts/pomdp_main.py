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

map = simple_no_left_4_way_intersection()
# Initialize components from pomdp_core
transition_model = TransitionModel(map)
observation_model = ObservationModel(map, 0.1)
reward_model = RewardModel(map)
policy_model = PolicyModel()

# Define the initial state
s0 = map.find_length_by_waypoint(9)
s1 = map.find_length_by_waypoint(0)
initial_state = State([(s0, 8.0, 0.0, 9), (s1, 8.0, 0.0, 0)])
num_particles = 100  # Number of particles in the belief

# Initialize belief with particles around the initial state
belief = Belief([initial_state] * num_particles, transition_model=transition_model)

# Initialize POMCPOW Solver
solver = POMCPOWSolver(
    belief=belief,
    transition_model=transition_model,
    observation_model=observation_model,
    reward_model=reward_model,
    policy_model=policy_model,
    max_depth=3,
    num_sims=20
)

# Run a test plan loop
# TODO: integrate with gym and see if the computation is too heavy then use discreted action space
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
