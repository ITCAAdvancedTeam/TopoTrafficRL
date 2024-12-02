"""
This POMDP main file is to debug the solver with a simple 4 way intersection and one traffic agent
"""

import numpy as np
from pomdp_core import *
from pomdp_solver import *
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

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

    wdict = {
        "in1": 0,
        "in2": 3,
        "in3": 6,
        "in4": 9,
        "in5": 1,
        "in6": 4,
        "in7": 7,
        "in8": 10,
        "mid1": 12,
        "mid2": 13,
        "mid3": 14,
        "mid4": 15,
        "out1": 16,
        "out2": 17,
        "out3": 18,
        "out4": 19,
        "out5": 20,
        "out6": 21,
        "out7": 22,
        "out8": 23,
        "turn1": 2,
        "turn2": 5,
        "turn3": 8,
        "turn4": 11
    }

    # Add waypoints
    map.add_waypoints(wdict["in1"], in1)
    map.add_waypoints(wdict["in5"], in5)
    map.add_waypoints(wdict["turn1"], turn1)
    map.add_waypoints(wdict["in2"], in2)
    map.add_waypoints(wdict["in6"], in6)
    map.add_waypoints(wdict["turn2"], turn2)
    map.add_waypoints(wdict["in3"], in3)
    map.add_waypoints(wdict["in7"], in7)
    map.add_waypoints(wdict["turn3"], turn3)
    map.add_waypoints(wdict["in4"], in4)
    map.add_waypoints(wdict["in8"], in8)
    map.add_waypoints(wdict["turn4"], turn4)
    map.add_waypoints(wdict["mid1"], mid1)
    map.add_waypoints(wdict["mid2"], mid2)
    map.add_waypoints(wdict["mid3"], mid3)
    map.add_waypoints(wdict["mid4"], mid4)
    map.add_waypoints(wdict["out1"], out1)
    map.add_waypoints(wdict["out2"], out2)
    map.add_waypoints(wdict["out3"], out3)
    map.add_waypoints(wdict["out4"], out4)
    map.add_waypoints(wdict["out5"], out5)
    map.add_waypoints(wdict["out6"], out6)
    map.add_waypoints(wdict["out7"], out7)
    map.add_waypoints(wdict["out8"], out8)

    map.add_connection(wdict["in1"], wdict["in5"])
    map.add_connection(wdict["in1"], wdict["turn1"])
    map.add_connection(wdict["in2"], wdict["in6"])
    map.add_connection(wdict["in2"], wdict["turn2"])
    map.add_connection(wdict["in3"], wdict["in7"])
    map.add_connection(wdict["in3"], wdict["turn3"])
    map.add_connection(wdict["in4"], wdict["in8"])
    map.add_connection(wdict["in4"], wdict["turn4"])
    map.add_connection(wdict["in5"], wdict["mid1"])
    map.add_connection(wdict["in6"], wdict["mid2"])
    map.add_connection(wdict["in7"], wdict["mid3"])
    map.add_connection(wdict["in8"], wdict["mid4"])
    map.add_connection(wdict["mid3"], wdict["out1"])
    map.add_connection(wdict["mid4"], wdict["out2"])
    map.add_connection(wdict["mid1"], wdict["out3"])
    map.add_connection(wdict["mid2"], wdict["out4"])
    map.add_connection(wdict["turn4"], wdict["out5"])
    map.add_connection(wdict["out1"], wdict["out5"])
    map.add_connection(wdict["turn1"], wdict["out6"])
    map.add_connection(wdict["out2"], wdict["out6"])
    map.add_connection(wdict["turn2"], wdict["out7"])
    map.add_connection(wdict["out3"], wdict["out7"])
    map.add_connection(wdict["turn3"], wdict["out8"])
    map.add_connection(wdict["out4"], wdict["out8"])

    # Add conflicts
    map.add_confliction(wdict["in5"], wdict["mid4"])
    map.add_confliction(wdict["in6"], wdict["mid1"])
    map.add_confliction(wdict["in7"], wdict["mid2"])
    map.add_confliction(wdict["in8"], wdict["mid3"])
    map.add_confliction(wdict["out1"], wdict["turn4"])
    map.add_confliction(wdict["out2"], wdict["turn1"])
    map.add_confliction(wdict["out3"], wdict["turn2"])
    map.add_confliction(wdict["out4"], wdict["turn3"])

    return map

def visualize_step(map, state, action, observation, step):
    """
    Visualize the map, current state, action, and observation.

    Parameters:
        map (TopoMap): The intersection map.
        state (State): The current state of the system.
        action (Action): The action taken.
        observation (Observation): The observation received.
        step (int): The current step number.
    """
    plt.figure(figsize=(10, 10))
    plt.title(f"Step {step}")

    # Plot the map waypoints
    for waypoint_id, waypoints in map.waypoints.items():
        x = [p[0] for p in waypoints]
        y = [p[1] for p in waypoints]
        plt.plot(x, y)

    # Plot the vehicles in the current state
    for vehicle_idx, (s, v, a, waypoint_id) in enumerate(state.data):
        waypoint = map.find_waypoint_by_length(waypoint_id, s)
        if waypoint is not None:
            x, y, yaw = waypoint
            plt.scatter(x, y, color="red", label=f"Vehicle {vehicle_idx} (State)")
            plt.arrow(
                x, y, 0.5 * np.cos(yaw), 0.5 * np.sin(yaw),
                head_width=0.3, head_length=0.5, fc="red", ec="red"
            )

    # Plot the vehicles in the observation
    for obs_idx, (x, y, vx, vy) in enumerate(observation.data):
        plt.scatter(x, y, color="blue", alpha=0.5, label=f"Vehicle {obs_idx} (Obs)")
        plt.arrow(
            x, y, 0.5 * vx, 0.5 * vy,
            head_width=0.3, head_length=0.5, fc="blue", ec="blue"
        )

    # Annotate the action
    plt.text(0.05, 0.95, f"Action: {action.data:.2f} m/s²", transform=plt.gca().transAxes, fontsize=12, color="green")

    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    plt.grid()
    plt.show()


# Main Logic
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
num_particles = 10  # Number of particles in the belief

# Initialize belief with particles around the initial state
belief = Belief([initial_state] * num_particles, transition_model=transition_model)

# Initialize POMCPOW Solver
solver = POMCPOWSolver(
    belief=belief,
    transition_model=transition_model,
    observation_model=observation_model,
    reward_model=reward_model,
    policy_model=policy_model,
    max_depth=5,
    num_sims=20
)

# Create lists to store step-wise data for visualization
steps = []
states = []
actions = []
observations = []

# Run a test plan loop
# TODO: integrate with gym with fresh initialization each iteration
# and see if the computation is too heavy then use discreted action space
for step in tqdm(range(20), desc="Simulating steps"):  # Progress bar for 30 steps
    action = solver.plan()
    next_state = transition_model.sample(initial_state, action)
    observation = observation_model.sample(next_state, action)
    belief = belief.update(action, [observation] * num_particles, observation_model)

    # Store step data for later visualization
    steps.append(step)
    states.append(next_state)
    actions.append(action)
    observations.append(observation)

    # Update initial state and solver for next iteration
    initial_state = next_state
    solver.reset(belief=belief)

# Visualization after the loop
for step, state, action, observation in zip(steps, states, actions, observations):
    print(f"Step {step} Observation: {observation.data}")
    print(f"Chosen Action: {action.data}")
    print(f"Updated State: {state.data}")
    print("------")
    visualize_step(map, state, action, observation, step)
