"""
This POMDP simulation file is to run POMCPOW solver in gym scenario
"""

import sys
import os
import json
import importlib
import math
import numpy as np
import gymnasium as gym
import ttrl_env
from ttrl_agent.agents.common.factory import load_environment
from simulation import Simulation
from utils import show_videos
from pomdp_core import *
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pomdp_simulation import *
from pomdp_agent import *

# Change the current working directory to scripts/
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

# use environment configuratino file, which is defined in /configs/IntersectionEnv/env.json
env_config = os.path.join(script_path, 'configs', 'IntersectionEnv', 'env.json')

# Load an environment from the configuration file.
env = load_environment(env_config)

# Reset the environment to initialize it
env.reset()

# Extract the road network
road = env.get_wrapper_attr('road')

def visualize_road_network(road, output_file="intersection_map.png", dpi=300):
    """
    Visualizes the road network and saves it as an image file.

    :param road: The road object containing the network.
    :param output_file: The file name to save the visualization.
    :param dpi: Resolution of the output image.
    """
    # Extract the road network's lane dictionary
    lanes_dict = road.network.lanes_dict()

    # Initialize the labels_added dictionary to track if a label has been added for each lane type
    labels_added = {"StraightLane": False, "CircularLane": False}

    # Boundary edge adjustment
    bd_edge = 0.5

    # Iterate through the lanes in lanes_dict
    for (from_, to_, i), lane in lanes_dict.items():
        # print(f"Lane Type = {type(lane)}, From: {from_}, To: {to_}, Index: {i}")

        # Annotate the nodes (from_ and to_) on the plot
        if isinstance(from_, tuple) and isinstance(to_, tuple):  # Ensure from_ and to_ are coordinates
            plt.scatter(*from_, color='red', s=50, label="Node" if i == 0 else None)  # Mark from_
            plt.text(from_[0], from_[1], f"{from_}", color='red', fontsize=8)  # Annotate from_

            plt.scatter(*to_, color='blue', s=50, label="Node" if i == 0 else None)  # Mark to_
            plt.text(to_[0], to_[1], f"{to_}", color='blue', fontsize=8)  # Annotate to_

        if isinstance(lane, ttrl_env.road.lane.StraightLane):
            # Plot StraightLane
            label = "Straight Lane" if not labels_added["StraightLane"] else "_nolegend_"
            # print(f"Straight Lane: Start: {lane.start}, End: {lane.end}")
            plt.plot(
                [lane.start[0], lane.end[0]],
                [lane.start[1], lane.end[1]],
                'k-', linewidth=2, label=label
            )
            labels_added["StraightLane"] = True
            # Plot lane boundaries based on line_types
            for lateral_offset, line_type in zip([-lane.width / 2 + bd_edge, lane.width / 2 - bd_edge], lane.line_types):
                if line_type == ttrl_env.road.lane.LineType.NONE:
                    continue
                boundary_start = lane.position(0, lateral_offset)
                boundary_end = lane.position(lane.length, lateral_offset)
                color = 'red' if line_type == ttrl_env.road.lane.LineType.CONTINUOUS else 'yellow'
                plt.plot(
                    [boundary_start[0], boundary_end[0]],
                    [boundary_start[1], boundary_end[1]],
                    color=color, linestyle='-' if line_type == ttrl_env.road.lane.LineType.CONTINUOUS else '--', linewidth=1
                )

        elif isinstance(lane, ttrl_env.road.lane.CircularLane):
            # Plot CircularLane
            # print(f"Circular Lane: Center: {lane.center}, Radius: {lane.radius}")
            center = lane.center
            radius = lane.radius
            start_angle = lane.start_phase
            end_angle = lane.end_phase
            angles = np.linspace(start_angle, end_angle, num=100)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            label = "Circular Lane" if not labels_added["CircularLane"] else "_nolegend_"
            plt.plot(x, y, 'b-', linewidth=2, label=label)
            labels_added["CircularLane"] = True
            # Plot lane boundaries based on line_types
            for lateral_offset, line_type in zip([-lane.width / 2 + bd_edge, lane.width / 2 - bd_edge], lane.line_types):
                if line_type == ttrl_env.road.lane.LineType.NONE:
                    continue
                boundary_x = center[0] + (radius + lateral_offset) * np.cos(angles)
                boundary_y = center[1] + (radius + lateral_offset) * np.sin(angles)
                color = 'red' if line_type == ttrl_env.road.lane.LineType.CONTINUOUS else 'yellow'
                plt.plot(
                    boundary_x, boundary_y, color=color, linestyle='-' if line_type == ttrl_env.road.lane.LineType.CONTINUOUS else '--', linewidth=1
                )

    # Add legend to the plot
    plt.legend()

    # Customize the plot
    plt.title("Road Network Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.axis('equal')

    # Save the plot to a file
    plt.savefig(output_file, dpi=dpi)
    print(f"Road network visualization saved as '{output_file}'")

# Debugging
# visualize_road_network(road)

# Convert gym map to TopoMap
lanes_dict = road.network.lanes_dict()
map = TopoMap()
start_dict = {}
end_dict = {}
index = 0
for (from_, to_, i), lane in lanes_dict.items():
    if isinstance(lane, ttrl_env.road.lane.StraightLane):
        yaw = math.atan2(lane.end[1] - lane.start[1], lane.end[0] - lane.start[0])
        waypoints = interpolate_line_with_yaw([lane.start[0], lane.start[1], yaw], [lane.end[0], lane.end[1], yaw])
    elif isinstance(lane, ttrl_env.road.lane.CircularLane):
        center = lane.center
        radius = lane.radius
        start_angle = lane.start_phase
        end_angle = lane.end_phase
        x0 = center[0] + radius * math.cos(start_angle)
        y0 = center[1] + radius * math.sin(start_angle)
        yaw0 = start_angle + np.pi / 2 * lane.direction
        x1 = center[0] + radius * math.cos(end_angle)
        y1 = center[1] + radius * math.sin(end_angle)
        yaw1 = end_angle + np.pi / 2 * lane.direction
        waypoints = interpolate_line_with_yaw([x0, y0, yaw0], [x1, y1, yaw1])
    # add waypoint
    map.add_waypoints(index, waypoints)
    if from_ not in start_dict:
        start_dict[from_] = []
    start_dict[from_].append(index)
    if to_ not in end_dict:
        end_dict[to_] = []
    end_dict[to_].append(index)
    # add relation
    if from_ in end_dict:
        for f in end_dict[from_]:
            map.add_connection(f, index)
    if to_ in start_dict:
        for t in start_dict[to_]:
            map.add_connection(index, t)
    if to_ in end_dict:
        for c in end_dict[to_]:
            map.add_confliction(index, c)
    index = index + 1

map.draw_tree()

# Load an agent from the class.
# The agent class must have:
# def reset(self): # reset agent
# def seed(self, seed=None): # init agent
# def act(self, state): # plan action at current state. This is where optimization based method implemented.
# def record(self, state, action, reward, next_state, done, info): # Record a transition by performing a Deep Q-Network iteration
# def save(self, filename): save the agent

agent = POMCPAgent(env, map)

# Run the simulation.
NUM_EPISODES = 100  #@param {type: "integer"}
simulation = POMCPSimulation(env, agent, num_episodes=NUM_EPISODES, display_env=True)
print(f"Ready to run {agent} on {env}")
simulation.run()
test_path = simulation.run_directory / "test"
show_videos(test_path)
