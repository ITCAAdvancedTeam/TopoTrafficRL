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
from topo_simulation import *
from topo_agent import *

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
agent = TopoAgent(env, map, num_particles=10, max_depth=5, num_sims=20)

# Run the simulation.
NUM_EPISODES = 100  #@param {type: "integer"}
simulation = TopoSimulation(env, agent, num_episodes=NUM_EPISODES, display_env=True)
print(f"Ready to run {agent} on {env}")
simulation.run()
test_path = simulation.run_directory / "test"
show_videos(test_path)
