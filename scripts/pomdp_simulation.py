# This is the self defined simulation class

import datetime
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, capped_cubic_video_schedule

import ttrl_agent.trainer.logger
from ttrl_agent.agents.common.graphics import AgentGraphics
from ttrl_agent.configuration import serialize
from ttrl_agent.trainer.graphics import RewardViewer

logger = logging.getLogger(__name__)


class POMCPSimulation(object):
    """
        The simulation of an agent interacting with an environment using POMCPOW.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.{}.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 run_directory=None,
                 num_episodes=1000,
                 sim_seed=None,
                 display_env=True,
                 display_rewards=True,
                 close_env=True,
                 step_callback_fn=None):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param AbstractAgent agent: The agent solving the environment
        :param Path directory: Workspace directory path
        :param Path run_directory: Run directory path
        :param int num_episodes: Number of episodes run
        :param sim_seed: The seed used for the environment/agent randomness source
        :param display_env: Render the environment, and have a monitor recording its videos
        :param display_agent: Add the agent graphics to the environment viewer, if supported
        :param display_rewards: Display the performances of the agent through the episodes
        :param close_env: Should the environment be closed when the evaluation is closed
        :param step_callback_fn: A callback function called after every environment step. It takes the following
               arguments: (episode, env, agent, transition, writer).

        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.sim_seed = sim_seed if sim_seed is not None else np.random.randint(0, 1e6)
        self.close_env = close_env
        self.display_env = display_env
        self.step_callback_fn = step_callback_fn

        self.directory = Path(directory or self.default_directory)
        self.run_directory = self.directory / (run_directory or self.default_run_directory)
        self.wrapped_env = RecordVideo(env,
                                       self.run_directory,
                                       episode_trigger=(None if self.display_env else lambda e: False))
        try:
            self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
        except AttributeError:
            pass
        self.wrapped_env = RecordEpisodeStatistics(self.wrapped_env)
        self.episode = 0

        self.reward_viewer = None
        if display_rewards:
            self.reward_viewer = RewardViewer()
        self.observation = None

    def run(self):
        self.run_episodes()
        self.close()

    def run_episodes(self):
        for self.episode in range(self.num_episodes):
            # Run episode
            terminal = False
            self.reset(seed=self.episode)
            rewards = []
            start_time = time.time()
            while not terminal:
                # Step until a terminal step is reached
                reward, terminal = self.step()
                rewards.append(reward)

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass

            # End of episode
            duration = time.time() - start_time

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence
        actions = self.agent.plan(self.observation)
        if not actions:
            raise Exception("The agent did not plan any action")

        # Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        transition = self.wrapped_env.step(action)
        self.observation, reward, done, truncated, info = transition
        terminal = done or truncated

        # Call callback
        if self.step_callback_fn is not None:
            self.step_callback_fn(self.episode, self.wrapped_env, self.agent, transition, self.writer)

        # Record the experience.
        try:
            self.agent.record(previous_observation, action, reward, self.observation, done, info)
        except NotImplementedError:
            pass

        return reward, terminal

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    def reset(self, seed=0):
        seed = self.sim_seed + seed if self.sim_seed is not None else None
        self.observation, info = self.wrapped_env.reset()
        self.agent.seed(seed)  # Seed the agent with the main environment seed
        self.agent.reset()

    def close(self):
        """
            Close the evaluation.
        """
        self.wrapped_env.close()
        if self.close_env:
            self.env.close()
