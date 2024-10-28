"""This POMDP comparison class is modified based on the classic Tiger problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

States: [s0,s1,...,sk], math:`S\subseteq[\mathbb{R},\mathbb{R},string]^(k+1)`
Actions: [a0,a1,...,ak], math:`A\subseteq[\mathbb{R}^(k+1)`
Rewards:
    - Crush reward
    - velocity reward
    - acceleration reward
Observations: [o0,o1,...,ok], math:`S\subseteq[\mathbb{R},\mathbb{R},\mathbb{R}]^(k+1)`

Note that in this example, it is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more
complicated examples.)
"""

import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy


# State space
stype = [('field1', 'f8'), ('field2', 'f8'), ('field3', 'U10')]  # 'f8' is for double, 'U10' is for strings up to 10 characters
class State(pomdp_py.State):
    def __init__(self, data):
        # Ensure that `data` is in the correct structured format
        if not isinstance(data, np.ndarray) or data.dtype != np.dtype(stype):
            raise ValueError("Data must be a NumPy structured array with the dtype: stype")
        self.data = data

    def __hash__(self):
        # Convert to a tuple for hash calculation
        return hash(tuple(self.data))

    def __eq__(self, other):
        if isinstance(other, State):
            # Compare structured arrays element-wise
            return np.array_equal(self.data, other.data)
        return False

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"State(data={self.data})"


# Action space
class Action:
    """The action is a vector of velocities."""

    def __init__(self, control):
        """
        Initializes an action with a vector of velocities.

        Args:
            control (array-like): array of velocities as doubles.
        """
        # Convert control to a NumPy array of floats
        self.control = np.array(control, dtype='float64')

    def __hash__(self):
        # Convert to tuple for hashing
        return hash(tuple(self.control))

    def __eq__(self, other):
        if isinstance(other, Action):
            # Use np.array_equal for element-wise comparison
            return np.array_equal(self.control, other.control)
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Action(control={self.control.tolist()})"


# Observation space
otype = [('field1', 'f8'), ('field2', 'f8'), ('field3', 'f8')]  # 'f8' for double
class Observation:
    def __init__(self, data):
        """
        Initializes an observation with a structured array of tuples (double, double, double).

        Args:
            data (array-like): Array of tuples, where each tuple is (double, double, double).
        """
        # Ensure data is a structured NumPy array with the correct dtype
        self.data = np.array(data, dtype=otype)

    def __hash__(self):
        # Convert structured array to a tuple for hashing
        return hash(tuple(map(tuple, self.data)))

    def __eq__(self, other):
        if isinstance(other, Observation):
            # Use np.array_equal for structured array comparison
            return np.array_equal(self.data, other.data)
        return False

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Observation(data={self.data})"


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            # heard the correct growl
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0, 1) < thresh:
            return Observation(next_state.name)
        else:
            return Observation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [Observation(s) for s in {"tiger-left", "tiger-right"}]


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return State(state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        """
        return [State(s) for s in {"tiger-left", "tiger-right"}]


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else:  # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
    small, finite action space"""

    ACTIONS = [Action(s) for s in {"open-left", "open-right", "listen"}]

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class TigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(),
            ObservationModel(obs_noise),
            RewardModel(),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(), RewardModel())
        super().__init__(agent, env, name="TigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right';
                         True state of the environment
            belief (float): Initial belief that the target is
                            on the left; Between 0-1.
            obs_noise (float): Noise for the observation
                               model (default 0.15)
        """
        init_true_state = State(state)
        init_belief = pomdp_py.Histogram(
            {State("tiger-left"): belief, State("tiger-right"): 1.0 - belief}
        )
        tiger_problem = TigerProblem(obs_noise, init_true_state, init_belief)
        tiger_problem.agent.set_belief(init_belief, prior=True)
        return tiger_problem


def test_planner(tiger_problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (TigerProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(tiger_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {tiger_problem.env.state}")
        print(f"Belief: {tiger_problem.agent.cur_belief}")
        print(f"Action: {action}")
        # There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #   reward = tiger_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        reward = tiger_problem.env.reward_model.sample(
            tiger_problem.env.state, action, None
        )
        print("Reward:", reward)

        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = tiger_problem.agent.observation_model.sample(tiger_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that tiger_problem.env.state stores the
        # environment state after action execution.
        real_observation = Observation(tiger_problem.env.state.name)
        print(">> Observation:", real_observation)
        tiger_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                tiger_problem.agent.cur_belief,
                action,
                real_observation,
                tiger_problem.agent.observation_model,
                tiger_problem.agent.transition_model,
            )
            tiger_problem.agent.set_belief(new_belief)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken
            # until every time door is opened.
            print("\n")


def make_tiger(noise=0.15, init_state="tiger-left", init_belief=[0.5, 0.5]):
    """Convenient function to quickly build a tiger domain.
    Useful for testing"""
    tiger = TigerProblem(
        noise,
        State(init_state),
        pomdp_py.Histogram(
            {
                State("tiger-left"): init_belief[0],
                State("tiger-right"): init_belief[1],
            }
        ),
    )
    return tiger


def main():
    init_true_state = random.choice(["tiger-left", "tiger-right"])
    init_belief = pomdp_py.Histogram(
        {State("tiger-left"): 0.5, State("tiger-right"): 0.5}
    )
    tiger = make_tiger(init_state=init_true_state)
    init_belief = tiger.agent.belief

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(tiger, vi, nsteps=3)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(
        max_depth=3,
        discount_factor=0.95,
        num_sims=4096,
        exploration_const=50,
        rollout_policy=tiger.agent.policy_model,
        show_progress=True,
    )
    test_planner(tiger, pouct, nsteps=10)
    TreeDebugger(tiger.agent.tree).pp

    # Reset agent belief
    tiger.agent.set_belief(init_belief, prior=True)
    tiger.agent.tree = None

    print("** Testing POMCP **")
    tiger.agent.set_belief(
        pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True
    )
    pomcp = pomdp_py.POMCP(
        max_depth=3,
        discount_factor=0.95,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=tiger.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    test_planner(tiger, pomcp, nsteps=10)
    TreeDebugger(tiger.agent.tree).pp


if __name__ == "__main__":
    main()
