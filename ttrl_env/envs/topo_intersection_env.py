from __future__ import annotations

import numpy as np

from ttrl_env import utils
from ttrl_env.envs.common.abstract import AbstractEnv
from ttrl_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from ttrl_env.road.regulation import RegulatedRoad
from ttrl_env.road.road import RoadNetwork
from ttrl_env.vehicle.kinematics import Vehicle


class TopoIntersectionEnv(AbstractEnv):
    ACTIONS: dict[int, str] = {0: "STOP", 1: "SLOWER", 2: "IDLE", 3: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 16,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.0, 9.0, 14],
                },
                "duration": 12,  # [s]
                "destination": "e1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 2,
                "reward_speed_range": [8.0, 11.0],
                "normalize_reward": False,
                "offroad_terminal": False,
                "show_trajectories": True,
                "speed_limit": 10
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 2 for vertical straight lanes and right-turns
            - 1 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left, o:out] | e:exit) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + 2 * lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 80  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 2
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array([0.0, access_length + outer_distance])
            end = rotation @ np.array([0.0, outer_distance])
            net.add_lane(
                "ol" + str(corner),
                "il" + str(corner),
                StraightLane(
                    start, end, line_types=[s, s], priority=priority, speed_limit=self.config["speed_limit"]
                ),
            )
            start = rotation @ np.array([-lane_width, access_length + outer_distance])
            end = rotation @ np.array([-lane_width, outer_distance])
            net.add_lane(
                "or" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[c, s], priority=priority, speed_limit=self.config["speed_limit"]
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([-outer_distance - lane_width / 2, outer_distance + lane_width / 2]))
            net.add_lane(
                "ir" + str(corner),
                "io" + str((corner + 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[s, n],
                    priority=priority,
                    speed_limit=self.config["speed_limit"],
                ),
            )
            # Left turn
            l_center = rotation @ (np.array([left_turn_radius, left_turn_radius - lane_width]))
            net.add_lane(
                "il" + str(corner),
                "io" + str((corner + 3) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    clockwise=True,
                    line_types=[n, s],
                    priority=priority - 2,
                    speed_limit=self.config["speed_limit"],
                ),
            )
            # Straight
            start = rotation @ np.array([-lane_width, outer_distance])
            end = rotation @ np.array([-lane_width, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "io" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=self.config["speed_limit"]
                ),
            )
            # Exit
            start = rotation @ np.array([lane_width, outer_distance])
            end = rotation @ np.array([lane_width, access_length + outer_distance])
            net.add_lane(
                "io" + str(corner),
                "e" + str(corner),
                StraightLane(
                    start, end, line_types=[c, c], priority=priority, speed_limit=10
                ),
            )
        # print(f"lane network: {net.graph}")

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 60, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            longitudinal=40,
            spawn_probability=1,
            must_straight=True,
            position_deviation=0.1,
            speed_deviation=0.1,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            destination_id = self.np_random.integers(2, 4)
            destination = self.config["destination"] or "e" + str(destination_id)
            if (destination_id - ego_id) % 4 == 1:
                ego_lane = self.road.network.get_lane(
                    (f"ol{ego_id % 4}", f"il{ego_id % 4}", 0)
                )
            else:
                ego_lane = self.road.network.get_lane(
                    (f"or{ego_id % 4}", f"ir{ego_id % 4}", 0)
                )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(50 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        must_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return
        if must_straight:
            which_lane = "r"
        else:
            is_straight = self.np_random.choice(range(2), size=1)[0]
            which_lane = "r" if is_straight == 1 else "l"
        route_start = self.np_random.choice(range(4), size=1)[0]
        if must_straight:
            route_end = (route_start + 2) % 4
        elif not is_straight:
            route_end = (route_start - 1) % 4
        else:
            route_end = (route_start + self.np_random.choice(range(1,3), size=1)[0]) % 4
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + which_lane + str(route_start), "i" + which_lane + str(route_start), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=10 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("e" + str(route_end))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "io" in vehicle.lane_index[0]
            and "e" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "io" in vehicle.lane_index[0]
            and "e" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )
