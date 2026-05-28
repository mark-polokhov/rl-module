import gymnasium as gym
import numpy as np
import grpc
from typing import Dict, Any, Tuple, Optional
import logging

import src.proto.racing_pb2 as pb2
import src.proto.racing_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class RacingEnv(gym.Env):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__()

        print('Environment initialization...')

        env_config = env_config or {}

        self.grpc_address = env_config.get("address", "localhost:50051")
        self.timeout = env_config.get("grpc_timeout", 10.0)

        self.obs_dim = env_config.get("obs_dim", 18)
        self.action_dim = env_config.get("action_dim", 2)

        self.max_episode_steps = env_config.get("max_episode_steps", 1000)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.channel = None
        self.stub = None
        self._connect()

        self.current_step = 0
        self.prev_obs = None  # для reward

    def _connect(self):
        try:
            logger.warning("Creating gRPC connection...")
            self.channel = grpc.insecure_channel(self.grpc_address)
            logger.warning("Waiting for gRPC channel ready...")

            grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            logger.warning("gRPC channel READY")

            self.stub = pb2_grpc.RacingServiceStub(self.channel)
            logger.warning(f"Connected to Engine: {self.grpc_address}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Engine: {e}")

    def reset(self, *, seed=None, options=None):
        try:
            super().reset(seed=seed)

            request = pb2.ResetRequest(seed=int(seed or 42))
            response = self.stub.Reset(request, timeout=self.timeout)

            obs = np.array(response.observation, dtype=np.float32)

            if obs.shape != (self.obs_dim,):
                raise ValueError(f"Invalid obs shape: {obs.shape}")

            self.current_step = 0
            self.prev_obs = obs

            return obs, {}

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return obs, {"error": str(e)}

    def step(self, action: np.ndarray):
        try:
            action = np.clip(action, -1.0, 1.0)

            request = pb2.StepRequest(
                action=action.astype(np.float32).tolist()
            )

            response = self.stub.Step(request, timeout=self.timeout)

            obs = np.array(response.observation, dtype=np.float32)

            if obs.shape != (self.obs_dim,):
                raise ValueError(f"Invalid obs shape: {obs.shape}")

            self.current_step += 1

            reward = self._compute_reward(obs, action)

            terminated = False
            truncated = self.current_step >= self.max_episode_steps

            self.prev_obs = obs

            return obs, reward, terminated, truncated, {}

        except Exception as e:
            logger.error(f"Step failed: {e}")
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return obs, -1.0, True, False, {"error": str(e)}

    def _compute_reward(self, obs, action):

        # spline_dist_to = obs[0]
        # spline_progress = obs[1]
        # spline_angle_to = obs[2]
        # spline_curv = obs[3]

        # ckpt_changed = obs[4]
        # ckpt_wrong = obs[5]
        # ckpt_dist_to = obs[6]
        # ckpt_angle_to = obs[7]
        
        # speed = obs[8]
        # speed_forward = obs[9]
        # speed_lateral = obs[10]

        # walls_collision = obs[11]
        # speed_is_backward = obs[12]

        # throttle = action[0]
        # steering = action[1]

        # spline_proximity = max(1 - abs(spline_dist_to), 0) 

        # reward = (
        #     + 0.1 * max(speed_forward, 0)
        #     + 10.0 * ckpt_changed
        #     # + 5.0 * spline_proximity
        #     + max(10.0 - spline_angle_to, 0)
        #     - 10.0 * walls_collision
        #     - 0.05
        # )

        ## SVATUKHIN TEST

        spline_dist_to = obs[0]
        spline_progress = obs[1]

        speed_forward = obs[6]
        speed_is_backward = obs[8]

        walls_collision = obs[7]

        reward = (
            + 0.1 * max(speed_forward, 0)
            # + 10.0 * ckpt_changed
            + 2.0 * spline_progress
            # + 5.0 * spline_proximity
            - 1.0 * spline_dist_to
            - 10.0 * walls_collision
            - 2.0 * speed_is_backward
            - 0.05
        )

        return float(reward)

    def close(self):
        if self.channel:
            self.channel.close()

    def render(self):
        pass