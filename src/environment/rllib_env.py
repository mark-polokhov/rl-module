import gymnasium as gym
import numpy as np
import grpc
import importlib
import logging

from pathlib import Path
from typing import Dict, Any, Optional

import src.proto.racing_pb2 as pb2
import src.proto.racing_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class RacingEnv(gym.Env):
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__()

        print("Environment initialization...")

        env_config = env_config or {}

        self.grpc_address = env_config.get(
            "address",
            "localhost:50051"
        )

        self.timeout = env_config.get(
            "grpc_timeout",
            10.0
        )

        self.obs_dim = env_config.get(
            "obs_dim",
            18
        )

        self.action_dim = env_config.get(
            "action_dim",
            2
        )

        self.max_episode_steps = env_config.get(
            "max_episode_steps",
            1000
        )

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
        self.prev_obs = None

        self.reward_file = Path(
            "configs/reward_function.py"
        )

        self.reward_file_mtime = None
        self.reward_fn = None

        self._load_reward_function()

    def _connect(self):
        try:
            logger.warning(
                "Creating gRPC connection..."
            )

            self.channel = grpc.insecure_channel(
                self.grpc_address
            )

            logger.warning(
                "Waiting for gRPC channel ready..."
            )

            grpc.channel_ready_future(
                self.channel
            ).result(timeout=self.timeout)

            logger.warning(
                "gRPC channel READY"
            )

            self.stub = pb2_grpc.RacingServiceStub(
                self.channel
            )

            logger.warning(
                f"Connected to Engine: {self.grpc_address}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Engine: {e}"
            )

    def _load_reward_function(self):
        try:
            module = importlib.import_module(
                "configs.reward_function"
            )

            importlib.reload(module)

            if not hasattr(module, "reward_fn"):
                raise AttributeError(
                    "reward_fn function not found"
                )

            self.reward_fn = module.reward_fn

            if self.reward_file.exists():
                self.reward_file_mtime = (
                    self.reward_file.stat().st_mtime
                )

            logger.info(
                "Reward function loaded successfully"
            )

        except Exception as e:
            logger.exception(
                "Cannot load reward function"
            )

            raise RuntimeError(
                f"Reward loading failed: {e}"
            )

    def _reload_reward_if_needed(self):
        try:
            if not self.reward_file.exists():
                return

            current_mtime = (
                self.reward_file.stat().st_mtime
            )

            if (
                self.reward_file_mtime is None
                or current_mtime != self.reward_file_mtime
            ):
                logger.warning(
                    "Reward file changed. Reloading..."
                )

                self._load_reward_function()

        except Exception:
            logger.exception(
                "Reward hot reload failed"
            )

    def reset(self, *, seed=None, options=None):
        try:
            super().reset(seed=seed)

            request = pb2.ResetRequest(
                seed=int(seed or 42)
            )

            response = self.stub.Reset(
                request,
                timeout=self.timeout
            )

            obs = np.array(
                response.observation,
                dtype=np.float32
            )

            if obs.shape != (self.obs_dim,):
                raise ValueError(
                    f"Invalid obs shape: {obs.shape}"
                )

            self.current_step = 0
            self.prev_obs = obs

            return obs, {}

        except Exception as e:
            logger.error(
                f"Reset failed: {e}"
            )

            obs = np.zeros(
                self.obs_dim,
                dtype=np.float32
            )

            return obs, {
                "error": str(e)
            }

    def step(self, action: np.ndarray):
        self._reload_reward_if_needed()

        try:
            action = np.clip(
                action,
                -1.0,
                1.0
            )

            request = pb2.StepRequest(
                action=action.astype(
                    np.float32
                ).tolist()
            )

            response = self.stub.Step(
                request,
                timeout=self.timeout
            )

            obs = np.array(
                response.observation,
                dtype=np.float32
            )

            if obs.shape != (self.obs_dim,):
                raise ValueError(
                    f"Invalid obs shape: {obs.shape}"
                )

            self.current_step += 1

            reward = self._compute_reward(
                obs,
                action
            )

            terminated = False

            truncated = (
                self.current_step
                >= self.max_episode_steps
            )

            self.prev_obs = obs

            return (
                obs,
                reward,
                terminated,
                truncated,
                {}
            )

        except Exception as e:
            logger.error(
                f"Step failed: {e}"
            )

            obs = np.zeros(
                self.obs_dim,
                dtype=np.float32
            )

            return (
                obs,
                -1.0,
                True,
                False,
                {"error": str(e)}
            )

    def _compute_reward(self, obs, action):
        try:
            return float(
                self.reward_fn(
                    self,
                    obs,
                    action
                )
            )

        except Exception as e:
            logger.exception(
                "Reward calculation failed"
            )

            return -1.0

    def close(self):
        if self.channel:
            self.channel.close()

    def render(self):
        pass