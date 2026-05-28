from src.environment.rllib_env import RacingEnv

import ray
from ray.tune.registry import register_env

import importlib
from pathlib import Path

import gymnasium as gym
from numpy import isnan
import torch


def env_creator(env_config):
    env = RacingEnv(env_config)

    return gym.wrappers.TimeLimit(
        env,
        max_episode_steps=env_config.get("max_episode_steps", 10000)
    )

class Trainer:
    def __init__(self, config):
        self.config = config
        self.env_name = config["env"]["name"]

        ray.init(ignore_reinit_error=True, log_to_driver=True)

        print('Training initialization...')

        register_env(self.env_name, env_creator)

        self.checkpoint_dir = Path("models/checkpoints").resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_algo_config(self):
        algo_name = self.config["agent"]["algorithm"].upper()

        module_path = f"ray.rllib.algorithms.{algo_name.lower()}"
        module = importlib.import_module(module_path)

        class_name = f"{algo_name}Config"
        ConfigClass = getattr(module, class_name)

        base_config = ConfigClass()

        num_gpus = 1 if self.config['training']['device'] == 'cuda' else 0

        if num_gpus > 0:
            print("Using CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

        cfg = (
            base_config
            .environment(
                env=self.env_name,
                env_config={
                    "address": self.config["env"]["address"],
                    "obs_dim": self.config['env']['obs_dim'],
                    "action_dim": self.config['env']['action_dim'],
                    "max_episode_steps": self.config['training']['max_episode_steps'],
                    # "grpc_timeout": 2.0,
                }
            )
            .framework("torch")
            .resources(num_gpus=num_gpus)
            .env_runners(
                num_env_runners=self.config['training']['num_workers'],
                rollout_fragment_length=self.config['training']['rollout_fragment_length'],
            )
            .training(
                train_batch_size=self.config['training']['train_batch_size'],
                minibatch_size=self.config['training']['minibatch_size'],
                # lr=self.config['training']['lr'],
            )
        )

        return cfg
        
    def _filter_none(d):
        if d is None:
            return {}
        return {k: v for k, v in d.items() if v is not None}


    def train(self):
        config = self._get_algo_config()
        agent = config.build_algo()

        print('Training initialization complete.')

        for i in range(self.config['training']['iterations']):
            result = agent.train()

            runners_stats = result.get("env_runners", {})
            reward = runners_stats.get("episode_return_mean")
            steps_total = result.get("num_env_steps_sampled_lifetime", 0)

            if (i + 1) % self.config['training']['log_freq'] == 0:
                reward_str = f"{reward:.2f}" if (reward is not None and not isnan(reward)) else "N/A"
                print(f"Iter {i+1}/{self.config['training']['iterations']} | Steps: {steps_total} | Reward: {reward_str}", flush=True)

            if (i + 1) % self.config["training"]["checkpoint_freq"] == 0:
                try:
                    agent.save(str(self.checkpoint_dir))
                    # print('Checkpoint export 1 - Success')
                    # agent.export_policy_model("models/onnx", onnx=12)
                    # print('Checkpoint export 2 - Success')
                except Exception as e:
                    print('Checkpoint export - Failure')
                    pass

        ray.shutdown()