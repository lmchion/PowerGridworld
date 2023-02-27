"""Script for running single-machine training.  If you want to run rllib on a 
cluster see, e.g., https://docs.ray.io/en/latest/cluster/deploy.html."""
import json
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import Callback
from ray.tune.logger import LoggerCallback
from ray.tune.registry import register_env

from gridworld.log import logger
from gridworld.scenarios.heterogeneous_hs import make_env_config


class HSAgentTrainingCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        episode.media["episode_data"] = defaultdict(list)
        
        # episode.user_data = {"final": {}, "running": defaultdict(list)}

    def on_episode_step(
        self, *, worker, base_env, episode, env_index, **kwargs
    ):
        # TODO change this in subcomponents to use the component name to remove hard-coding.
        episode.media["episode_data"]['ev_step_cost'].append(episode.last_info_for().get('ev_step_cost', None))
        episode.media["episode_data"]['es_step_cost'].append(episode.last_info_for().get('es_step_cost', None))

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for name, value in episode.media["episode_data"].items():
            episode.media["episode_data"][name] = np.array(value).tolist()
        
class HSDataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = osp.join(trial.logdir, "episode_data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        if "episode_data" not in result["episode_media"]:
            return

        step = result['timesteps_total']
        data_file = osp.join(
            self._trial_local_dir[trial], f"data-{step:08d}.json"
        )

        num_episodes = result["episodes_this_iter"]
        data = result["episode_media"]["episode_data"]
        episode_data = data[-num_episodes:]

        json.dump(episode_data, open(data_file, "w"))  
    

def env_creator(config: dict):
    """Simple wrapper that takes a config dict and returns an env instance."""
    
    from gridworld import HSMultiComponentEnv

    return HSMultiComponentEnv(**config)


def main(**args):

    # Log the args used for training.
    logger.info(f"ARGS: {args}")

    # Run ray on a single node.  If running on VPN, you might need some bash 
    # magic, see e.g. train.sh.
    ray.init(_node_ip_address=args["node_ip_address"])
   
    # Register the environment.
    env_name = args["env_name"]
    register_env(env_name, env_creator)

    # Create the env configuration with option to change max episode steps
    # for debugging.
    env_config = make_env_config(
        rescale_spaces=True
    )
    env_config.update({"max_episode_steps": args["max_episode_steps"]})

    print("ENV CONFIG", env_config)

    # Create an env instance to introspect the gym spaces and episode length
    # when setting up the multiagent policy.
    env = env_creator(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    _ = env.reset()

    # Collect params related to train batch size and resources.
    rollout_fragment_length = env.max_episode_steps
    num_workers = args["num_cpus"]

    # Set any stopping conditions.
    stop = {
        'training_iteration': args["stop_iters"],
        'timesteps_total': args["stop_timesteps"],
        'episode_reward_mean': args["stop_reward"]
    }

    # Configure the deep learning framework.
    framework_config = {
        "framework": tune.grid_search([args["framework"]]),  # adds framework to trial name
        "eager_tracing": True  # ~3-4x faster than False
    }

    # Configure policy evaluation.  Evaluation appears to be broken using
    # pytorch, so consider omitting this.
    evaluation_config = {}
    if framework_config["framework"] == "tf2":
        evaluation_config = {
            "evaluation_interval": 1,
            "evaluation_num_episodes": 1,
            "evaluation_config": {"explore": False}
        }

    # Configure hyperparameters of the RL algorithm.  train_batch_size is fixed
    # so that results are reproducible, but 34 CPU workers were used in training 
    # -- expect slower performence if using fewer.
    hyperparam_config = {
        "lr": 1e-3,
        "num_sgd_iter": 10,
        "entropy_coeff": 0.0,
        "train_batch_size": rollout_fragment_length * 34,   # ensure reproducible
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
    }

    # Run the trial.
    experiment = tune.run(
        args["run"],
        local_dir=args["local_dir"],
        checkpoint_freq=1,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=100,
        stop=stop,
        callbacks=[HSDataLoggerCallback()],
        config={
            "env": env_name,
            "env_config": env_config,
            "num_gpus": args["num_gpus"],
            "num_workers": num_workers,
            "callbacks": HSAgentTrainingCallback,
            # "multiagent": {
            #     "policies": {
            #         agent_id: (None, obs_space[agent_id], act_space[agent_id], {}) 
            #             for agent_id in obs_space 
            #     },
            #     "policy_mapping_fn": (lambda agent_id: agent_id)
            # },
            "log_level": args["log_level"].upper(),
            **framework_config,
            **hyperparam_config,
            **evaluation_config
        }
    )

    return experiment


if __name__ == "__main__":

    from args import parser

    args = parser.parse_args()

    _ = main(**vars(args))