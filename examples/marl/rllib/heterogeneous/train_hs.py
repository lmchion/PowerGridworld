"""Script for running single-machine training.  If you want to run rllib on a 
cluster see, e.g., https://docs.ray.io/en/latest/cluster/deploy.html."""
import json
import os
import os.path as osp
import threading

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import LoggerCallback
from ray.tune.registry import register_env

from gridworld.log import logger
from gridworld.scenarios.heterogeneous_hs import make_env_config


class HSAgentTrainingCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        #episode.user_data["episode_data"] = defaultdict(list)
        episode.media["episode_data"] = []

    def on_episode_step(
        self, *, worker, base_env, episode, env_index, **kwargs
    ):
        # TODO change this in subcomponents to use the component name to remove hard-coding.
        ep_lastinfo = episode.last_info_for()
        step_meta = ep_lastinfo.get('step_meta', None)
        grid_cost = ep_lastinfo.get('grid_cost', None)
        es_cost = ep_lastinfo.get('es_cost', None)
        hvac_power = ep_lastinfo.get('hvac_power', None)
        other_power = ep_lastinfo.get('other_power', None)
        # step_meta = episode.last_info_for().get('step_meta', None)
        # step_meta = episode.last_info_for().get('step_meta', None)
        for step_meta_item in step_meta:
            episode.media["episode_data"].append([step_meta_item["device_id"], 
                                                  step_meta_item["timestamp"], 
                                                  step_meta_item["cost"], 
                                                  step_meta_item["reward"],
                                                  step_meta_item["action"], 
                                                  step_meta_item["pv_power"], 
                                                  step_meta_item["es_power"], 
                                                  step_meta_item["grid_power"],
                                                  grid_cost,
                                                  es_cost,
                                                  hvac_power,
                                                  other_power,
                                                  step_meta_item["device_custom_info"]])
        #episode.media["episode_data"]['es_step_cost'].append(episode.last_info_for().get('es_step_cost', None))

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        episode_data = episode.media["episode_data"]

class HSDataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = osp.join(trial.logdir, "episode_data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        file_lock = threading.Lock()

        with file_lock:
            episode_media = result["episode_media"]
            if "episode_data" not in episode_media:
                return

            step = result['timesteps_total']
            dump_file_name = osp.join(
                self._trial_local_dir[trial], f"data-{step:08d}.json"
            )

            num_episodes = result["episodes_this_iter"]
            data = episode_media["episode_data"]            

            episode_data = data[-num_episodes:]

            extract_columns = ["device", 
                               "timestamp", 
                               "cost", 
                               "reward",
                               "action", 
                               "pv_power", 
                               "es_power", 
                               "grid_power",
                               "grid_cost",
                                "es_cost",
                                "hvac_power",
                                "other_power",
                                "device_custom_info"]
            #logger.info("Trial Result dumping to", dump_file_name)
            df = pd.DataFrame(np.array([]).reshape((-1, len(extract_columns))), columns = extract_columns)
            for tranche in episode_data:
                if not tranche:
                    logger.info("Episode data tranche is empty while logging. skipping.")
                    continue
                
                tmp_df = pd.DataFrame(tranche, columns=extract_columns)

                df = df.append(tmp_df)

            device_list = df['device'].unique()
            final_json = []
            for device in device_list:
                device_data = {}
                device_data['device_id'] = device
                tmp_device_data = df[df["device"]==device].drop("device", axis=1)
                device_data['columns'] = list(tmp_device_data.columns.values)
                device_data['usage_data'] = tmp_device_data.values.tolist()
                final_json.append(device_data)
            
            with open(dump_file_name, mode='w+') as thisfile:
                json.dump(final_json, thisfile)  

                

def env_creator(config: dict):
    """Simple wrapper that takes a config dict and returns an env instance."""
    
    from gridworld import HSMultiComponentEnv

    return HSMultiComponentEnv(**config)


def main(**args):

    # Log the args used for training.
    logger.info(f"ARGS: {args}")

    # Run ray on a single node.  If running on VPN, you might need some bash 
    # magic, see e.g. train.sh.
    ray.init(_node_ip_address=args["node_ip_address"], log_to_driver=True, logging_level="error")
   
    # Register the environment.
    env_name = args["env_name"]
    register_env(env_name, env_creator)

    # Create the env configuration with option to change max episode steps
    # for debugging.
    env_config = make_env_config()
    env_config.update({"max_episode_steps": args["max_episode_steps"]})

    logger.info("ENV CONFIG", env_config)

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
        },
        verbose=0
    )

    return experiment


if __name__ == "__main__":

    from args import parser

    args = parser.parse_args()

    _ = main(**vars(args))