"""Script for running single-machine training.  If you want to run rllib on a 
cluster see, e.g., https://docs.ray.io/en/latest/cluster/deploy.html."""
import csv
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import ray
import requests
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import LoggerCallback
from ray.tune.registry import register_env

from gridworld.log import logger
from gridworld.scenarios.heterogeneous_hs import make_env_config


class HSAgentTrainingCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._total_episode_cost = 0.0

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        #episode.user_data["episode_data"] = defaultdict(list)
        episode.media["episode_data"] = []
        self._total_episode_cost = 0.0
        self._total_datapoints = 0
        

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        ep_lastinfo = episode.last_info_for()
        step_meta = ep_lastinfo.get('step_meta', None)
        grid_cost = ep_lastinfo.get('grid_cost', None)
        es_cost = ep_lastinfo.get('es_cost', None)
        hvac_power = ep_lastinfo.get('hvac_power', None)
        other_power = ep_lastinfo.get('other_power', None)
        total_cost = 0

        for step_meta_item in step_meta:
            episode.media["episode_data"].append([step_meta_item["device_id"], 
                                                  step_meta_item["timestamp"], 
                                                  step_meta_item["cost"], 
                                                  step_meta_item["reward"],
                                                  step_meta_item["action"], 
                                                  step_meta_item["solar_power_consumed"], 
                                                  step_meta_item["es_power_consumed"], 
                                                  step_meta_item["grid_power_consumed"],
                                                  grid_cost,
                                                  es_cost,
                                                  hvac_power,
                                                  other_power,
                                                  step_meta_item["device_custom_info"]])
            self._total_episode_cost += step_meta_item["cost"]
            self._total_datapoints += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        episode.custom_metrics["total_cost"] = self._total_episode_cost


class HSDataLoggerCallback(LoggerCallback):
    def __init__(self, scenario_id, is_push_data_inline):
        super().__init__()

        self._trial_continue = {}
        self._trial_local_dir = {}
        self._scenario_id = scenario_id
        self._is_push_data_inline = is_push_data_inline

    def log_trial_start(self, trial):
        trial.init_logdir()
        # self._trial_local_dir[trial] = osp.join(trial.logdir, "episode_data")
        # os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):

        episode_media = result["episode_media"]
        if "episode_data" not in episode_media:
            return

        junk1 = osp.join(trial.logdir, "progress.csv")
        junk2 = osp.join(trial.logdir, "result.json")
        if os.path.exists(junk1):
            os.remove(junk1)
        if os.path.exists(junk2):
            os.remove(junk2)
 

    def _push_data(self, logdir, csvname):
        csv_file_name = osp.join(logdir, csvname+".csv")
        json_file_name = osp.join(logdir, csvname+".json")

        with open(csv_file_name, encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)
                
            # Convert each row into a dictionary
            # and add it to file

            file = [{k:v for k,v in rows.items() if k!=''} for rows in csvReader]
            
            output = {'result':file}
            
            with open(json_file_name, 'w', encoding='utf-8') as jsonf:
                jsonf.write(json.dumps(output, indent=4))
            
            import requests

            # defining the api-endpoint 
            API_ENDPOINT = "http://44.214.125.207:443/result"
            headers = {"Content-Type": "application/json; charset=utf-8"}
            
            # sending post request and saving response as response object
            r= requests.post(url=API_ENDPOINT, headers=headers, json=output)
            
            # extracting response text
            response_status = r.status_code 
            response_content = r.json()
            logger.info("data push to store; status: "+str(response_status))
            logger.info("data push to store; response: "+str(response_content))

    def on_experiment_end(self, trials, **info):
        print("on_experiment_end dumping the last result for validation..")
        result = trials[0].last_result
        logdir = trials[0].logdir
        episode_media = result["episode_media"]
        if "episode_data" not in episode_media:
            return

        data = episode_media["episode_data"]            

        episode_data = data[-1]

        extract_columns = ["device", 
                            "timestamp", 
                            "cost", 
                            "reward",
                            "action", 
                            "solar_power_consumed", 
                            "es_power_consumed", 
                            "grid_power_consumed",
                            "grid_cost",
                            "es_cost",
                            "hvac_power",
                            "other_power",
                            "device_custom_info"]

        if not episode_data:
            logger.info("Episode data tranche is empty while logging. skipping.")
        
        df = pd.DataFrame(episode_data, columns=extract_columns)

        timestamps = df['timestamp'].unique()
        final_csv_rows = []
        for t_stp in timestamps:
            timestamp_data = {}
            timestamp_data['timestamp'] = t_stp

            tmp_timestamp_data = df[df["timestamp"]==t_stp].drop("timestamp", axis=1)
            
            timestamp_data["timestamp"] = t_stp
            
            for i in tmp_timestamp_data.itertuples():
                if i.device == 'storage':
                    timestamp_data["scenario_id"] = self._scenario_id
                    timestamp_data["grid_price"] = i.grid_cost
                    timestamp_data["es_cost"] = i.cost
                    timestamp_data["es_reward"] = i.reward
                    timestamp_data["es_action"] = i.action[-1]
                    timestamp_data["es_power_ask"] = i.device_custom_info["power_ask"]
                    timestamp_data["es_current_storage"] = i.device_custom_info["current_storage"]
                    timestamp_data["es_solar_power_consumed"] = i.solar_power_consumed
                    timestamp_data["es_grid_power_consumed"] = i.grid_power_consumed
                    timestamp_data["es_post_solar_power_available"] = i.device_custom_info["solar_power_available"]
                    timestamp_data["es_post_grid_power_available"] = i.device_custom_info["grid_power_available"]
                    timestamp_data["es_post_es_power_available"] = i.device_custom_info["es_power_available"]
                elif i.device == 'ev-charging':
                    timestamp_data["ev_cost"] = i.cost
                    timestamp_data["ev_reward"] = i.reward
                    timestamp_data["ev_action"] = i.action[-1]
                    timestamp_data["ev_power_ask"] = i.device_custom_info["power_ask"]
                    timestamp_data["ev_power_unserved"] = i.device_custom_info["power_unserved"]
                    timestamp_data["ev_charging_vehicle"] = i.device_custom_info["charging_vehicle"]
                    timestamp_data["ev_vehicle_charged"] = i.device_custom_info["vehicle_charged"]
                    timestamp_data["ev_post_solar_power_available"] = i.device_custom_info["solar_power_available"]
                    timestamp_data["ev_post_es_power_available"] = i.device_custom_info["es_power_available"]
                    timestamp_data["ev_post_grid_power_available"] = i.device_custom_info["grid_power_available"]
                    timestamp_data["ev_solar_power_consumed"] = i.solar_power_consumed
                    timestamp_data["ev_es_power_consumed"] = i.es_power_consumed
                    timestamp_data["ev_grid_power_consumed"] = i.grid_power_consumed
                elif i.device == 'other-devices':
                    timestamp_data["oth_dev_cost"] = i.cost
                    timestamp_data["oth_dev_reward"] = i.reward
                    timestamp_data["oth_dev_action"] = i.action[-1]
                    timestamp_data["oth_dev_solar_power_consumed"] = i.solar_power_consumed
                    timestamp_data["oth_dev_es_power_consumed"] = i.es_power_consumed
                    timestamp_data["oth_dev_grid_power_consumed"] = i.grid_power_consumed
                    timestamp_data["oth_dev_power_ask"] = i.device_custom_info["power_ask"]
                    timestamp_data["oth_dev_post_solar_power_available"] = i.device_custom_info["solar_power_available"]
                    timestamp_data["oth_dev_post_es_power_available"] = i.device_custom_info["es_power_available"]
                    timestamp_data["oth_dev_post_grid_power_available"] = i.device_custom_info["grid_power_available"]
                elif i.device == 'pv':
                    timestamp_data["pv_reward"] = i.reward
                    timestamp_data["solar_action"] = i.action[-1]
                    timestamp_data["solar_available_power"] = i.device_custom_info["pv_available_power"]
                    timestamp_data["solar_actionable_power"] = i.device_custom_info["pv_actionable_power"]


            final_csv_rows.append(timestamp_data)
        
        csvname = "final_validation"
        dump_file_name = osp.join(logdir, csvname+".csv")

        final_df = pd.DataFrame(final_csv_rows)
        final_df.to_csv(dump_file_name, sep=',', encoding='utf-8')

        if self._is_push_data_inline:
            self._push_data(logdir, csvname)


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

    scenario_id = args["scenario_id"]
    is_push_data_inline = args["push_data_inline"]

    # Configure hyperparameters of the RL algorithm.  train_batch_size is fixed
    # so that results are reproducible, but 34 CPU workers were used in training 
    # -- expect slower performence if using fewer.
    hyperparam_config = {
        "lr": 1e-3,
        "num_sgd_iter": 10,
        "entropy_coeff": 0.0,
        "train_batch_size": rollout_fragment_length,   # ensure reproducible
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
    }

    # Run the trial.
    experiment = tune.run(
        args["run"],
        local_dir=args["local_dir"],
        checkpoint_freq=100,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=100,
        stop=stop,
        callbacks=[HSDataLoggerCallback(scenario_id, is_push_data_inline)],
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