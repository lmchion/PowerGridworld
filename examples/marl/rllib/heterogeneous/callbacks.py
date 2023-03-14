
import os
import os.path as osp
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.tune.logger import LoggerCallback
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from gridworld.log import logger
import json

class HSAgentTrainingCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker : RolloutWorker, base_env : BaseEnv, 
        policies: Dict[str, Policy], episode : Episode , env_index : int, **kwargs
    ):
        #episode.user_data["episode_data"] = defaultdict(list)
        episode.media["episode_data"] = []

    def on_episode_step(
        self, *, worker : RolloutWorker, base_env : BaseEnv, episode : Episode , env_index : int, **kwargs
    ):
        # TODO change this in subcomponents to use the component name to remove hard-coding.
        #print(dir(episode))
        #ep_lastinfo = episode.last_info_for()
        agents = episode.get_agents()
        ep_lastinfo = episode._last_infos[agents[-1]]


        step_meta = ep_lastinfo.get('step_meta', None)
        grid_cost = ep_lastinfo.get('grid_cost', None)
        es_cost = ep_lastinfo.get('es_cost', None)
        hvac_power = ep_lastinfo.get('hvac_power', None)
        other_power = ep_lastinfo.get('other_power', None)
        
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

class HSDataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = osp.join(trial.logdir, "episode_data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

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


        step = result['timesteps_total']
        dump_file_name = osp.join(
            self._trial_local_dir[trial], f"data-{step:08d}.json"
        )

        data = episode_media["episode_data"]            

        episode_data = data[-1]

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

        if not episode_data:
            logger.info("Episode data tranche is empty while logging. skipping.")
        
        df = pd.DataFrame(episode_data, columns=extract_columns)

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
                            "pv_power", 
                            "es_power", 
                            "grid_power",
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
                    timestamp_data["grid_cost"] = i.grid_cost
                    timestamp_data["es_cost"] = i.cost
                    timestamp_data["es_reward"] = i.reward
                    timestamp_data["es_dev_action"] = i.action
                    timestamp_data["es_current_storage"] = i.device_custom_info["current_storage"]
                    timestamp_data["es_current_psudo_cost"] = i.device_custom_info["current_cost"]
                    timestamp_data["es_consumed_pv_power"] = i.pv_power
                    timestamp_data["es_consumed_grid_power"] = i.grid_power
                elif i.device == 'ev-charging':
                    timestamp_data["ev_cost"] = i.cost
                    timestamp_data["ev_reward"] = i.reward
                    timestamp_data["ev_dev_action"] = i.action
                    timestamp_data["ev_power_unserved"] = i.device_custom_info["power_unserved"]
                    timestamp_data["ev_consumed_es_power"] = i.es_power
                    timestamp_data["ev_consumed_pv_power"] = i.pv_power
                    timestamp_data["ev_consumed_grid_power"] = i.grid_power
                elif i.device == 'other-devices':
                    timestamp_data["oth_dev_cost"] = i.cost
                    timestamp_data["oth_dev_reward"] = i.reward
                    timestamp_data["oth_dev_action"] = i.action
                    timestamp_data["oth_dev_consumed_es_power"] = i.es_power
                    timestamp_data["oth_dev_consumed_pv_power"] = i.pv_power
                    timestamp_data["oth_dev_consumed_grid_power"] = i.grid_power
                elif i.device == 'pv':
                    timestamp_data["pv_action"] = i.action
                    timestamp_data["pv_power"] = i.pv_power


            final_csv_rows.append(timestamp_data)
        
        dump_file_name = osp.join(logdir, "final_validation.csv")

        final_df = pd.DataFrame(final_csv_rows)
        final_df.to_csv(dump_file_name, sep=',', encoding='utf-8')