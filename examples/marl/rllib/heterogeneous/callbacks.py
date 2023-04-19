
import json
import os
import os.path as osp
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import  RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2 
from ray.rllib.policy import Policy
from ray.tune.logger import LoggerCallback

from gridworld.log import logger


class HSAgentTrainingCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self._total_episode_cost = 0.0   


    def on_episode_start(
        self, *, worker : RolloutWorker, base_env : BaseEnv, 
        policies: Dict[str, Policy], episode : EpisodeV2 , env_index : int, **kwargs
    ):
        #episode.user_data["episode_data"] = defaultdict(list)
        episode.media["episode_data"] = []
        self._total_episode_cost = 0.0
        self._total_datapoints = 0
        self._total_timestamps = 0
        self._devices=[]

    def on_episode_step(
        self, *, worker : RolloutWorker, base_env : BaseEnv, episode : EpisodeV2 , env_index : int, **kwargs
    ):
        #agents = episode.get_agents()
        #ep_lastinfo = episode._last_infos[agents[-1]]

        last_agent = max(episode._agent_to_index)
        ep_lastinfo = episode._last_infos[last_agent]

        step_meta = ep_lastinfo.get('step_meta', None)
        grid_cost = ep_lastinfo.get('grid_cost', None)
        es_cost = ep_lastinfo.get('es_cost', None)
        hvac_power = ep_lastinfo.get('hvac_power', None)
        other_power = ep_lastinfo.get('other_power', None)
        total_cost = 0
        timestamp=None
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

            if step_meta_item["device_id"] not in self._devices:
                self._devices.append(step_meta_item["device_id"])

            if timestamp!=step_meta_item["timestamp"]:
                timestamp=step_meta_item["timestamp"]
                self._total_timestamps +=1

            self._total_datapoints += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        episode.custom_metrics["total_cost"] = self._total_episode_cost*self._total_timestamps*len(self._devices) / self._total_datapoints 

class HSDataLoggerCallback(LoggerCallback):
    def __init__(self, scenario_id):
        super().__init__()

        self._trial_continue = {}
        self._trial_local_dir = {}
        self._scenario_id = scenario_id

    def log_trial_start(self, trial):
        trial.init_logdir()

    def log_trial_result(self, iteration, trial, result):

        episode_media = result["episode_media"]
        if "episode_data" not in episode_media:
            return

        junk1 = osp.join(trial.logdir, "progress.csv")
        junk2 = osp.join(trial.logdir, "result.json")
        #if os.path.exists(junk1):
        #    os.remove(junk1)
        #if os.path.exists(junk2):
        #    os.remove(junk2)  


    def on_experiment_end(self, trials, **info):
        #print("on_experiment_end dumping the last result for validation..")
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
                    #discharge
                    timestamp_data["es_action_d"] = i.action[1]
                    timestamp_data["es_power_ask_d"] = i.device_custom_info["power_ask"][1]
                    
                    #charge
                    timestamp_data["es_action_c"] = i.action[0]
                    timestamp_data["es_power_ask_c"] = i.device_custom_info["power_ask"][0]
                    
                    #consolidated
                    timestamp_data["scenario_id"] = self._scenario_id
                    timestamp_data["grid_price"] = i.grid_cost
                    timestamp_data["es_cost"] = i.cost
                    timestamp_data["es_reward"] = i.reward
                    timestamp_data["es_action"] = sum(i.action)
                    timestamp_data["es_power_ask"] = sum(i.device_custom_info["power_ask"])
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

            timestamp_data["ev_post_es_power_available"] = round(timestamp_data["es_post_es_power_available"] - timestamp_data["ev_es_power_consumed"],5)+0.0
            timestamp_data["ev_post_solar_power_available"] = round(timestamp_data["es_post_solar_power_available"] - timestamp_data["ev_solar_power_consumed"],5)+0.0

            timestamp_data["oth_dev_post_es_power_available"] = round(timestamp_data["ev_post_es_power_available"] - timestamp_data["oth_dev_es_power_consumed"],5)+0.0
            timestamp_data["oth_dev_post_solar_power_available"] = round(timestamp_data["ev_post_solar_power_available"] - timestamp_data["oth_dev_solar_power_consumed"],5)+0.0

            final_csv_rows.append(timestamp_data)
        
        dump_file_name = osp.join(logdir, "final_validation.csv")

        final_df = pd.DataFrame(final_csv_rows)
        final_df.to_csv(dump_file_name, sep=',', encoding='utf-8')