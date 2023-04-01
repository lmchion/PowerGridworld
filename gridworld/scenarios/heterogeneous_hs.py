import json
import os
import sys
import time
from os import system

import pandas as pd

from gridworld import MultiAgentEnv, MultiComponentEnv
from gridworld.agents.devices import HSDevicesEnv
from gridworld.agents.energy_storage import HSEnergyStorageEnv
from gridworld.agents.pv import HSPVEnv, PVEnv
from gridworld.agents.vehicles import HSEVChargingEnv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_grid_cost(start_time: str = None, end_time: str = None):
    """Returns exogenous data dataframe, and state space model (per-zone) dict."""

    df = pd.read_csv(os.path.join(THIS_DIR, "data/grid_cost.csv"), delimiter=',')
    time_col = df["time"]
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index)

    df['timestamp'] = time_col.to_list()
    start_time = pd.Timestamp(start_time) if start_time else df.index[0]
    end_time = pd.Timestamp(end_time) if end_time else df.index[-1]

    _df = df.loc[start_time:end_time]

    if _df is None or len(_df) == 0:
        raise ValueError(
            f"start and/or end times ({start_time}, {end_time}) " +
            "resulted in empty dataframe.  First and last indices are " +
            f"({df.index[0]}, {df.index[-1]}), choose values in this range.")

    return (_df['timestamp'].tolist(), _df['grid_cost'].tolist())




def make_env_config(env_config):

    # with open(os.path.join(THIS_DIR, "data/env_config.json"), 'r') as f:
    #     env_config = json.load(f)

    max_grid_power=env_config['max_grid_power']
         
    for elem in env_config['components']:
        if elem['cls']=="HSPVEnv":
            max_pv_power=max(elem['config']['profile_data'])

        if elem['cls']=="HSEnergyStorageEnv":
            max_es_power=elem['config']['max_power']

        

    for elemx in env_config['components']:
        if elemx['cls'] in ["HSEVChargingEnv","HSDevicesEnv","HSEnergyStorageEnv"]:
            elemx['config']['max_grid_power']=max_grid_power
            elemx['config']['max_es_power']=max_es_power
            elemx['config']['max_pv_power']=max_pv_power
        
        elemx['cls']= getattr(sys.modules[__name__], elemx['cls'])

    env_config['control_timedelta']  =  pd.Timedelta(env_config['control_timedelta'])

    return env_config

if __name__ == "__main__":
    
    test=make_env_config()
    print(test)
    r=1





