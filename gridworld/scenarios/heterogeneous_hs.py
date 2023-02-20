from os import system
import pandas as pd

from os import system
import pandas as pd

from gridworld import MultiComponentEnv
from gridworld import MultiAgentEnv

from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.agents.hvac import HVACEnv
import os


def load_grid_cost(start_time: str = None, end_time: str = None) -> list:
    """Returns exogenous data dataframe, and state space model (per-zone) dict."""

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(THIS_DIR, "data/grid_cost.csv"), index_col=0)
    df.index = pd.DatetimeIndex(df.index)

    start_time = pd.Timestamp(start_time) if start_time else df.index[0]
    end_time = pd.Timestamp(end_time) if end_time else df.index[-1]

    _df = df.loc[start_time:end_time]

    if _df is None or len(_df) == 0:
        raise ValueError(
            f"start and/or end times ({start_time}, {end_time}) " +
            "resulted in empty dataframe.  First and last indices are " +
            f"({df.index[0]}, {df.index[-1]}), choose values in this range.")

    return _df['grid_cost'].tolist()


def make_env_config( rescale_spaces=True):

    # Make the multi-component building

    grid_cost=load_grid_cost(env_config['start_time'],env_config['end_time'] )

    pv = {
        "name": "pv",
        "cls": PVEnv,
        "config": {
            "profile_csv": "pv_profile_hs.csv",
            "scaling_factor": 40.,
            "rescale_spaces": rescale_spaces
        }
    }

    battery = {
        "name": "storage",
        "cls": EnergyStorageEnv,
        "config": {
            "max_power"                 : 10.0,
            "storage_range"             : (1., 28.),
            "initial_storage_mean"      : 14.0,  # 7*2
            "initial_storage_std"       : 2.828, # 2*sqrt(2)
            "charge_efficiency"         : 0.95,
            "discharge_efficiency"      : 0.95,
            "init_storage"              : 0.0,
            "rescale_spaces"            : rescale_spaces,
            'initial_storage_cost'      : min(grid_cost),
            'max_storage_cost'          : max(grid_cost)
        } 
    }

    ev = {
            "name": "ev-charging",
            "cls": EVChargingEnv,
            "config": {
                "num_vehicles"          : 1,
                "minutes_per_step"      : 5,
                "max_charge_rate_kw"    : 11.,
                "peak_threshold"        : 200.,
                "vehicle_multiplier"    : 40.,
                "rescale_spaces"        : rescale_spaces
            }
        }

    hvac = {
        "name": "storage",
        "cls": HVACEnv,
        "config": {
            "max_power": 20.,
            "storage_range": (3., 250.),
            "rescale_spaces": rescale_spaces
        } 
    }




    house_components = [pv, battery, ev ]



    env_config={ "components"       : house_components,
                 "start_time"       : "08-31-2020 00:00:00",
                 "end_time"         : "08-31-2020 23:55:00",
                 "control_timedelta": pd.Timedelta(300, "s"),
                 'max_grid_power'   :  48,
                 
                   }
    env_config['grid_cost']=grid_cost

    return env_config




