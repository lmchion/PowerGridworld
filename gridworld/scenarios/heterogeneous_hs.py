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


def make_env_config( rescale_spaces=True):

    # Make the multi-component building
    

    pv = {
        "name": "pv",
        "cls": PVEnv,
        "config": {
            "profile_csv": "off-peak.csv",
            "scaling_factor": 40.,
            "rescale_spaces": rescale_spaces
        }
    }

    battery = {
        "name": "storage",
        "cls": EnergyStorageEnv,
        "config": {
            "max_power": 20.,
            "storage_range": (3., 250.),
            "rescale_spaces": rescale_spaces
        } 
    }

    ev = {
            "name": "ev-charging",
            "cls": EVChargingEnv,
            "config": {
                "num_vehicles": 25,
                "minutes_per_step": 5,
                "max_charge_rate_kw": 7.,
                "peak_threshold": 200.,
                "vehicle_multiplier": 40.,
                "rescale_spaces": rescale_spaces
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




    house_components = [pv, battery ]



    env_config={ "components"       : house_components,
                 "start_time"       : "08-12-2020 00:00:00",
                 "end_time"         : "08-13-2020 00:00:00",
                 "control_timedelta": pd.Timedelta(300, "s") }

    return env_config
