from os import system
import pandas as pd

from os import system
import pandas as pd

from gridworld import MultiComponentEnv
from gridworld import MultiAgentEnv
from gridworld.agents.buildings import FiveZoneROMThermalEnergyEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.distribution_system import OpenDSSSolver


def make_env_config( rescale_spaces=True):

    # Make the multi-component building
    house_components = [ ]

    house_components.append({
        "name": "pv",
        "cls": PVEnv,
        "config": {
            "profile_csv": "off-peak.csv",
            "scaling_factor": 40.,
            "rescale_spaces": rescale_spaces
        }
    })
    house_components.append({
        "name": "storage",
        "cls": EnergyStorageEnv,
        "config": {
            "max_power": 20.,
            "storage_range": (3., 250.),
            "rescale_spaces": rescale_spaces
        } 
    })







    env_config={ "components"       : house_components,
                 "start_time"       : "08-12-2020 00:00:00",
                 "end_time"         : "08-13-2020 00:00:00",
                 "control_timedelta": pd.Timedelta(300, "s") }

    return env_config
