

from gridworld.agents.pv import PVEnv


class HSPVEnv(PVEnv):

    def get_obs(self, **kwargs):
    
        obs, meta =super().get_obs(**kwargs)
        meta.update(kwargs)
        meta['pv_power']=meta['real_power']

        
        return obs,meta
    
    def reset(self, **kwargs):
        """Resetting consists of simply putting the index back to 0."""
        self.index = 0
        return self.get_obs(**kwargs)
    
    def is_terminal(self):
        """The episode is done when the end of the data is reached."""
        return self.index == (self.episode_length)