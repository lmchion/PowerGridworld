"""Script for running single-machine training.  If you want to run rllib on a 
cluster see, e.g., https://docs.ray.io/en/latest/cluster/deploy.html."""
import json



import ray
from ray import tune

from ray.tune.registry import register_env

from gridworld.log import logger
from gridworld.scenarios.heterogeneous_hs import make_env_config
import json
from collections import OrderedDict
from itertools import permutations,cycle, islice
import random
from callbacks import HSAgentTrainingCallback, HSDataLoggerCallback

        

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

    with open(args["input_dir"]+ '/map.json', 'r') as f:
        map = json.load(f)



    last_checkpoint=args["last_checkpoint"]

    for env in list(map.keys()):
        print("env",env)

        
        # Register the environment.
        #env_name = args["env_name"]
        env_name = env
        register_env(env_name, env_creator)

        # Create the env configuration with option to change max episode steps
        # for debugging.
        env_config = make_env_config(args["input_dir"]+'/'+ env +'.json')
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
            #'training_iteration': args["stop_iters"],
            'training_iteration': 1,
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
        evaluation_config = { "simple_optimizer" : True,
                               "num_workers"  : 0,
                               "timesteps_per_iteration" : 0,
                                "evaluation_interval" : 1}  # indicates how many train calls an evaluation step is run
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
         #   "train_batch_size": rollout_fragment_length,   # ensure reproducible
            #"rollout_fragment_length": rollout_fragment_length,
            "rollout_fragment_length": 'auto',
            "batch_mode": "complete_episodes",
            "observation_filter": "MeanStdFilter",
        }

        # Run the trial.
        experiment = tune.run(
            args["run"],
            local_dir=args["local_dir"],
            checkpoint_freq=100,
            checkpoint_at_end=False,
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
                "restore" : last_checkpoint,
                **framework_config,
                **hyperparam_config,
                **evaluation_config
            },
            verbose=0
        )



        #return experiment


if __name__ == "__main__":

    from args_hs import parser

    args = parser.parse_args()

    _ = main(**vars(args))