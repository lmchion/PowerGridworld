"""Script for running single-machine training.  If you want to run rllib on a 
cluster see, e.g., https://docs.ray.io/en/latest/cluster/deploy.html."""
import json
import sys
import time
from collections import OrderedDict

import ray
from callbacks import HSAgentTrainingCallback, HSDataLoggerCallback
from ray import tune
from ray.air.checkpoint import Checkpoint
from ray.tune.registry import register_env
from ray.cluster_utils import Cluster

from gridworld.log import logger
from gridworld.scenarios.heterogeneous_hs import make_env_config
import gymnasium as gym



def env_creator(config: dict):
    """Simple wrapper that takes a config dict and returns an env instance."""
    
    
    gym.register(id=config['name']+'-v0',
                          entry_point='gridworld.base_hs:HSMultiComponentEnv',
                          max_episode_steps=config['max_episode_steps']
                         )
    env = gym.make( id='gridworld.base_hs:'+config['name']+'-v0',**config )
    #env.action_space.seed(123)

    return env

def main(**args):

    # Log the args used for training.
    logger.info(f"ARGS: {args}")
    
    # Run ray on a single node.  If running on VPN, you might need some bash 
    # magic, see e.g. train.sh.
    ray.init(_node_ip_address=args["node_ip_address"],  log_to_driver=True, logging_level="error")
    #num_cpus=args['num_cpus'], num_gpus=args['num_gpus'],

    if args['last_checkpoint']!='None':
        #checkpoint=Checkpoint(local_path= args['last_checkpoint'])
        checkpoint=args['last_checkpoint']
    else:
        checkpoint=None
            
    # Register the environment.
    env_name = args["scenario_id"]
    register_env(env_name, env_creator)

    # Create the env configuration with option to change max episode steps
    # for debugging.
    with open(args["input_dir"]+'/'+ str(env_name) +'.json', 'r') as f:
        env_config = json.load(f)

    env_config = make_env_config(env_config)
    #env_config.update({"max_episode_steps": args["max_episode_steps"]})

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
    scenario_id = args["scenario_id"]


    # Set any stopping conditions.
    stop = {
        #'training_iteration': args["stop_iters"],
        'training_iteration': int(args["training_iteration"]),
       # 'timesteps_total': int(args["training_iteration"]),
        # 'max_episode_steps' : int(args["stop_timesteps"]),
        'episode_reward_mean': float(args["stop_reward"])
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
        "train_batch_size": rollout_fragment_length*30,   # ensure reproducible
        "rollout_fragment_length": rollout_fragment_length,
        #"rollout_fragment_length": 'auto',
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
    }

    # Run the trial.
    experiment = tune.run(
        args["run"],
        local_dir=args["local_dir"],
        checkpoint_freq=1000,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=100,
        stop=stop,
        callbacks=[HSDataLoggerCallback(scenario_id)],
        restore=checkpoint,
        #resume="AUTO",
        config={
            "env": env_name,
            "env_config": env_config,
            "num_gpus": args["num_gpus"],
            "num_workers": num_workers,
            "horizon" : rollout_fragment_length,
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

    #dir(experiment)
    # not_ready=[]
    # while len(not_ready)==0:
    #     ready, not_ready = ray.wait([env])
    #     print("ready : ",ready,"not ready : ",not_ready)

    # while (ray.global_state.cluster_resources() !=
    #    ray.global_state.available_resources()):
    #     time.sleep(1)
    # start=time.time()
    # while (experiment.get_best_logdir("training_iteration", mode="max") not in experiment.trial_dataframes.keys() ): # and time.time()-start < 2*60 ):
    #     print("waiting")
    #     time.sleep(1)
    #ray.shutdown(_exiting_interpreter= False)

    last_checkpoint=experiment.get_last_checkpoint()
    return(last_checkpoint)

    


if __name__ == "__main__":

    from args_hs import parser

    args = parser.parse_args()

    last_check = main(**vars(args))
    print(last_check, file=sys.stdout)