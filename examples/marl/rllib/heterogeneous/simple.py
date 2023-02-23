import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from gridworld import HSMultiComponentEnv
from gridworld.log import logger
from gridworld.scenarios.heterogeneous_hs import make_env_config


def env_creator(config: dict):
    """Simple wrapper that takes a config dict and returns an env instance."""

    return HSMultiComponentEnv(**config)


def main(**args):

    # Log the args used for training.
    logger.info(f"ARGS: {args}")

    # Run ray on a single node.  If running on VPN, you might need some bash
    # magic, see e.g. train.sh.
    # ray.init(_node_ip_address=args["node_ip_address"])

    # Register the environment.
    env_name = 'test_env'  # args["env_name"]
    register_env(env_name, env_creator)

    # Create the env configuration with option to change max episode steps
    # for debugging.
    env_config = make_env_config(
        rescale_spaces=True
    )
    env_config.update({"max_episode_steps": args["max_episode_steps"]})

    print("ENV CONFIG", env_config)

    # Create an env instance to introspect the gym spaces and episode length
    # when setting up the multiagent policy.
    #env = env_creator(env_config)

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment(env=env_name, env_config=env_config)
        .rollouts(num_rollout_workers=2)
        .framework("tf2")
        .training(model={"fcnet_hiddens": [64, 64]})  # hidden layers
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.


if __name__ == "__main__":

    from args import parser

    args = parser.parse_args()

    _ = main(**vars(args))
