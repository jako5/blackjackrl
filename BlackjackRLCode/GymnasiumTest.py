# import gymnasium as gym
# env = gym.make("Blackjack-v1")
# print(env.rollout(10))

from torchrl.envs.libs.gym import GymEnv
env = GymEnv("Blackjack-v1")

# NotImplementedError: method is not currently implemented. If you are interested in this feature please submit an issue at https://github.com/pytorch/rl/issues