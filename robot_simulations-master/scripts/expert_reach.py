import gym
import numpy as np
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

import behaviour_gym


env = gym.make("Reach-Ur103f-v0")

def expert(obs):
    armPos = obs[:3]
    blockPos = obs[22:25]
    goalPos = blockPos + env.goalPosOffset
    dist = goalPos - armPos
    action = dist / env.maxArmMove
    return action

generate_expert_traj(expert, '../tmp/expert_reach_Ur103f_v0', env,
                     n_episodes=150)
