import argparse

import gym
import numpy as np

from stable_baselines import PPO2, SAC, TD3
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

import behaviour_gym


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='environment name',
                        default='Reach-Ur103f-v0')
    parser.add_argument('--algo', type=str, help='algorithm name',
                        default='SAC')
    parser.add_argument('--log-id', type=str, help='ID of model to load')
    parser.add_argument('--final', type=str2bool,
                        help='run final instead of best model', nargs='?',
                        const=True,  default=False)
    parser.add_argument('--episodes', type=int, help='no episodes to run',
                        default=10)
    args = parser.parse_args()

    envName = args.env
    algoName = args.algo.upper()
    useFinal = args.final
    episodes = args.episodes
    logPath = "../logs/" + envName + "/" + algoName + args.log_id
    if useFinal:
        modelPath = logPath + "/final_model"
        statsPath = logPath + "/final_env_normalize.pkl"
    else:
        modelPath = logPath + "/best_model"
        statsPath = logPath + "/best_env_normalize.pkl"


    env = DummyVecEnv([lambda: gym.make(envName, renders=True)])
    env = VecNormalize.load(statsPath, env)
    env.training = False
    env.norm_reward = False

    if algoName == "PPO":
        model = PPO2.load(modelPath)
    elif algoName == "SAC":
        model = SAC.load(modelPath)
    elif algoName == "TD3":
        model = TD3.load(modelPath)
    else:
        exit()
    model.set_env(env)

    episode_rewards, episode_lengths, episode_dists, episode_collisions, episode_success = [], [], [], [], []
    for i in range(episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_dists.append(info[0]['goal distance'])
        episode_collisions.append(info[0]['collisions'])
        episode_success.append(info[0]['is_success'])

    print("No Episodes: ", len(episode_rewards))
    print("Mean Reward: ", np.mean(episode_rewards))
    print("Mean Success Rate: {}%".format(np.mean(episode_success)))
    print("Mean Final Goal Distance: {}cm".format(np.mean(episode_dists)))
    print("Mean No Collisions: ", np.mean(episode_collisions))
