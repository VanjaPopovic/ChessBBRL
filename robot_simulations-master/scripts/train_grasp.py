import argparse
import os
import warnings
import typing
from typing import Union, Optional

import gym
import numpy as np
from stable_baselines import SAC, PPO2, TD3
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import EventCallback, BaseCallback
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecEnv, sync_envs_normalization, SubprocVecEnv
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.sac.policies import FeedForwardPolicy as SacFeedForwardPolicy
from stable_baselines.td3.policies import MlpPolicy as Td3MlpPolicy
import tensorflow as tf

import behaviour_gym

def make_env(env_id, rank, env_kwargs={}, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, **env_kwargs)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='environment name',
                        default='Grasp-Ur103f-v0')
    parser.add_argument('--algo', type=str, help='algorithm name',
                        default='SAC')
    parser.add_argument('--log-id', type=str, help='ID for saving data')
    parser.add_argument('--seed', help='Random generator seed', type=int,
                        default=0)
    args = parser.parse_args()

    envName = args.env
    algoName = args.algo.upper()
    logPath = "../logs/" + envName + "/" + algoName + args.log_id


    if algoName == "PPO":
        env = SubprocVecEnv([make_env(envName, i) for i in range(32)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
                           clip_obs=10., training=True)
    elif algoName == "SAC":
        env = DummyVecEnv([lambda: gym.make(envName)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
                           clip_obs=10., training=True)
    elif algoName == "TD3":
        env = DummyVecEnv([lambda: gym.make(envName)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False,
        clip_obs=10., training=True)
    else:
        exit()

    if args.seed < 0:
        args.seed = np.random.randint(2**32 - 1)
    set_global_seeds(args.seed)
    env.seed(args.seed)

    def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                        render=False, callback=None, reward_threshold=None,
                        return_episode_rewards=False):
        """
        Runs policy for `n_eval_episodes` episodes and returns average reward.
        This is made to work only with one env.

        :param model: (BaseRLModel) The RL agent you want to evaluate.
        :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
            this must contain only one environment.
        :param n_eval_episodes: (int) Number of episode to evaluate the agent
        :param deterministic: (bool) Whether to use deterministic or stochastic actions
        :param render: (bool) Whether to render the environment or not
        :param callback: (callable) callback function to do additional checks,
            called after each step.
        :param reward_threshold: (float) Minimum expected reward per episode,
            this will raise an error if the performance is not met
        :param return_episode_rewards: (bool) If True, a list of reward per episode
            will be returned instead of the mean.
        :return: (float, float) Mean reward per episode, std of reward per episode
            returns ([float], [int]) when `return_episode_rewards` is True
        """
        if isinstance(env, VecEnv):
            assert env.num_envs == 1, "You must pass only one environment when using this function"

        episode_rewards, episode_lengths, episode_success, episode_contacts, episode_lift_dists, episode_orn_dist, episode_centre_dist, episode_finger_tip_dist = [], [], [], [], [], [], [], []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            while not done:
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                if callback is not None:
                    callback(locals(), globals())
                episode_length += 1
                if render:
                    env.render()
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            info = info[0]
            episode_lift_dists.append(info['lift distance'])
            episode_contacts.append(info['num finger contacts'])
            episode_success.append(info['is_success'])
            episode_orn_dist.append(info['Orientaton distance'])
            episode_centre_dist.append(info['Centering distance'])
            episode_finger_tip_dist.append(info['Finger Tip distance'])

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        if reward_threshold is not None:
            assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
                                             '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
        if return_episode_rewards:
            return episode_rewards, episode_lengths, episode_contacts, episode_lift_dists, episode_success, episode_orn_dist, episode_centre_dist, episode_finger_tip_dist
        return mean_reward, std_reward


    class EvalCallback(EventCallback):
        """
        Callback for evaluating an agent.

        :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
        :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
            when there is a new best model according to the `mean_reward`
        :param n_eval_episodes: (int) The number of episodes to test the agent
        :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
        :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
            will be saved. It will be updated at each evaluation.
        :param best_model_save_path: (str) Path to a folder where the best model
            according to performance on the eval env will be saved.
        :param deterministic: (bool) Whether the evaluation should
            use a stochastic or deterministic actions.
        :param render: (bool) Whether to render or not the environment during evaluation
        :param verbose: (int)
        """
        def __init__(self, eval_env: Union[gym.Env, VecEnv],
                     callback_on_new_best: Optional[BaseCallback] = None,
                     n_eval_episodes: int = 5,
                     eval_freq: int = 10000,
                     log_path: str = None,
                     best_model_save_path: str = None,
                     deterministic: bool = True,
                     render: bool = False,
                     verbose: int = 1,
                     training_env = None):
            super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
            self.n_eval_episodes = n_eval_episodes
            self.eval_freq = eval_freq
            self.best_mean_reward = -np.inf
            self.last_mean_reward = -np.inf
            self.deterministic = deterministic
            self.render = render

            # Convert to VecEnv for consistency
            if not isinstance(eval_env, VecEnv):
                eval_env = DummyVecEnv([lambda: eval_env])

            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

            self.eval_env = eval_env
            self.best_model_save_path = best_model_save_path
            # Logs will be written in `evaluations.npz`
            if log_path is not None:
                log_path = os.path.join(log_path, 'evaluations')
            self.log_path = log_path
            self.evaluations_results = []
            self.evaluations_timesteps = []
            self.evaluations_length = []

        def _init_callback(self):
            # Does not work in some corner cases, where the wrapper is not the same
            if not type(self.training_env) is type(self.eval_env):
                warnings.warn("Training and eval env are not of the same type"
                              "{} != {}".format(self.training_env, self.eval_env))

            # Create folders if needed
            if self.best_model_save_path is not None:
                os.makedirs(self.best_model_save_path, exist_ok=True)
            if self.log_path is not None:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        def _on_step(self) -> bool:

            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                # Sync training and eval env if there is VecNormalize
                sync_envs_normalization(env, self.eval_env)

                episode_rewards, episode_lengths, episode_contacts, episode_lift_dists, episode_success, episode_orn_dist, episode_centre_dist, episode_finger_tip_dist = evaluate_policy(self.model, self.eval_env,
                                                                   n_eval_episodes=self.n_eval_episodes,
                                                                   render=self.render,
                                                                   deterministic=self.deterministic,
                                                                   return_episode_rewards=True)

                if self.log_path is not None:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)
                    np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                             results=self.evaluations_results, ep_lengths=self.evaluations_length)

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                # Keep track of the last evaluation, useful for classes that derive from this callback
                self.last_mean_reward = mean_reward

                if self.verbose > 0:
                    print("Eval num_timesteps={}, "
                          "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                    print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                        self.eval_env.save(os.path.join(self.best_model_save_path, "best_env_normalize.pkl"))

                    self.best_mean_reward = mean_reward
                    # Trigger callback if needed
                    if self.callback is not None:
                        return self._on_event()

                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/mean_episode_reward', simple_value=mean_reward)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/mean_episode_succcess', simple_value=np.mean(episode_success))])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/final_goal_lift_dist', simple_value=np.mean(episode_lift_dists))])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/final_num_finger_contacts', simple_value=np.mean(episode_contacts))])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/final_orn_dist', simple_value=np.mean(episode_orn_dist))])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/final_centre_dist', simple_value=np.mean(episode_centre_dist))])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                summary = tf.Summary(value=[tf.Summary.Value(tag='eval/final_finger_tip_dist', simple_value=np.mean(episode_finger_tip_dist))])
                self.locals['writer'].add_summary(summary, self.num_timesteps)

            return True

    eval_env = DummyVecEnv([lambda: gym.make("Grasp-Ur103f-v0")])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10., training=False)

    if algoName == "PPO":
        eval_callback = EvalCallback(eval_env, best_model_save_path=logPath, n_eval_episodes=10,
                                     log_path=logPath, eval_freq=1000, deterministic=True, render=False, training_env=env)

        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   net_arch=[128, 128],
                                                   feature_extraction="mlp")

        register_policy('CustomPolicy', CustomPolicy)

        # model = PPO2(CustomPolicy, env, n_steps=512, nminibatches=256, noptepochs=10,
        #              verbose=1, tensorboard_log=logPath)
        # model = PPO2(CustomPolicy, env, n_steps=512, nminibatches=512, noptepochs=10,
        #              verbose=1, tensorboard_log=logPath)
        # model = PPO2(CustomPolicy, env, n_steps=512, nminibatches=128, noptepochs=10,
        #              verbose=1, tensorboard_log=logPath)
        # model = PPO2(CustomPolicy, env, n_steps=1024, nminibatches=256, noptepochs=10,
        #              verbose=1, tensorboard_log=logPath)
        model = PPO2(CustomPolicy, env, n_steps=512, nminibatches=256, noptepochs=15,
                     verbose=1, tensorboard_log=logPath)
    elif algoName == "SAC":
        eval_callback = EvalCallback(eval_env, best_model_save_path=logPath, n_eval_episodes=10,
                                     log_path=logPath, eval_freq=5000, deterministic=True, render=False, training_env=env)

        """
        Sac32 from hyperparam optim

        SacReach:   'gamma': 0.9, 'lr': 0.0001368199916822424, 'batch_size': 64, 'buffer_size': 100000, 'learning_starts': 1000, 'train_freq': 1, 'ent_coef': 0.05, 'net_arch': 'small'
        Sac32:      'gamma': 0.9, 'lr': 0.0025823829724934038, 'batch_size': 256, 'buffer_size': 100000, 'learning_starts': 10000, 'train_freq': 10, 'ent_coef': 0.05, 'net_arch': 'medium'
        """

        class CustomPolicy(SacFeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   layers=[256, 256],
                                                   feature_extraction="mlp")

        register_policy('CustomPolicy', CustomPolicy)

        model = SAC(CustomPolicy, env, gamma=0.9,
                    learning_rate=0.0025823829724934038, buffer_size=1000000,
                    learning_starts=10000, train_freq=10, batch_size=256,
                    ent_coef=0.05, verbose=1, tensorboard_log=logPath)
        # model = SAC(CustomPolicy, env, buffer_size=1000000, batch_size=256,
        #             ent_coef='auto0.02', learning_starts=1000, verbose=1,
        #             tensorboard_log=logPath)
    elif algoName == "TD3":
        eval_callback = EvalCallback(eval_env, best_model_save_path=logPath, n_eval_episodes=10,
                                     log_path=logPath, eval_freq=3200, deterministic=True, render=False, training_env=env)

        action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))
        model = TD3(Td3MlpPolicy, env, buffer_size=1000000, learning_starts=1000,
                    action_noise=action_noise, verbose=1, tensorboard_log=logPath)

    model.learn(total_timesteps=1000000, log_interval=10, callback=eval_callback)
    model.save(logPath + "/final_model")
    env.save(logPath + "/final_env_normalize.pkl")
