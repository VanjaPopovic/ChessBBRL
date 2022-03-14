import gym
import behaviour_gym

env = gym.make("Grasp-Ur103f-v0", renders=True, timestep=1./60.)
obs = env.reset()
for _ in range(100):
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()

env.close()