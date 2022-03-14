import time
import math
import numpy as np

from mjrl.utils.gym_env import GymEnv
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import time as timer
import pickle
import gym
from gym.envs.registration import register

import behaviour_gym
from behaviour_gym.scene import Table
from behaviour_gym.primitive import Reach as Reach
SEED = 500

class Env(gym.Env):
    def __init__(self):
        # self.scene = Basic(guiMode=False)
        self.scene = Table(guiMode=True)
        self.robot = self.scene.loadRobot("ur103f")
        self.robot.resetBasePose(pos=[0, 0, 0.6])
        self.scene.setInit()
        
        goalObj = "block"
        goalOffset = [0.00, 0.00, 0.04]
        maxMove = 0.08

        self.reach = Reach(goalObj, goalOffset, scene=self.scene, robot=self.robot, timestep=1. / 240.)

        # Gym metadata and spaces
        self.metadata = {'render.modes': ['human', 'rgbd_array']}
        nObs = self.reach._getNumObs()
        self.observation_space = gym.spaces.Box(np.float32(-np.inf),
                                                np.float32(np.inf),
                                                shape=(nObs,),
                                                dtype=np.float32)
        nActions = self.reach._getNumActions()
        self.action_space = gym.spaces.Box(np.float32(-1), np.float32(1),
                                           shape=(nActions,),
                                           dtype=np.float32)

    def seed(self, seed=None):
        return self.reach.seed(seed)

    def reset(self):
        obs = self.reach.reset()
        #self.scene.visualisePose(self.reach.goalPos, self.reach.goalOrn)
        return obs

    def step(self, action):
        return self.reach.step(action)

    def render(self, mode="rgbd_array"):
        return self.reach.render(mode)

    def close(self):
        return self.reach.close()

    def get_obs(self):
        return self.reach.getObs()

# class Env:
#     def __init__(self):
#         scene = Basic(guiMode=True)
#         robot = scene.loadRobot("ur103f")
#         robot.resetBasePose(pos=[0,0,0])
#         scene.setInit()
#         reach = Reach(scene=scene, robot=robot)
#
#     @property
#     def spec(self):
#         return self.env.spec


register(
    id='myenv-v0',
    entry_point=Env,
    max_episode_steps=100
)

e = GymEnv("myenv-v0")
robot = e.env.robot
reach = e.env.reach

class Expert:
    def get_action(self, observation):
      def get_action(self, observation):
        # Get the objects pose from the scene
        goalObj = "block"
        goalOffset = [0.0, 0.0, 0.04]
        maxMove = 0.08
        objPos, _ = scene.getPose(goalObj)

        # Add goalOffset to find goal position
        goalPos = np.add(objPos, goalOffset)

        # Get the robot's pose
        armPos, armOrn = robot.getPose()

        # Find the difference between the robot and goal pose
        dif = np.subtract(goalPos, armPos)
        print("dif")
        print(dif)
        # Clip difference between max move
        clampedDif = np.clip(dif, -maxMove, maxMove)

        # Rescale from [-0.05, 0.05] to [-1, 1]
        action = 2 * ((clampedDif + 0.05) / (0.1)) - 1

        action = [action[0], action[1], action[2], 0, 0, 0]
        print(action)
        action = np.array(action)
        return action, {'mean': action, 'log_std': np.array([0, 0, 0, 0, 0, 0]), 'evaluation': action}

# ------------------------------
# Load trained agent
policy_file = 'policy_bc.pickle'
with open("logs/reach/" + policy_file, 'rb') as pickle_file:
    policy = pickle.load(pickle_file)

# ------------------------------
# Evaluate Policies
time.sleep(5)
bc_pol_score = e.evaluate_policy(policy, num_episodes=50, mean_action=True)
expert_score = e.evaluate_policy(Expert(), num_episodes=50, mean_action=True)
print("Expert policy performance (eval mode) = %f" % expert_score[0][0])
print("BC policy performance (eval mode) = %f" % bc_pol_score[0][0])
