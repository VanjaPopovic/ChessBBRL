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
import mjrl.envs
import time as timer
import pickle
import gym
from gym.envs.registration import register

import behaviour_gym
from behaviour_gym.scene import Basic
from behaviour_gym.primitive import ReachBasicNoIk as Reach
SEED = 500

class Env(gym.Env):
    def __init__(self):
        # self.scene = Basic(guiMode=False)
        self.scene = Basic(guiMode=True)
        self.robot = self.scene.loadRobot("ur103f")
        self.robot.resetBasePose(pos=[0,0,0])
        self.scene.setInit()
        self.reach = Reach(scene=self.scene, robot=self.robot)

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
        self.scene.visualisePose(self.reach.goalPos, self.reach.goalOrn)
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
        currentPos = robot.getArmJointAngles()
        # goalPos = []
        # for i in range(6):
        #     startPos = currentPos[i]
        #     goalPos = reach.goalJointAngles[i]
        #
        #     # dif = abs(startPos - potGoalPos)
        #     # potGoalPos2 = 0
        #     # if potGoalPos < 0:
        #     #     potGoalPos2 = potGoalPos + math.pi*2
        #     #     dif2 = abs(startPos - (potGoalPos + math.pi*2))
        #     # else:
        #     #     potGoalPos2 = potGoalPos - math.pi*2
        #     #     dif2 = abs(startPos - (potGoalPos - math.pi*2))
        #     # if dif2 < dif:
        #     #     goalPos.append(potGoalPos2)
        #     # else:
        #     #     goalPos.append(potGoalPos)
        difs = np.array(reach.goalJointAngles) - np.array(currentPos)
        armVel = robot.getArmJointVel()
        action = []
        for i in range(6):
            dif = difs[i]
            vel = armVel[i]
            if abs(dif) <= 0.0001:
                action.append(np.clip(-vel*10, -1, 1))
            else:
                if abs(dif) < 0.001:
                    maxMag = 0.001
                elif abs(dif) < 0.01:
                    maxMag = 0.01
                elif abs(dif) < 0.05:
                    maxMag = 0.1
                elif abs(dif) < 0.1:
                    maxMag = 0.2
                elif abs(dif) < 0.15:
                    maxMag = 0.3
                elif abs(dif) < 0.2:
                    maxMag = 0.4
                elif abs(dif) < 0.25:
                    maxMag = 0.5
                elif abs(dif) < 0.3:
                    maxMag = 0.6
                elif abs(dif) < 0.35:
                    maxMag = 0.7
                elif abs(dif) < 0.4:
                    maxMag = 0.8
                elif abs(dif) < 0.45:
                    maxMag = 0.9
                else:
                    maxMag = 1.0
                magError = maxMag - abs(vel)
                magCorr = np.clip(magError*10, -1, 1)
                if dif < 0:
                    action.append(-magCorr)
                else:
                    action.append(magCorr)
        action = np.array(action)
        return action, {'mean': action, 'log_std': np.array([0,0,0,0,0,0]), 'evaluation': action}

# ------------------------------
# Load trained agent
policy_file = 'policy_bc.pickle'
with open("logs/" + policy_file, 'rb') as pickle_file:
    policy = pickle.load(pickle_file)

# ------------------------------
# Evaluate Policies
time.sleep(5)
bc_pol_score = e.evaluate_policy(policy, num_episodes=50, mean_action=True)
expert_score = e.evaluate_policy(Expert(), num_episodes=50, mean_action=True)
print("Expert policy performance (eval mode) = %f" % expert_score[0][0])
print("BC policy performance (eval mode) = %f" % bc_pol_score[0][0])
