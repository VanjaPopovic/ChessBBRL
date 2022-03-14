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
from stockfish import Stockfish
import chess
from Chessnut import Game
import time as timer
import pickle
import gym
from gym.envs.registration import register


from environment import PickAndPlace
from pick_place import PickPlaceScene
from utils import *
SEED = 500

class Env(gym.Env):
    def __init__(self):

        self.scene = PickPlaceScene(guiMode=True)
        # self.scene = Basic(guiMode=True)
        self.robot = self.scene.loadRobot()
        # self.robot.resetBasePose(pos=[0, 0, 0.6])
        # self.scene.setInit()
        self.stockfish = Stockfish(r'C:\Users\fream\Desktop\PHD\robot_simulations-master\chess_env\stockfish2.exe')

        self.reach = PickAndPlace(self.scene, self.robot, step_fn_expert_behaviours)
        # Gym metadata and spaces
        self.metadata = {'render.modes': ['human', 'rgbd_array']}
        nObs = self.reach._get_obs_shape()
        self.observation_space = gym.spaces.Box(np.float32(-np.inf),
                                                np.float32(np.inf),
                                                shape=(nObs,),
                                                dtype=np.float32)
        nActions = self.reach._get_num_actions()
        self.action_space = gym.spaces.Box(np.float32(-1), np.float32(1),
                                           shape=(nActions,),
                                           dtype=np.float32)
        self.stockfish.set_fen_position(self.scene.current_fen_string)
        self.board = chess.Board(self.scene.current_fen_string)
        print(self.stockfish.get_board_visual())


        self.move = self.stockfish.get_best_move()
        self.chessMove = chess.Move.from_uci(self.move)

    def seed(self, seed=None):
        return self.reach.seed(seed)

    def reset(self):
        obs = self.reach.reset()
        # self.scene.visualisePose(self.reach.goalPos, self.reach.goalOrn)
        return obs

    def step(self, action):
        return self.reach.step(action)

    def render(self, mode="rgbd_array"):
        return self.reach.render(mode)

    def close(self):
        return self.reach.close()

    def get_obs(self):
        return self.reach.getObs()

register(
    id='myenv-v0',
    entry_point=Env,
    max_episode_steps=100
)

e = GymEnv("myenv-v0")
robot = e.env.robot
reach = e.env.reach
scene = e.env.scene

class Expert:
    def get_action(self, observation):
        # Get the objects pose from the scene

        
        # goalObj = "A1"
        # goalOffset = [0.0, 0.0, 0.04]
        # maxMove = 0.08
        # objPos, _ = scene.getPose(goalObj)

        # # Add goalOffset to find goal position
        # goalPos = reach.reachPos
        # print("Reach Goal")
        # print(goalPos)
    
    
        # # Get the robot's pose
        # armPos, armOrn = robot.getPose()
        # print("ARM POS")
        # print(armPos)
        # # Find the difference between the robot and goal pose
        # dif = np.subtract(goalPos, armPos)
        # print("dif")
        # print(dif)
        # # Clip difference between max move
        # clampedDif = np.clip(dif, -maxMove, maxMove)

        # # Rescale from [-0.05, 0.05] to [-1, 1]
        # action = 2 * ((clampedDif + 0.05) / (0.1)) - 1

        # action = [action[0], action[1], action[2], 0, 0, 0]
        # print(action)
        # action = np.array(action)
        # return action, {'mean': action, 'log_std': np.array([0, 0, 0, 0, 0, 0]), 'evaluation': action}
        robPos, robOrn = robot.getPose()
        blockPos = scene.getTarget()
        goalPos = np.add(blockPos, [0, 0, 0.3])

        dif = np.subtract(goalPos, robPos)
        clampedDif = np.clip(dif, -0.08, 0.08)

        # Rescale from [-0.05, 0.05] to [-1, 1]
        action = 2 * ((clampedDif + 0.05) / (0.1)) - 1
        print("Approaching")
        print(blockPos)
        print(robPos)
        action = [goalPos[0], goalPos[1], goalPos[2], 0, 0, 0]
        print(action)
        action = np.array(action)
        return action, {'mean': action, 'log_std': np.array([0, 0, 0, 0, 0, 0]), 'evaluation': action}

# ------------------------------
# Get demonstrations
print("========================================")
print("Collecting expert demonstrations")
print("========================================")

demo_paths = sample_paths(num_traj=100, policy=Expert(), env=e)
expert_demo_file = 'expert_demos.pickle'
pickle.dump(demo_paths, open('logs/reachWithObject/' + expert_demo_file, 'wb'))
#with open("logs/reachWithObject/" + expert_demo_file, 'rb') as pickle_file:
    #demo_paths = pickle.load(pickle_file)

# ------------------------------
# Train BC
policy = MLP(e.spec, hidden_sizes=(400,300), seed=SEED)
bc_agent = BC(demo_paths, policy=policy, epochs=50, batch_size=64, lr=1e-4) # will use Adam by default
ts = timer.time()
print("========================================")
print("Running BC with expert demonstrations")
print("========================================")
bc_agent.train()
print("========================================")
print("BC training complete !!!")
print("time taken = %f" % (timer.time()-ts))
print("========================================")

# ------------------------------
# Evaluate Policies
bc_pol_score = e.evaluate_policy(policy, num_episodes=50, mean_action=True)
expert_score = e.evaluate_policy(Expert(), num_episodes=50, mean_action=True)
print("Expert policy performance (eval mode) = %f" % expert_score[0][0])
print("BC policy performance (eval mode) = %f" % bc_pol_score[0][0])

bc_agent.logger.save_log('logs/reachWithObject')
make_train_plots(log=bc_agent.logger.log, keys=["loss"], save_loc='logs/reachWithObject')
policy_file = 'policy_bc.pickle'
pickle.dump(bc_agent.policy, open('logs/reachWithObject' + policy_file, 'wb'))
