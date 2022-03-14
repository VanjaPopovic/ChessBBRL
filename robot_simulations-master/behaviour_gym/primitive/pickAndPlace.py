import math
import random

import cv2
import gym
from gym.utils import seeding
import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t
from behaviour_gym.scene import Table


class PickAndPlace(gym.Env):

    def __init__(self, guiMode=True):
        self.scene = Table(guiMode=guiMode)
        self.p = self.scene.getPhysicsClient()
        self.robot = self.scene.loadRobot("ur103f", basePos=[0,0,0.6])
        self.scene.setInit()

        # Gym metadata and spaces
        self.metadata = {'render.modes': ['human', 'rgbd_array']}
        self.observation_space = gym.spaces.Box(np.float32(-np.inf),
                                                np.float32(np.inf),
                                                shape=(19,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

        self.seed()


    def seed(self, seed=None):
        """Seeds the random number generator with a strong seed.

        Note that if seed=None then an OS specific source of randomness is used.

        Args:
            seed (int): a positive int value or None.

        Returns:
            the seed in an array

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.scene.reset(random=True)

        armStartPos = [random.uniform(0.4, 0.95),
                       random.uniform(-0.4, 0.4),
                       random.uniform(0.9, 1.1)]
        armStartOrn = q.random()

        self.robot.reset(armStartPos, armStartOrn)
        self.scene.start()

        blockPos, _ = self.scene.getPose("block")
        dif = 0
        while dif < 0.01:
            self.goalPlacePos = [random.uniform(0.4, 0.95),
                                 random.uniform(-0.4, 0.4),
                                 0.64]
            dif = np.linalg.norm(blockPos-self.goalPlacePos)

        self.stepCounter = 0

        return self.getObs()


    def step(self, action):
        self.stepCounter += 1

        if action == 0:
            # Reach
            robPos, robOrn = self.robot.getPose()
            goalOrn = self.robot.restOrn
            blockPos, _ = self.scene.getPose("block")

            if blockPos[2] > 0.67:
                robPos = np.array(robPos)
                goalPos = np.add(self.goalPlacePos, [0,0,0.02])
                if np.linalg.norm(robPos-goalPos) < 0.02:
                    return self.getObs(), self.getReward(), self.getDone(), self.getInfo()
                poses = t.interpolate(robPos, robOrn, goalPos, robOrn, 30)
                for pose in poses:
                    self.robot.applyPose(*pose, relative=True)
                    self.scene.step(20)
                self.robot.applyPose(goalPos, robOrn)
                self.scene.step(50)
            else:
                robPos = np.array(robPos)
                if np.linalg.norm(robPos-blockPos) < 0.02:
                    return self.getObs(), self.getReward(), self.getDone(), self.getInfo()

                goalPos = np.array(robPos).copy()
                goalPos[2] = 1.0
                poses = t.interpolate(robPos, robOrn, goalPos, robOrn, 5)
                for pose in poses:
                    self.robot.applyPose(*pose, relative=True)
                    self.scene.step(20)
                self.robot.applyPose(goalPos, robOrn)
                self.scene.step(50)

                robPos, robOrn = self.robot.getPose()
                blockPos, _ = self.scene.getPose("block")
                goalPos = np.add(blockPos, [0,0,0.1])
                poses = t.interpolate(robPos, robOrn, goalPos, goalOrn, 30)
                for pose in poses:
                    self.robot.applyPose(*pose, relative=True)
                    self.scene.step(20)
                self.robot.applyPose(goalPos, goalOrn, relative=True)
                self.scene.step(50)
                self.robot.applyPose(goalPos, goalOrn, relative=True)
                self.scene.step(50)

                robPos, robOrn = self.robot.getPose()
                blockPos, _ = self.scene.getPose("block")
                goalPos = np.add(blockPos, [0,0,0.01])
                poses = t.interpolate(robPos, robOrn, goalPos, goalOrn, 4)
                self.robot.applyPose(goalPos, goalOrn, relative=True)
                for pose in poses:
                    self.robot.applyPose(*pose, relative=True)
                    self.scene.step(20)
                self.robot.applyPose(goalPos, goalOrn, relative=True)
                self.scene.step(50)

        elif action == 1:
            # Grasp
            robPos, robOrn = self.robot.getPose()
            blockPos, _ = self.scene.getPose("block")
            robPos = np.array(robPos)
            blockPos = np.array(blockPos)
            if self.robot.getGripState()[0] > 0.6:
                return self.getObs(), self.getReward(), self.getDone(), self.getInfo()
            if np.linalg.norm(robPos-blockPos) > 0.02:
                return self.getObs(), self.getReward(), self.getDone(), self.getInfo()

            self.robot.applyGripAction([1])
            self.scene.step(20)
            self.robot.applyGripAction([1])
            self.scene.step(20)
            self.robot.applyGripAction([1])
            self.scene.step(20)
            self.robot.applyGripAction([1])
            self.scene.step(20)
            self.robot.applyGripAction([1])
            self.scene.step(20)
            self.robot.applyGripAction([1])
            self.scene.step(20)
            self.robot.applyGripAction([.5])
            self.scene.step(20)

            robPos, robOrn = self.robot.getPose()
            goalPos = np.add(robPos, [0,0,0.05])
            self.robot.applyPose(goalPos, robOrn, relative=True)
            self.scene.step(50)
        else:
            # Place
            robPos, _ = self.robot.getPose()
            robPos = np.array(robPos)
            if self.robot.getGripState()[0] < 0.6:
                return self.getObs(), self.getReward(), self.getDone(), self.getInfo()
            goalPos = np.add(self.goalPlacePos, [0,0,0.02])
            if np.linalg.norm(robPos-goalPos) > 0.02:
                return self.getObs(), self.getReward(), self.getDone(), self.getInfo()

            self.robot.applyGripAction([-1])
            self.scene.step(20)
            self.robot.applyGripAction([-1])
            self.scene.step(20)
            self.robot.applyGripAction([-1])
            self.scene.step(20)
            self.robot.applyGripAction([-1])
            self.scene.step(20)
            self.robot.applyGripAction([-1])
            self.scene.step(20)
            self.robot.applyGripAction([-1])
            self.scene.step(20)
            self.robot.applyGripAction([-.5])
            self.scene.step(20)

        return self.getObs(), self.getReward(), self.getDone(), self.getInfo()


    def render(self, mode="human"):
        pass


    def close(self):
        pass


    def getReward(self):
        robPos, _ = self.robot.getPose()
        robPos = np.array(robPos)
        blockPos, _ = self.scene.getPose("block")
        if np.linalg.norm(blockPos-self.goalPlacePos) < 0.01:
            return 1
        else:
            return 0


    def getDone(self):
        if self.getReward() > 0:
            return True
        elif self.stepCounter > 50:
            return True
        return False


    def getInfo(self):
        if self.getReward() > 0:
            return {"is_success": True}
        return {"is_success": False}


    def getObs(self):
        robPos, robOrn = self.robot.getPose()
        blockPos, blockOrn = self.scene.getPose("block")
        gripState = self.robot.getGripState()
        return np.concatenate((robPos, robOrn, blockPos, blockOrn, gripState,
                               self.goalPlacePos, [self.stepCounter])).flatten()
