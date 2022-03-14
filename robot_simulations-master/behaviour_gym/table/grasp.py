import os
import random
import time
import math

import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import gym
from gym.utils import seeding

from behaviour_gym.table import environment


class Grasp(environment.Environment):

    def __init__(self, maxArmMove=0.01, startPosOffset=[0.0, 0.0, 0.18],
                 startPosRange=0.03, liftGoal=0.04, sparse=False,
                 liftThreshold=0.03, **kwargs):
        """
        Initialises grasp environment.

        Args:
            maxArmMove (float): largest position change in metres
            startPosOffset (float): offset added to block position to set arm's
                                    start position
            startPosRange (float): range in metres of random noise to add to the
                                   arm start position
            liftGoal (float): goal distance in metres to lift block
            sparse (boolean): whether or not to return a sparse reward
            liftThreshold (float): distance in metres from goal considered
                                   success
        """
        # Env variables
        self.maxArmMove = maxArmMove
        self.startPosOffset = startPosOffset
        self.startPosRange = startPosRange
        self.liftGoal = liftGoal
        self.sparse = sparse
        self.liftThreshold = liftThreshold

        super(Grasp, self).__init__(**kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _reset(self):
        # Get a random starting pose for the block
        blockInitialPos = [random.uniform(0.3, 1.0),
                           random.uniform(-0.42, 0.42),
                           0.635]
        # blockOrn = [0, 0, random.uniform(-math.pi/2, math.pi/2)]
        blockInitialOrn = [0, 0, 0]
        blockInitialOrn = self._p.getQuaternionFromEuler(blockInitialOrn)

        # Move arm to random start position near the block
        armInitalPos = np.add(blockInitialPos, self.startPosOffset)
        armPosOffset = np.random.uniform(-self.startPosRange,
                                         self.startPosRange,
                                         size=(3,))
        armStartPos = np.add(armInitalPos, armPosOffset)
        self.robot.applyArmPos(armStartPos)
        self.robot.applyGripRestPos()
        for i in range(180):
            self._p.stepSimulation()

        # Spawn block and let fall
        blockUrdfPath = os.path.join(self._urdfRoot, "block.urdf")
        self._blockId = self._p.loadURDF(blockUrdfPath, blockInitialPos,
                                         blockInitialOrn)
        self._p.setGravity(0, 0, -10)
        for i in range(20):
            self._p.stepSimulation()

        # Save block pose before episode starts
        self.blockStartPos, self.blockStartOrn = self._p.getBasePositionAndOrientation(self._blockId)

        return True


    def _setAction(self, action):
        # Get actions
        action = action.copy()
        armPosOffset = action[:3] * self.maxArmMove
        gripAction = action[3:]

        # Calculate goal arm pos
        armPos = self.robot.getArmPos()
        goalArmPos = np.add(armPos, armPosOffset)

        # Apply arm pos and gripper action
        self.robot.applyArmPos(goalArmPos)
        self.robot.applyGripAction(gripAction)


    def _getObs(self):
        robotObs = self.robot.getObs()
        blockPos, blockOrn = self._p.getBasePositionAndOrientation(self._blockId)
        blockLinVel, blockAngVel = self._p.getBaseVelocity(self._blockId)
        armPos = self.robot.getArmPos()
        relativePos = [blockPos[0] - armPos[0],
                       blockPos[1] - armPos[1],
                       blockPos[2] - armPos[2]]
        obs = np.concatenate((robotObs, blockPos, blockOrn, blockLinVel,
                              blockAngVel, relativePos, self.blockStartPos,
                              self.blockStartOrn))
        return obs


    def _getReward(self):
        if self.sparse:
            if self._isSuccess():
                reward = 1
            else:
                reward = 0
        else:
            # Reward for getting finger tips close to the block
            fingerDist = np.clip(self._getFingerDist(), 0.07, 0.12)
            fingerDist = (fingerDist-0.07) / (0.12-0.07)
            fingerReward = 1 - fingerDist**0.4

            # Reward for touching block with finger tips
            fingerContacts = self._getNumFingerTipContacts()
            numFingers = len(self.robot.getFingerTipLinks())
            contactReward = fingerContacts / numFingers

            # Reward for lifting block
            liftDist = np.clip(self._getLiftGoalDist(), 0, self.liftGoal)
            liftDist = liftDist / self.liftGoal
            liftReward = 1 - liftDist

            # Penalty for changing original orientation while lifting
            ornDist = np.clip(self._getOrientationDist(), 0, math.pi)
            ornDist = ornDist / math.pi
            ornPenalty = 1 - ornDist*0.8

            # Penalty for changing original x and y position while lifting by
            # more than 0.01 metre
            centeringDist = np.clip(self._getCenteringDist(), 0, 0.1)
            centeringDist = centeringDist / 0.1
            centeringPenalty = 1 - centeringDist*0.8

            reward = (fingerReward
                      + contactReward
                      + centeringPenalty*ornPenalty*liftReward)

        return reward


    def _isDone(self):
        if self.sparse and self._isSuccess():
            return True
        return False


    def _isSuccess(self):
        liftDist = self._getLiftGoalDist()
        if liftDist < self.liftThreshold:
            return True
        return False


    def _getNumObs(self):
        return self.robot.nObs + 23


    def _getNumActions(self):
        return 3 + self.robot.nGripActions


    def _getInfo(self):
        info = {"lift distance" : self._getLiftGoalDist(),
                "Finger Tip distance" : self._getFingerDist(),
                "Orientaton distance" : self._getOrientationDist(),
                "Centering distance" : self._getCenteringDist(),
                "num finger contacts" : self._getNumFingerTipContacts()}
        return info


    def _stepCallback(self):
        pass


    # Helper methods
    # --------------------------------------------------------------------------

    def _getCenteringDist(self):
        """
        Gets the distance of the block's X and Y positions from the goal
        position.
        """
        blockPos, _ = self._p.getBasePositionAndOrientation(self._blockId)
        return self._getDist(blockPos[:2], self.blockStartPos[:2])


    def _getLiftGoalDist(self):
        """Gets the distance of the block's Z position from the goal position"""
        blockPos, _ = self._p.getBasePositionAndOrientation(self._blockId)
        goalLiftPos = self.blockStartPos[2] + self.liftGoal
        return abs(goalLiftPos - blockPos[2])


    def _getOrientationDist(self):
        """Gets the distance between the current and goal orientation."""
        _, blockOrn = self._p.getBasePositionAndOrientation(self._blockId)
        return self._getQuaternionDist(blockOrn, self.blockStartOrn)


    def _getFingerDist(self):
        """Gets the distance from the finger tips to the block."""
        fingerPos = self.robot.getFingerTipPos()
        blockPos, _ = self._p.getBasePositionAndOrientation(self._blockId)
        dists = []
        for pos in fingerPos:
            dists.append(self._getDist(pos, blockPos))
        return np.mean(dists)


    def _getNumFingerTipContacts(self):
        """Get the number of finger tips in contact with the block."""
        contactPointsBlock = self._p.getContactPoints(self.robot.robotId,
                                                      self._blockId)
        fingerTips = self.robot.getFingerTipLinks()
        contacts = []
        for contactPoint in contactPointsBlock:
            if contactPoint[3] in fingerTips:
                contacts.append(contactPoint[3])

        numUniqueContacts = len(set(contacts))
        return numUniqueContacts
