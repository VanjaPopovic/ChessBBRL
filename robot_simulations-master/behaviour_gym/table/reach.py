import os
import random
import math

import pybullet_utils.bullet_client as bc
import numpy as np

from behaviour_gym.table import environment


class Reach(environment.Environment):

    def __init__(self, maxArmMove=0.05, goalPosOffset=[0.0, 0.0, 0.18],
                 collisionPenalty=0.05, sparse=False, distanceThreshold=0.05,
                 **kwargs):
        """
        Initialises reach environment.

        Args:
            maxArmMove (float): largest position change in metres
            goalPosOffset (float): added to block position to calculate the
                                   goal position
            collisionPenalty (float): how much to remove from reward when a
                                      collision occurs
            sparse (boolean): whether or not to return a sparse reward
            distanceThreshold (float): distance from goal considered success
        """
        # Env variables
        self.maxArmMove = maxArmMove
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.goalPosOffset = goalPosOffset
        self.distanceThreshold = distanceThreshold
        self.collisions = 0.0

        super(Reach, self).__init__(**kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _reset(self):
        # Move arm to random start position
        armStartPos = [random.uniform(0.3, 1.0),
                       random.uniform(-0.42, 0.42),
                       random.uniform(0.9, 1.3)]
        self.robot.applyArmPos(armStartPos)
        self.robot.applyGripRestPos()
        for i in range(180):
            self._p.stepSimulation()

        # Spawn random goal position at least 10cm away from starting position
        armGoalDist = 0.0
        while armGoalDist < 0.1:
            self.blockStartX = random.uniform(0.3, 0.95)
            self.blockStartY = random.uniform(-0.42, 0.42)
            blockPos = [self.blockStartX, self.blockStartY, 0.635]
            goalArmPos = np.add(blockPos, self.goalPosOffset)
            armGoalDist = self._getDist(armStartPos, goalArmPos)
        blockOrn = [0, 0, random.uniform(-math.pi/2, math.pi/2)]
        blockOrn = self._p.getQuaternionFromEuler(blockOrn)
        blockUrdfPath = os.path.join(self._urdfRoot, "block.urdf")
        self._blockId = self._p.loadURDF(blockUrdfPath, blockPos, blockOrn)

        # Let block drop
        self._p.setGravity(0, 0, -10)
        for i in range(20):
            self._p.stepSimulation()

        # Reset info
        self.collisions = 0

        return True


    def _setAction(self, action):
        action = action.copy()

        # Calculate goal arm pos
        armPos = self.robot.getArmPos()
        goalArmPos = [armPos[0] + action[0]*self.maxArmMove,
                      armPos[1] + action[1]*self.maxArmMove,
                      armPos[2] + action[2]*self.maxArmMove]

        # Apply arm and gripper pos
        self.robot.applyArmPos(goalArmPos)
        self.robot.applyGripRestPos()


    def _getObs(self):
        robotObs = self.robot.getObs()
        blockPos, blockOrn = self._p.getBasePositionAndOrientation(self._blockId)
        blockLinVel, blockAngVel = self._p.getBaseVelocity(self._blockId)
        armPos = self.robot.getArmPos()
        relativePos = [blockPos[0] - armPos[0],
                       blockPos[1] - armPos[1],
                       blockPos[2] - armPos[2]]
        obs = np.concatenate((robotObs, blockPos, blockOrn, blockLinVel,
                              blockAngVel, relativePos))
        return obs


    def _getReward(self):
        if self.sparse:
            if self._isSuccess():
                reward = 1
            else:
                reward = 0
        else:
            dist = self._getGoalDist()
            reward = 1 - np.clip(dist, 0, 1)**0.4

            # Penalise contact
            if self._isContact():
                reward -= self.collisionPenalty

        return reward


    def _isDone(self):
        if self.sparse and self._isSuccess():
            return True
        return False


    def _isSuccess(self):
        goalDist = self._getGoalDist()
        if goalDist < self.distanceThreshold and not self._isContact():
            return True
        return False


    def _getNumObs(self):
        return self.robot.nObs + 16


    def _getNumActions(self):
        return 3


    def _getInfo(self):
        info = {"collisions" : self.collisions,
                "goal distance" : self._getGoalDist()}
        return info


    def _stepCallback(self):
        if self._isContact():
            self.collisions += 1


    # Helper methods
    # --------------------------------------------------------------------------

    def _getGoalDist(self):
        currArmPos = self.robot.getArmPos()
        currBlockPos = self._p.getBasePositionAndOrientation(self._blockId)[0]
        goalArmPos = np.add(currBlockPos, self.goalPosOffset)
        return self._getDist(currArmPos, goalArmPos)
