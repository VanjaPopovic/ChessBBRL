import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q


class Reach(primitive.Primitive):

    def __init__(self, goalObject, goalPosOffset=[0,0,0.0], maxMove=0.01,
                 maxRotation=math.pi/20,
                 posRange=[[0.3,1.0],[-0.42,0.42],[0.9,1.3]],
                 ornRange=[[-math.pi/2, math.pi/2], [-math.pi/2, math.pi/2],
                           [-math.pi/2, math.pi/2]],
                 collisionPenalty=0.02, sparse=False, distanceThreshold=0.05,
                 *args, **kwargs):
        """Initialises reach primitive.

        Args:
            goalObject (string): name of the object in the scene to reach for.
            goalPosOffset ([float]): offset applied to objects position to
                                     calculate goal position of the robot.
                                     [X,Y,Z] in metres.
            maxMove (float): largest position change along each axes in metres.
            posRange ([float]): range for sampling starting positions
                                     [[xMin,xMax],[yMin,yMax],[zMin,zMax]].
            collisionPenalty (float): how much to remove from reward when a
                                      collision occurs
            sparse (boolean): whether or not to return a sparse reward
            distanceThreshold (float): distance in metres from goal considered
                                       success

        """
        # Env variables
        self.goalObj = goalObject
        self.goalPosOffset = goalPosOffset
        self.maxMove = maxMove
        self.maxRot = maxRotation
        self.posRange = posRange
        self.ornRange = ornRange
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.goalPosOffset = goalPosOffset
        self.distanceThreshold = distanceThreshold

        self._collisions = 0.0

        super(Reach, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get random starting position
        armStartPos = [random.uniform(self.posRange[0][0],
                                      self.posRange[0][1]),
                       random.uniform(self.posRange[1][0],
                                      self.posRange[1][1]),
                       random.uniform(self.posRange[2][0],
                                      self.posRange[2][1])]

        # Offset arm starting rotation
        # armRestOrn = self.robot.armRestOrn
        # rotationOffset = [random.uniform(self.ornRange[0][0],
        #                                  self.ornRange[0][1]),
        #                   random.uniform(self.ornRange[1][0],
        #                                  self.ornRange[1][1]),
        #                   random.uniform(self.ornRange[2][0],
        #                                  self.ornRange[2][1])]
        # armStartOrn = q.rotateQuaternion(armRestOrn, rotationOffset[0],
        #                                  rotationOffset[1], rotationOffset[2])
        #
        # # Instantly reset joint state and update motors
        # self.robot.reset(armStartPos, armStartOrn, self.robot.gripRestPos)
        #self.robot.reset(armStartPos)

        # Reset info
        self._collisions = 0

        return True


    def _getAction(self, action):
        action = action.copy()
        print(action)
        # Calculate distances in metres
        xDist = action[0]*self.maxMove
        yDist = action[1]*self.maxMove
        zDist = action[2]*self.maxMove

        # Calculate rotation in radians
        roll = action[3]*self.maxRot
        yaw = action[4]*self.maxRot
        pitch = action[5]*self.maxRot

        # Get robot's relative position and orienation
        pos, orn = self.robot.getPose()

        # Calculate the goal pose in the base frame
        goalPosBase = [pos[0] + xDist,
                       pos[1] + yDist,
                       pos[2] + zDist]
        goalOrnBase = q.rotateGlobal(orn, roll, yaw, pitch)

        # Don't interact with the gripper
        gripAction = [0.00]

        return goalPosBase, goalOrnBase, gripAction



    def _getObs(self):
        # Get object state
        objPos, objOrn = self.scene.getPose(self.goalObj)
        objLinVel, objAngVel = self.scene.getVelocity(self.goalObj)

        # Get robot observation vector
        pos, orn = self.robot.getPose()

        # Get relative distance between robot and object
        armPos, _ = self.robot.getPose()
        relativePos = [objPos[0] - pos[0],
                       objPos[1] - pos[1],
                       objPos[2] - pos[2]]

        obs = np.concatenate((objPos, objOrn, objLinVel, objAngVel,
                              relativePos))
        return obs


    def _getReward(self, action):
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
        if goalDist < self.distanceThreshold:
            return True
        return False


    def _getNumObs(self):
        return self.robot.nObs + 16


    def _getNumActions(self):
        return 6


    def _getInfo(self):
        info = {"collisions" : self._collisions,
                "goal distance" : self._getGoalDist()}
        return info


    def _stepCallback(self):
        if self._isContact():
            self._collisions += 1
        print(self._getInfo())


    # Helper methods
    # --------------------------------------------------------------------------

    def _getGoalDist(self):
        """Gets the distance between the robot's current and goal poses."""
        armPos, _ = self.robot.getPose()
        objPos, _ = self.scene.getPose(self.goalObj)
        goalPos = np.add(objPos, self.goalPosOffset)
        return self._getDist(armPos, goalPos)


    def _isContact(self):
        """Check if contact made with the robot.

        Ignores links with id <1 which are usually part of the base.

        Returns:
            True if contact, false otherwise.

        """
        # Get contact points
        contactPoints = self.p.getContactPoints(self.robot.id)

        # Check if contact with any robot link that has an id higher than 0
        for point in contactPoints:
            if point[3] > 0:
                return True
        return False
