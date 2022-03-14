import random
import math

import numpy as np

from behaviour_gym.primitive import primitiveNoIk
from behaviour_gym.utils import quaternion as q


class ReachBasicNoIk(primitiveNoIk.PrimitiveNoIk):

    def __init__(self, maxAction=0.1,
                 posRange=[[0.4,0.95],[-0.4,0.4],[0.65,1.0]],
                 ornRange=None, collisionPenalty=1, sparse=False,
                 distanceThreshold=0.05, *args, **kwargs):
        """Initialises reach primitive.

        Args:
            goalObject (string): name of the object in the scene to reach for.
            goalPosOffset ([float]): offset applied to objects position to
                                     calculate goal position of the robot.
                                     [X,Y,Z] in metres.
            maxMove (float): largest position change along each axes in metres.
            posRange ([float]): range for spawning and limiting end-effector
                                position.
            collisionPenalty (float): how much to remove from reward when a
                                      collision occurs.
            sparse (boolean): whether or not to return a sparse reward
            distanceThreshold (float): error that position and orientation
                                       accuracy must be higher than to be
                                       considered successful.

        """
        # Env variables
        self.maxAction = maxAction
        self.posRange = posRange
        self.ornRange = ornRange
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.distanceThreshold = distanceThreshold

        # Info
        self._collisions = 0.0

        super(ReachBasicNoIk, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Generate goal pose by sampling a random joint configuration
        randPositions = np.random.uniform(-math.pi*2, math.pi*2, (6,))
        self.goalJointAngles = randPositions
        self.robot.resetArmJoints(randPositions)
        self.goalPos, self.goalOrn = self.robot.getPose()

        # Start in a random joint configuration
        startPositions = np.random.uniform(-math.pi*2, math.pi*2, (6,))
        self.startJointAngles = startPositions
        self.robot.resetArmJoints(startPositions)
        self.robot.setArmMotorsVel([0,0,0,0,0,0])

        # offset = np.random.uniform(-0.3, 0.3, (6,))
        # startPositions = randPositions + offset
        # # startPositions = np.random.uniform(-math.pi*2, math.pi*2, (6,))
        # self.robot.resetArmJoints(startPositions)
        # self.robot.setArmMotorsVel([0,0,0,0,0,0])

        return True


    def _applyAction(self, action):
        # Copy and rescale action
        action = action.copy()
        action = action * self.maxAction

        # Combine current velocities with action vector
        currVel = self.robot.getArmJointVel()
        goalVel = np.add(currVel, action)

        # Update robot motors
        self.robot.setArmMotorsVel(goalVel)


    def _getObs(self):
        # Concatenate and return observation
        obs = np.concatenate((self.goalPos, self.goalOrn, [self.stepCount]))
        return obs


    def _getReward(self, action):
        # Reward for being in a successful state
        successReward = 0
        if self._isSuccess():
            successReward = 1

        # Penalty for collision
        collisionPenalty = 0
        if self._isContact():
            collisionPenalty = self.collisionPenalty

        if self.sparse:
            # If sparse then return sparse rewards for success and collision
            reward = successReward - collisionPenalty
            return reward
        else:
            # Reward accuracy of pose
            armPos, armOrn = self.robot.getPose()
            posDist = self._getDist(armPos, self.goalPos)
            ornDist = q.distance(armOrn, self.goalOrn)
            error = posDist + ornDist
            accuracyReward = 10 * math.exp(-error)
            # accuracyReward = (10 * math.exp(-posDist)) + (10 * math.exp(-ornDist))

            # Penalise mean action magnitude
            actionPenalty = 0.05 * (np.linalg.norm(action, ord=1)/len(action))

            # Return dense reward
            return accuracyReward


    def _isDone(self):
        if self.sparse and self._isSuccess():
            return True
        if self.stepCount >= 100:
            return True
        return False


    def _isSuccess(self):
        armPos, armOrn = self.robot.getPose()
        posDist = self._getDist(armPos, self.goalPos)
        ornDist = q.distance(armOrn, self.goalOrn)

        if posDist < self.distanceThreshold and ornDist < self.distanceThreshold and not self._isContact():
            return True
        return False


    def _getNumObs(self):
        return 33


    def _getNumActions(self):
        return 6


    def _getInfo(self):
        armPos, armOrn = self.robot.getPose()
        posDist = self._getDist(armPos, self.goalPos)
        ornDist = q.distance(armOrn, self.goalOrn)

        info = {"collisions" : self._collisions,
                "position distance" : posDist,
                "orientation distance" : ornDist}
        return info


    def _stepCallback(self):
        # Track collisions
        if self._isContact():
            self._collisions += 1
        self.stepCount += 1

        # if not self.lastObs is None:
        #     prevArmPos = self.lastObs[0:3]
        #     prevArmOrn = self.lastObs[3:7]
        #
        #
        # if not self.lastPrevObs is None:
        #     prevArmPos = self.lastPrevObs[0:3]
        #     prevArmOrn = self.lastPrevObs[3:7]
        #
        #
        #
        #     # Calculate distances in metres
        #     action = self.lastAction
        #     xDist = action[0]*self.maxMove
        #     yDist = action[1]*self.maxMove
        #     zDist = action[2]*self.maxMove
        #
        #     # Calculate rotation in radians
        #     roll = action[3]*self.maxRot
        #     yaw = action[4]*self.maxRot
        #     pitch = action[5]*self.maxRot
        #
        #     # Calculate the goal pose in the world frame
        #     goalPos = [prevArmPos[0] + xDist,
        #                prevArmPos[1] + yDist,
        #                prevArmPos[2] + zDist]
        #     goalOrn = q.rotateGlobal(prevArmOrn, roll, yaw, pitch)
        #
        #     actionPosError = self._getDist(armPos, goalPos)
        #     actionOrnError = q.distance(armOrn, goalOrn)
        #     if actionPosError > 0.01:
        #         print("BAD ACTION - pos error - ", actionPosError)
        #     if actionOrnError > 0.01:
        #         print("BAD ACTION - orn error - ", actionOrnError)
        #
        #
        #     posDist = self.lastInfo["position distance"]
        #     ornDist = self.lastInfo["orientation distance"]
        #     self.prevError = posDist + ornDist


    def _resetCallback(self):
        if self._isContact():
            # Reset if starting in contact state
            self.reset()
        else:
            # If a successful reset then initialise error tracker
            armPos, armOrn = self.robot.getPose()
            posDist = self._getDist(armPos, self.goalPos)
            ornDist = q.distance(armOrn, self.goalOrn)
            self.prevError = posDist + ornDist

            # Initialise info
            self._collisions = 0
            self.stepCount = 0


    # Helper methods
    # --------------------------------------------------------------------------

    def _getGoalDist(self):
        """Gets the distance between the robot's current and goal poses."""
        armPos, armOrn = self.robot.getPose()
        objPos, _ = self.scene.getPose(self.goalObj)
        goalPos = np.add(objPos, self.goalPosOffset)
        posDist = self._getDist(armPos, self.goalPos)

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
