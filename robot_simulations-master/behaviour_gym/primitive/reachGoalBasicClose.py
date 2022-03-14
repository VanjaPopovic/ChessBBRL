import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q


class ReachGoalBasicClose(primitive.Primitive):

    def __init__(self, maxMove=0.01, maxRotation=math.pi/180,
                 posRange=[[0.35,1.0],[-0.47,0.47],[0.65,1.1]],
                 ornRange=None, collisionPenalty=-1, sparse=False,
                 distanceThreshold=0.01, *args, **kwargs):
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
            distanceThreshold (float): distance in metres from goal considered
                                       success

        """
        # Env variables
        self.maxMove = maxMove
        self.maxRot = maxRotation
        self.posRange = posRange
        self.ornRange = ornRange
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.distanceThreshold = distanceThreshold

        # Info
        self._collisions = 0.0

        super(ReachGoalBasicClose, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        restPos, restOrn, restGrip = self.robot.getRest()
        reachAble = False
        # Generate random reachable goal
        while not reachAble:
            self.goalPos = [random.uniform(self.posRange[0][0],
                                           self.posRange[0][1]),
                            random.uniform(self.posRange[1][0],
                                           self.posRange[1][1]),
                            random.uniform(self.posRange[2][0],
                                           self.posRange[2][1])]
            self.goalOrn = q.random()
            self.robot.reset(self.goalPos, self.goalOrn, restGrip)
            armPos, armOrn = self.robot.getPose()
            if self._getDist(armPos, self.goalPos) < 0.01:
                if q.distance(armOrn, self.goalOrn) < 0.01:
                    reachAble = True

        reachAble = False
        # Start robot arm in random pose
        while not reachAble:
            armStartPosOffset = np.random.uniform(-0.1, 0.1, (3))
            armStartPos = np.array(self.goalPos) + armStartPosOffset
            armStartOrnOffset = np.random.uniform(-0.2, 0.2, (3))
            armStartOrn = q.rotateGlobal(self.goalOrn, *armStartOrnOffset)
            self.robot.reset(armStartPos, armStartOrn, restGrip)
            armPos, armOrn = self.robot.getPose()
            if self._getDist(armPos, armStartPos) < 0.01:
                if q.distance(armOrn, armStartOrn) < 0.01:
                    reachAble = True

        # Initialise error trackers for reward calculation
        posDist = self._getDist(armPos, self.goalPos)
        ornDist = q.distance(armOrn, self.goalOrn)
        self.prevError = posDist + ornDist
        self.currError = posDist + ornDist

        # Reset info
        self._collisions = 0

        return True


    def _getAction(self, action):
        action = action.copy()

        # Calculate distances in metres
        xDist = action[0]*self.maxMove
        yDist = action[1]*self.maxMove
        zDist = action[2]*self.maxMove

        # Calculate rotation in radians
        roll = action[3]*self.maxRot
        yaw = action[4]*self.maxRot
        pitch = action[5]*self.maxRot

        # Calculate the goal pose in the world frame
        pos, orn = self.robot.getPose()
        goalPos = [pos[0] + xDist,
                   pos[1] + yDist,
                   pos[2] + zDist]
        orn = q.rotateGlobal(orn, roll, yaw, pitch)

        # Clip goal position within posRange
        # minPos = [self.posRange[0][0], self.posRange[1][0], self.posRange[2][0]]
        # maxPos = [self.posRange[0][1], self.posRange[1][1], self.posRange[2][1]]
        # goalPos = np.clip(goalPos, minPos, maxPos)

        # Don't interact with the gripper
        gripAction = None

        return goalPos, orn, gripAction


    def _getObs(self):
        # Get object state in world frame
        # objPosWorld, objOrnWorld = self.scene.getPose(self.goalObj)
        # objLinVelWorld, objAngVelWorld = self.scene.getVelocity(self.goalObj)

        # Get relative distance between robot and object
        pos, orn = self.robot.getPose()
        relativePos = [self.goalPos[0] - pos[0],
                       self.goalPos[1] - pos[1],
                       self.goalPos[2] - pos[2]]
        relativeOrn = self.p.getDifferenceQuaternion(orn, self.goalOrn)

        obs = np.concatenate((self.goalPos, self.goalOrn, relativePos,
                              relativeOrn))
        return obs


    def _getReward(self):
        successReward = 0
        if self._isSuccess():
            successReward = 1

        if self.sparse:
            reward = successReward
        else:
            armPos, armOrn = self.robot.getPose()
            posDist = self._getDist(armPos, self.goalPos)
            ornDist = q.distance(armOrn, self.goalOrn)
            error = posDist + ornDist

            reward = math.exp(25*-error)
            # if error < self.prevError:
            #     reward = 1 * self.prevError/error
            #
            #
            # if error > 0:
            #     gainReward = self.prevError/error
            # else:
            #     gainReward = 10
            #
            # reward = (self.prevError/error) * accuracyReward
            # if error < self.prevError:
            #     reward = accuracyReward
            # else:
            #     reward = -1
            # reward = (self.prevError/error) * accuracyReward

            # reward = accuracyReward
            # reward = accuracyReward + gainReward



            # rewardPos = 1 - np.clip(posDist, 0, 1)**0.4
            # rewardOrn = 1 - np.clip(ornDist, 0, 1)**0.4
            # reward = rewardPos + rewardOrn
            #
            # if posDist < 0.1 and ornDist < 0.1:
            #     reward += 10 - (posDist*100)
            #     reward += 10 - (ornDist*100)

            # Penalise contact
            if self._isContact():
                reward = self.collisionPenalty

        return reward


    def _isDone(self):
        if self.sparse and self._isSuccess():
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
        return self.robot.nObs + 14


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
        if self._isContact():
            self._collisions += 1
        armPos, armOrn = self.robot.getPose()
        posDist = self.getManhattanDist(armPos, self.goalPos)
        ornDist = q.distance(armOrn, self.goalOrn)
        self.prevError = posDist + ornDist


    def _resetCallback(self):
        if self._isContact():
            self.reset()


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
