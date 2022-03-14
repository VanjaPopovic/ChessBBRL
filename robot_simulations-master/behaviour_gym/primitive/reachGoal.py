import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q


import time

class ReachGoal(primitive.Primitive):

    def __init__(self, goalObj, maxMove=0.05, maxRotation=math.pi/20,
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
        self.goalObj = goalObj
        self.maxMove = maxMove
        self.maxRot = maxRotation
        self.posRange = posRange
        self.ornRange = ornRange
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.distanceThreshold = distanceThreshold

        # Info
        self._collisions = 0.0

        super(ReachGoal, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        restPos, restOrn, restGrip = self.robot.getRest()

        # Generate goal pose
        objPos, objOrn = self.scene.getPose(self.goalObj)
        self.goalPos = np.add(objPos, [0,0,0.01])
        objOrnEuler = self.p.getEulerFromQuaternion(objOrn)
        self.goalOrn = q.rotateGlobal(restOrn, 0, 0, objOrnEuler[2] + math.pi/2)

        # If goal pose not reachable then reset the environment again
        self.robot.reset(self.goalPos, self.goalOrn, restGrip)
        armPos, armOrn = self.robot.getPose()
        if self._getDist(armPos, self.goalPos) > 0.01:
            print("goal posdist: ", self._getDist(armPos, self.goalPos))
            time.sleep(1)
            return False
        if q.distance(armOrn, self.goalOrn) > 0.01:
            return False
            print("goal orndist: ", q.distance(armOrn, self.goalOrn))
            time.sleep(1)

        # Start robot arm in a random pose
        # reachAble = False
        # while not reachAble:
        #     # Generate random starting position
        #     armStartPos = [random.uniform(self.posRange[0][0],
        #                                   self.posRange[0][1]),
        #                    random.uniform(self.posRange[1][0],
        #                                   self.posRange[1][1]),
        #                    random.uniform(self.posRange[2][0],
        #                                   self.posRange[2][1])]
        #
        #     # Generate random starting orientation
        #     if self.ornRange is None:
        #         armStartOrn = q.random()
        #     else:
        #         rotationOffset = [random.uniform(self.ornRange[0][0],
        #                                          self.ornRange[0][1]),
        #                           random.uniform(self.ornRange[1][0],
        #                                          self.ornRange[1][1]),
        #                           random.uniform(self.ornRange[2][0],
        #                                          self.ornRange[2][1])]
        #         armStartOrn = q.rotateGlobal(restOrn, rotationOffset[0],
        #                                      rotationOffset[1],
        #                                      rotationOffset[2])
        #
        #     # Reset to the pose and check it is reachable
        #     self.robot.reset(armStartPos, armStartOrn, restGrip)
        #     armPos, armOrn = self.robot.getPose()
        #     if self._getDist(armPos, armStartPos) < 0.01:
        #         if q.distance(armOrn, armStartOrn) < 0.01:
        #             reachAble = True
        #         else:
        #             print("orndist: ", q.distance(armOrn, armStartOrn))
        #             time.sleep(1)
        #     else:
        #         print("posdist: ", self._getDist(armPos, armStartPos))
        #         time.sleep(1)
        # Generate random starting position
        armStartPos = [random.uniform(self.posRange[0][0],
                                      self.posRange[0][1]),
                       random.uniform(self.posRange[1][0],
                                      self.posRange[1][1]),
                       random.uniform(self.posRange[2][0],
                                      self.posRange[2][1])]

        # Generate random starting orientation
        if self.ornRange is None:
            armStartOrn = q.random()
        else:
            rotationOffset = [random.uniform(self.ornRange[0][0],
                                             self.ornRange[0][1]),
                              random.uniform(self.ornRange[1][0],
                                             self.ornRange[1][1]),
                              random.uniform(self.ornRange[2][0],
                                             self.ornRange[2][1])]
            armStartOrn = q.rotateGlobal(restOrn, rotationOffset[0],
                                         rotationOffset[1],
                                         rotationOffset[2])

        # Reset to the pose and check it is reachable
        self.robot.reset(armStartPos, armStartOrn, restGrip)

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

        # Get pose from the last step
        pos = self.lastObs[0:3]
        orn = self.lastObs[3:7]

        # Calculate the goal pose
        goalPos = np.add(pos, [xDist, yDist, zDist])
        goalOrn = q.rotateGlobal(orn, roll, yaw, pitch)

        # pos, orn = self.robot.getPose()
        # goalPos = [pos[0] + xDist,
        #            pos[1] + yDist,
        #            pos[2] + zDist]
        # orn = q.rotateGlobal(orn, roll, yaw, pitch)

        # Don't interact with the gripper
        gripAction = None

        return goalPos, goalOrn, gripAction


    def _getObs(self):
        # Get object pose and velocity in world frame
        objPos, objOrn = self.scene.getPose(self.goalObj)
        objLinVel, objAngVel = self.scene.getVelocity(self.goalObj)

        # Get robot arm pose
        armPos, armOrn = self.robot.getPose()

        # Get relative pose between robot arm and object
        relativeObjPos = np.subtract(objPos, armPos)
        relativeObjOrn = self.p.getDifferenceQuaternion(armOrn, objOrn)

        # Get relative pose between robot arm and the goal pose
        relativeGoalPos = np.subtract(self.goalPos, armPos)
        relativeGoalOrn = self.p.getDifferenceQuaternion(armOrn, self.goalOrn)

        # Concatenate and return observation
        obs = np.concatenate((objPos, objOrn, objLinVel, objAngVel,
                              relativeObjPos, relativeObjOrn, self.goalPos,
                              self.goalOrn, relativeGoalPos, relativeGoalOrn))
        return obs


    def _getReward(self, action):
        # Reward for being in a successful state
        successReward = 0
        if self._isSuccess():
            successReward = 1

        # Penalty for collision
        collisionPenalty = 0
        if self._isContact():
            collsionPenalty = self.collisionPenalty

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
            accuracyReward = math.exp(2*-error)

            # Penalise mean action magnitude
            actionPenalty = 0.3 * (np.linalg.norm(action, ord=1)/len(action))

            # Return dense reward
            return accuracyReward + successReward - actionPenalty - collisionPenalty


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
        return self.robot.nObs + 34


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
            time.sleep(1)
            self.reset()
        else:
            # If a successful reset then initialise error tracker
            armPos, armOrn = self.robot.getPose()
            posDist = self._getDist(armPos, self.goalPos)
            ornDist = q.distance(armOrn, self.goalOrn)
            self.prevError = posDist + ornDist

            # Initialise info
            self._collisions = 0


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
