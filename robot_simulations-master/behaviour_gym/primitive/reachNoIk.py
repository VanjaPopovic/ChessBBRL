import random
import math

import numpy as np

from behaviour_gym.primitive import primitiveNoIk
from behaviour_gym.utils import quaternion as q


class ReachNoIk(primitiveNoIk.PrimitiveNoIk):

    def __init__(self, maxAction=0.1, lowPosRange=[0.4, -0.4, 0.9],
                 highPosRange=[0.95, 0.4, 1.2], collisionPenalty=1,
                 sparse=False, distanceThreshold=0.05, *args, **kwargs):
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
        self.lowPosRange = lowPosRange
        self.highPosRange = highPosRange
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.distanceThreshold = distanceThreshold

        super(ReachNoIk, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        if np.random.randint(0, 2) == 1:
            # Reach while holding object
            self.holdingObj = True

            # Get goal place pose
            pos = np.random.uniform([.4, -.4], [.95, .4])
            self.goalPos = np.append(pos, .65)
            self.goalOrn = self.robot.restOrn

            # Start in random pose close to object
            objPos, objOrn = self.scene.getPose("block")
            posNoise = np.random.uniform([-0.02, -0.02, 0.01], [.02, .02, .04])
            startPos = np.add(objPos, posNoise)
            ornNoise = np.random.uniform(-.05, .05, (3,))
            startOrn = q.rotateGlobal(self.robot.restOrn, *ornNoise)
            self.robot.reset(startPos, startOrn)

            # Reset object to be in hand
            self.objId = self.scene.getObject("block")[0]
            self.p.resetBasePositionAndOrientation(self.objId, startPos,
                                                   objOrn)

            # Reset gripper then set motors to continually close
            self.robot.resetGripState([.75])
            self.robot.applyGripState([.775])
        else:
            # Reach without holding object
            self.holdingObj = False

            # Get goal grasp pose
            objPos, _ = self.scene.getPose("block")
            self.goalPos = np.add(objPos, [0, 0, 0.01])
            self.goalOrn = self.robot.restOrn

            # Start in a random pose
            armStartPos = np.random.uniform(self.lowPosRange, self.highPosRange)
            self.robot.reset(armStartPos, q.random())

        return True


    def _applyAction(self, action):
        # Copy and rescale action
        action = action.copy()
        action = action * self.maxAction

        # Combine current velocities with action vector
        currVel = self.robot.getArmJointsVel()
        goalVel = np.add(currVel, action)

        # Update robot motors
        self.robot.setArmMotorsVel(goalVel)


    def _getObs(self):
        objPos, objOrn = self.scene.getPose("block")
        objLinVel, objAngVel = self.scene.getVelocity("block")

        # Concatenate and return observation
        obs = np.concatenate((objPos, objOrn, objLinVel, objAngVel,
                              self.goalPos, self.goalOrn, [self.stepCount]))
        return obs


    def _getReward(self, action):
        # If dropped object then return penalty
        if self.holdingObj and self.fingerContacts < 2:
            return -1

        # Reward for being in a successful state
        successReward = 0
        if self.success:
            successReward = 1

        # Penalty for collision
        collisionPenalty = 0
        if self.collision:
            collisionPenalty = self.collisionPenalty

        if self.sparse:
            # If sparse then return sparse rewards for success and collision
            reward = successReward - collisionPenalty
            return reward
        else:
            # Reward accuracy of pose
            error = self.posError + self.ornError
            accuracyReward = 10 * math.exp(-error)

            # Penalise mean action magnitude
            actionPenalty = 0.05 * (np.linalg.norm(action, ord=1)/len(action))

            # Return dense reward
            return accuracyReward - collisionPenalty


    def _isDone(self):
        if self.sparse and self.success:
            return True
        if self.stepCount >= 100:
            return True
        if self.holdingObj and self.fingerContacts < 2:
            return True
        return False


    def _isSuccess(self):
        if self.posError < self.distanceThreshold and self.ornError < self.distanceThreshold and not self.collision:
            return True
        return False


    def _getNumObs(self):
        return 68


    def _getNumActions(self):
        return 6


    def _getInfo(self):
        info = {"collisions" : self._collisions,
                "position distance" : self.posError,
                "orientation distance" : self.ornError}
        return info


    def _stepCallback(self, action):
        # Update step counter
        self.stepCount += 1

        # Track collisions
        if self._isCollision():
            self.collision = True
            self._collisions += 1
        else:
            self.collision = False

        # Track Errors
        armPos, armOrn = self.robot.getPose()
        self.posError = self._getDist(armPos, self.goalPos)
        self.ornError = q.distance(armOrn, self.goalOrn)
        self.success = self._isSuccess()

        # Track contacts if holding object
        if self.holdingObj:
            self.fingerContacts = self._getNumFingerTipContacts()


    def _resetCallback(self):
        if self._isCollision():
            # Reset if starting in contact state
            self.reset()
        else:
            # Initialise step counter
            self.stepCount = 0

            # Initialise collisions
            self._collisions = 0
            self.collision = False

            # Initialise error tracker
            armPos, armOrn = self.robot.getPose()
            self.posError = self._getDist(armPos, self.goalPos)
            self.ornError = q.distance(armOrn, self.goalOrn)
            self.success = self._isSuccess()

            # Initialise contacts if holding object
            if self.holdingObj:
                self.fingerContacts = self._getNumFingerTipContacts()


    # Helper methods
    # --------------------------------------------------------------------------

    def _isCollision(self):
        """Check for illegal collisions.

        Returns:
            True if contact, false otherwise.

        """
        if self.holdingObj:
            # Check for contact between object and any body other than the robot
            contactPoints = self.p.getContactPoints(self.objId)
            for point in contactPoints:
                if point[2] != self.robot.id:
                    return True

            # Check for contact between robot and any body other than the object
            contactPoints = self.p.getContactPoints(self.robot.id)
            for point in contactPoints:
                if point[2] != self.objId:
                    if point[3] > 0:
                        return True
        else:
            # Check for contact between robot and any body
            contactPoints = self.p.getContactPoints(self.robot.id)
            for point in contactPoints:
                if point[3] > 0:
                    return True
        return False


    def _getNumFingerTipContacts(self):
        """Get the number of finger tips in contact with the block."""
        contactPointsBlock = self.p.getContactPoints(self.robot.id,
                                                     self.objId)
        fingerTips = self.robot.getFingerTipLinks()
        contacts = []
        for contactPoint in contactPointsBlock:
            if contactPoint[3] in fingerTips:
                contacts.append(contactPoint[3])

        numUniqueContacts = len(set(contacts))
        return numUniqueContacts
