import random
import math

import numpy as np

from behaviour_gym.primitive import primitiveNoIk
from behaviour_gym.utils import quaternion as q


class PlaceNoIk(primitiveNoIk.PrimitiveNoIk):

    def __init__(self, maxArmAction=0.02, maxGripAction=0.01,
                 collisionPenalty=1, sparse=False, distanceThreshold=0.02,
                 *args, **kwargs):
        """Initialises reach primitive.

        Args:
            maxAction (float): largest position change along each axes in metres.
            collisionPenalty (float): how much to remove from reward when a
                                      collision occurs.
            sparse (boolean): whether or not to return a sparse reward
            distanceThreshold (float): error that position and orientation
                                       accuracy must be higher than to be
                                       considered successful.

        """
        # Env variables
        self.maxArmAction = maxArmAction
        self.maxGripAction = maxGripAction
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.distanceThreshold = distanceThreshold

        super(PlaceNoIk, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get goal place pose
        pos = np.random.uniform([.4, -.4], [.95, .4])
        self.goalPos = np.append(pos, .64)
        self.goalOrn = self.p.getQuaternionFromEuler([0, 0, math.pi/2])

        # Start near goal place pose
        posNoise = np.random.uniform([-0.02, -0.02, 0.02], [.02, .02, .04])
        startPos = np.add(self.goalPos, posNoise)
        ornNoise = np.random.uniform(-.1, .1, (3,))
        startOrn = q.rotateGlobal(self.robot.restOrn, *ornNoise)
        self.robot.reset(startPos, startOrn)

        # Reset object to be in hand
        self.objId = self.scene.getObject("block")[0]
        objPos, objOrn = self.scene.getPose("block")
        posNoise = np.random.uniform([-0.005, -0.005, -0.015], [.005, .005, 0.01])
        startPos = np.add(startPos, posNoise)
        ornNoise = np.random.uniform(-.1, .1, (3,))
        startOrn = q.rotateGlobal(objOrn, *ornNoise)
        self.p.resetBasePositionAndOrientation(self.objId, startPos,
                                               startOrn)

        # Reset gripper then set motors to continually close
        self.robot.resetGripState([.765])
        self.robot.applyGripState([.775])

        return True


    def _applyAction(self, action):
        # Copy and rescale action
        action = action.copy()
        armAction = action[:6]
        armAction = armAction * self.maxArmAction
        gripAction = action[6:]
        # gripAction = gripAction * self.maxGripAction

        # Combine current arm velocities with action vector
        currVel = self.robot.getArmJointsVel()
        goalVel = np.add(currVel, armAction)
        goalVel = np.clip(goalVel, -0.1, 0.1)

        # Update robot motors
        self.robot.setArmMotorsVel(goalVel)

        # # Combine current gripper positions with action vector
        # currPos = self.robot.getGripJointAngles()
        # goalPos = np.add(currPos, gripAction)
        #
        # # Update robot motors
        # self.robot.setGripMotorsPos(goalPos)

        self.robot.applyGripAction(gripAction)


    def _getObs(self):
        # Get object pose and velocities
        objPos, objOrn = self.scene.getPose("block")
        objLinVel, objAngVel = self.scene.getVelocity("block")

        # Concatenate and return observation
        obs = np.concatenate((objPos, objOrn, objLinVel, objAngVel,
                              self.goalPos, self.goalOrn, [self.stepCount]))
        return obs


    def _getReward(self, action):
        # Penalise dropping the object
        if self.objDropped:
            return -1

        # Reward for being in a success state
        successReward = 0
        if self.success:
            successReward = 1

        # Penalty for collisions
        collisionPenalty = 0
        if self.collision:
            collisionPenalty = self.collisionPenalty

        if self.sparse:
            # If sparse then return sparse rewards for success and collision
            reward = successReward - collisionPenalty
            return reward
        else:
            # Reward accuracy of block pose
            error = self.posError + self.ornError
            accuracyReward = 10 * math.exp(25 * -error)

            # If object placed successful then reward returning to rest config
            restReward = 0
            if self.success:
                restError = self.restPosError + self.restOrnError + self.restGripError
                restReward = 5 * math.exp(10 * -restError)

            # Penalise mean action magnitude
            actionPenalty = 0.05 * (np.linalg.norm(action, ord=1)/len(action))

            # Return dense reward
            return accuracyReward + restReward - collisionPenalty


    def _isDone(self):
        if self.sparse and self.success:
            return True
        elif self.stepCount >= 100:
            return True
        elif self.posError > 0.1 or self.ornError > 0.2:
            # End episode early if error high
            return True
        elif self.objDropped:
            # End episode early if object dropped
            return True
        return False


    def _isSuccess(self):
        if self.posError < self.distanceThreshold and self.ornError < self.distanceThreshold and not self.objGrasped and not self.objDropped:
            return True
        return False


    def _getNumObs(self):
        return 68


    def _getNumActions(self):
        return 7


    def _getInfo(self):
        info = {"collisions" : self._collisions,
                "position distance" : self.posError,
                "orientation distance" : self.ornError,
                "rest position distance": self.restPosError,
                "rest orientation distance": self.restOrnError,
                "rest gripper distance": self.restGripError,
                "object dropped": self.objDropped}
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
        blockPos, blockOrn = self.scene.getPose("block")
        self.posError = self._getDist(blockPos, self.goalPos)
        self.ornError = q.distance(blockOrn, self.goalOrn)
        armPos, armOrn = self.robot.getPose()
        restPos = np.add(blockPos, [0,0,.05])
        self.restPosError = self._getDist(armPos, restPos)
        self.restOrnError = q.distance(armOrn, self.robot.restOrn)
        gripState = self.robot.getGripState()[0]
        self.restGripError = abs(gripState - self.robot.restGripState[0])

        # Update contacts
        self.fingerContacts = self._getNumFingerTipContacts()
        self.objGrasped = self.fingerContacts > 1

        # Check if object dropped
        posChange = self.prevObjPos[2] - blockPos[2]
        self.objDropped = posChange > 0.03 and not self.objGrasped
        self.prevObjPos = blockPos

        # Check if successfully placed
        self.success = self._isSuccess()


    def _resetCallback(self, allowReset=True):
        if allowReset and self._isCollision():
            # Reset if starting in a collision state
            self.reset()
        else:
            # Initialise step counter
            self.stepCount = 0

            # Initialise collisions
            self._collisions = 0
            self.collision = False

            # Initialise error tracker
            blockPos, blockOrn = self.scene.getPose("block")
            self.posError = self._getDist(blockPos, self.goalPos)
            self.ornError = q.distance(blockOrn, self.goalOrn)
            armPos, armOrn = self.robot.getPose()
            restPos = np.add(blockPos, [0,0,.05])
            self.restPosError = self._getDist(armPos, restPos)
            self.restOrnError = q.distance(armOrn, self.robot.restOrn)
            gripState = self.robot.getGripState()[0]
            self.restGripError = abs(gripState - self.robot.restGripState[0])

            # Initialise contacts
            self.fingerContacts = self._getNumFingerTipContacts()
            self.objGrasped = self.fingerContacts > 1
            if allowReset and not self.objGrasped:
                self.reset()

            # Check if object dropped
            self.prevObjPos = blockPos
            posChange = self.prevObjPos[2] - blockPos[2]
            self.objDropped = posChange > 0.01 and not self.objGrasped
            self.success = self._isSuccess()


    # Helper methods
    # --------------------------------------------------------------------------

    def _isCollision(self):
        """Check for illegal collisions.

        Returns:
            True if collision, false otherwise.

        """
        # Check for contact between robot and any body other than the object
        contactPoints = self.p.getContactPoints(self.robot.id)
        for point in contactPoints:
            if point[2] != self.objId:
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
            # Only consider contacts with at least 0.1mm penetration and 0.1N force
            # Note that this will miss some legimate contacts
            if contactPoint[3] in fingerTips and contactPoint[8] < -0.0001 and contactPoint[9] > 0.1:
                contacts.append(contactPoint[3])
        numUniqueContacts = len(set(contacts))
        return numUniqueContacts
