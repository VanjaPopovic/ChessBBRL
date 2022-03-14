import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q


class GraspEasy(primitive.Primitive):

    def __init__(self, goalObject, graspGenerator, goalPosOffset=[0,0,0.01],
                 maxMove=0.01, maxRotation=math.pi/12,
                 posRange=[[-0.02,0.02],[-0.02,0.02],[-0.01,0.02]],
                 ornRange=[[-math.pi/2, math.pi/2], [-math.pi/2, math.pi/2],
                           [-math.pi/2, math.pi/2]],
                 collisionPenalty=0.0, sparse=False, distanceThreshold=0.03,
                 *args, **kwargs):
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

        # Env info
        self._collisions = 0.0

        super(GraspEasy, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get random starting position
        blockPos, blockOrn = self.scene.getPose(self.goalObj)
        armStartPos = [random.uniform(self.posRange[0][0] + blockPos[0],
                                      self.posRange[0][1] + blockPos[0]),
                       random.uniform(self.posRange[1][0] + blockPos[1],
                                      self.posRange[1][1] + blockPos[1]),
                       random.uniform(self.posRange[2][0] + blockPos[2] + 0.02,
                                      self.posRange[2][1] + blockPos[2])]

        # Reset robot
        self.robot.reset(armStartPos)

        # Reset info
        self._collisions = 0

        # # Set episode goal
        # objPos, self.goalOrn = scene.getPose(self.goalObj)
        # self.goalPos = np.add(objPos, self.goalPosOffset)

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
        goalOrn = q.rotateGlobal(orn, roll, yaw, pitch)

        # Clip goal position within posRange
        blockPos, blockOrn = self.scene.getPose(self.goalObj)
        minPos = [self.posRange[0][0] + blockPos[0],
                  self.posRange[1][0] + blockPos[1],
                  self.posRange[2][0] + blockPos[2]]
        maxPos = [self.posRange[0][1] + blockPos[0],
                  self.posRange[1][1] + blockPos[1],
                  self.posRange[2][1] + blockPos[2]]
        goalPos = np.clip(goalPos, minPos, maxPos)

        # Get Gripper action
        gripAction = [action[3]]

        return goalPos, goalOrn, gripAction



    def _getObs(self):
        # Get object state in world frame
        objPosWorld, objOrnWorld = self.scene.getPose(self.goalObj)
        objLinVelWorld, objAngVelWorld = self.scene.getVelocity(self.goalObj)

        # Get relative distance between robot and object
        pos, orn = self.robot.getPose()
        relativePos = [objPosWorld[0] - pos[0],
                       objPosWorld[1] - pos[1],
                       objPosWorld[2] - pos[2]]

        obs = np.concatenate((objPosWorld, objOrnWorld, objLinVelWorld,
                              objAngVelWorld, relativePos))
        return obs


    def _getReward(self):
        if self.sparse:
            if self._isSuccess():
                reward = 1
            else:
                reward = 0
        else:
            # Reward for reaching goal pose for grasping
            # fingerDist = np.clip(self._getGraspDist(), 0.0, 0.12)
            # fingerDist = (fingerDist) / (0.12)
            # fingerReward = 1 - fingerDist**0.4

            # Reward for touching block with finger tips
            # fingerContacts = self._getNumFingerTipContacts()
            fingerContacts = np.clip(self._getNumFingerTipContacts(), 0, 2)
            # numFingers = len(self.robot.getFingerTipLinks())
            numFingers = np.clip(len(self.robot.getFingerTipLinks()), 0, 2)
            contactReward = fingerContacts / numFingers

            # Reward for lifting block
            liftDist = np.clip(self._getLiftGoalDist(), 0, 0.05)
            liftDist = liftDist / 0.05
            liftReward = 1 - liftDist

            # Max penalty of -50% for a 15 degree change in orientation
            ornDist = np.clip(self._getOrientationDrift(), 0, math.pi/12)
            # ornDist = np.clip(self._getOrientationDrift(), 0, math.pi/2)
            ornDist = ornDist / (math.pi/2)
            ornPenalty = 1 - ornDist*0.5
            # ornPenalty = 1 - ornDist*0.75

            # Max penalty of -50% for a 2cm change in position
            centeringDist = np.clip(self._getPoseDrift(), 0, 0.02)
            # centeringDist = np.clip(self._getPoseDrift(), 0, 0.05)
            centeringDist = centeringDist / 0.05
            centeringPenalty = 1 - centeringDist*0.5
            # centeringPenalty = 1 - centeringDist*0.75

            totalReward = contactReward + liftReward
            # totalReward = fingerReward + contactReward + liftReward
            totalPenalty = centeringPenalty*ornPenalty
            reward = totalReward*totalPenalty

        return reward


    def _isDone(self):
        if self.sparse and self._isSuccess():
            return True
        return False


    def _isSuccess(self):
        goalDist = self._getLiftGoalDist()
        if goalDist < self.distanceThreshold:
            return True
        return False


    def _getNumObs(self):
        return self.robot.nObs + 16


    def _getNumActions(self):
        return 7


    def _getInfo(self):
        info = {"lift distance" : self._getLiftGoalDist(),
                "Orientaton drift" : self._getTotalOrientationDrift(),
                "Centering drift" : self._getTotalPoseDrift(),
                "contact reward" : self._getNumFingerTipContacts() / len(self.robot.getFingerTipLinks()),
                "grasp pose distance" : self._getGraspDist(),
                "is_success": self._isSuccess()}
        return info


    def _stepCallback(self):
        self.prevPos, self.prevOrn = self.scene.getPose(self.goalObj)


    def _resetCallback(self):
        self.prevPos, self.prevOrn = self.scene.getPose(self.goalObj)
        self.initPos, self.initOrn = self.prevPos, self.prevOrn


    # Helper methods
    # --------------------------------------------------------------------------

    def _getGraspDist(self):
        """Gets the distance between the robot's current and goal poses."""
        armPos, _ = self.robot.getPose()
        objPos, _ = self.scene.getPose(self.goalObj)
        goalPos = np.add(objPos, self.goalPosOffset)
        return self._getDist(armPos, goalPos)


    def _getLiftGoalDist(self):
        """Gets the distance of the block's Z position from the goal position"""
        blockPos, _ = self.scene.getPose(self.goalObj)
        return abs(0.69 - blockPos[2])


    def _getOrientationDrift(self):
        """Gets the distance between the current and goal orientation."""
        pos, orn = self.scene.getPose(self.goalObj)
        return self._getQuaternionDist(orn, self.prevOrn)


    def _getTotalOrientationDrift(self):
        pos, orn = self.scene.getPose(self.goalObj)
        return self._getQuaternionDist(orn, self.initOrn)


    def _getPoseDrift(self):
        """Gets the distance between the current and goal orientation."""
        pos, orn = self.scene.getPose(self.goalObj)
        return self._getDist(pos[:2], self.prevPos[:2])


    def _getTotalPoseDrift(self):
        pos, orn = self.scene.getPose(self.goalObj)
        return self._getDist(pos[:2], self.initPos[:2])


    def _getNumFingerTipContacts(self):
        """Get the number of finger tips in contact with the block."""
        contactPointsBlock = self.p.getContactPoints(self.robot.id,
                                                     self.scene.getObjects()[self.goalObj][0])
        fingerTips = self.robot.getFingerTipLinks()
        contacts = []
        for contactPoint in contactPointsBlock:
            # print(contactPoint)
            if contactPoint[3] in fingerTips:
                contacts.append(contactPoint[3])

        numUniqueContacts = len(set(contacts))
        return numUniqueContacts
