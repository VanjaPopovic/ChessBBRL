import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q
import random


class PressNoIk(primitive.Primitive):

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
        self.maxMove = 0.05
        self.maxRot = math.pi/20
        self.collisionPenalty = collisionPenalty
        self.sparse = sparse
        self.distanceThreshold = distanceThreshold

        super(PressNoIk, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get goal reaching pose
        objPos, objOrn = self.scene.getPose("button")
        self.goalPos = np.add(objPos, [0, 0, 0.3])
        self.goalOrn = self.robot.restOrn

        # Start near the goal reaching pose
        #posNoise = np.random.uniform([-.02, -.02, 0], [.02, .02, .04])
        #startPos = np.add(self.goalPos, posNoise)
        startPos = self.goalPos
        ornNoise = np.random.uniform(-.1, .1, (3,))
        startOrn = q.rotateGlobal(self.goalOrn, *ornNoise)
        self.robot.reset(startPos, startOrn)

        # Get goal lifting pose
        self.goalLiftPos = np.add(objPos, [0, 0, 0])
        self.goalLiftOrn = self.p.getQuaternionFromEuler([0, 0, math.pi/2])

        # Get object id for collision calculation
        self.objId = self.scene.getObject("button")[0]

        return True


    def _applyAction(self, action):
        # Copy and rescale action
        # action = action.copy()
        # armAction = action[:6]
        # armAction = armAction * self.maxArmAction
        # gripAction = action[6:]
        # # gripAction = gripAction * self.maxGripAction
        #
        # # Combine current arm velocities with action vector
        # currVel = self.robot.getArmJointsVel()
        # goalVel = np.add(currVel, armAction)
        # goalVel = np.clip(goalVel, -0.1, 0.1)
        #
        # # Update robot motors
        # self.robot.setArmMotorsVel(goalVel)
        #
        # # # Combine current gripper positions with action vector
        # # currPos = self.robot.getGripJointAngles()
        # # goalPos = np.add(currPos, gripAction)
        # #
        # # # Update robot motors
        # # self.robot.setGripMotorsPos(goalPos)
        #
        # self.robot.applyGripAction(gripAction)
        print(action)
        action = action.copy()
        xDist = action[0] * self.maxMove
        yDist = action[1] * self.maxMove
        zDist = action[2] * self.maxMove

        # Calculate rotation in radians
        roll = action[3] * self.maxRot
        yaw = action[4] * self.maxRot
        pitch = action[5] * self.maxRot

        # Get robot's relative position and orienation
        pos, orn = self.robot.getPose()

        # Calculate the goal pose in the base frame
        goalPosBase = [pos[0] + xDist,
                       pos[1] + yDist,
                       pos[2] + zDist]
        goalOrnBase = q.rotateGlobal(orn, roll, yaw, pitch)

        # Don't interact with the gripper
        gripAction = None
        #Calculate distances in metres

        return goalPos, goalOrn, gripAction

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
        gripAction = None

        return pos, orn, gripAction



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
        return 6


    def _getInfo(self):
        info = {"collisions" : self._collisions,
                "goal distance" : self._getGoalDist()}
        return info

    def _stepCallback(self):
        # Update step counter
        self.stepCount += 1

        # Track collisions
        if self._isCollision():
            self.collision = True
            self._collisions += 1
        else:
            self.collision = False

        # Track Errors
        buttonPos, buttonOrn = self.scene.getPose("button")
        self.posError = self._getDist(buttonPos, self.goalLiftPos)
        self.ornError = q.distance(buttonOrn, self.goalLiftOrn)
        self.success = self._isSuccess()
        armPos, armOrn = self.robot.getPose()
        self.armPosError = self._getDist(armPos, self.goalPos)
        self.armOrnError = q.distance(armOrn, self.goalOrn)

        # Track contacts
        self.fingerContacts = self._getNumFingerTipContacts()
        self.objGrasped = self.fingerContacts > 1

        buttonId, linkId = self.scene.getObject("button")
        # print(buttonId)
        # print(linkId)
        #print(self.p.getLinkState(buttonId, 0)[4][2])
        buttonZ = self.p.getLinkState(buttonId, 0)[4][2]
        if buttonZ > 0.63:
            print("button pressed")
            self.p.changeVisualShape(buttonId, 0,
                                rgbaColor=[random.random(), random.random(), random.random(), 1])
        #self.printInfo()





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
            buttonPos, buttonOrn = self.scene.getPose("button")
            self.posError = self._getDist(buttonPos, self.goalLiftPos)
            self.ornError = q.distance(buttonOrn, self.goalLiftOrn)
            self.success = self._isSuccess()
            armPos, armOrn = self.robot.getPose()
            self.armPosError = self._getDist(armPos, self.goalPos)
            self.armOrnError = q.distance(armOrn, self.goalOrn)

            # Initialise contacts
            self.fingerContacts = self._getNumFingerTipContacts()
            self.objGrasped = self.fingerContacts > 1


    # Helper methods
    # --------------------------------------------------------------------------

    def _isCollision(self):
        """Check if contact made with the robot.

        Ignores links with id <1 which are usually part of the base.

        Returns:
            True if contact, false otherwise.

        """
        # Check for contact between robot and any body other than the object
        
        #contactPoints = self.p.getContactPoints(self.robot.id)
        #for point in contactPoints:
            #if point[2] != self.objId:
                #if point[3] > 0:
                    #return True

        return False


    def _getNumFingerTipContacts(self):
        """Get the number of finger tips in contact with the button."""
        contactPointsbutton = self.p.getContactPoints(self.robot.id,
                                                     self.objId)
        fingerTips = self.robot.getFingerTipLinks()
        contacts = []
        for contactPoint in contactPointsbutton:
            if contactPoint[3] in fingerTips:
                contacts.append(contactPoint[3])

        numUniqueContacts = len(set(contacts))
        return 3

    # def _getNumFingerTipContacts(self):
    #     """Get the number of finger tips in contact with the button."""
    #     contactPointsbutton = self.p.getContactPoints(self.robot.id,
    #                                                  self.objId)
    #     fingerTips = self.robot.getFingerTipLinks()
    #     contacts = []
    #     for contactPoint in contactPointsbutton:
    #         # Only consider contacts with at least 0.1mm penetration and 0.1N force
    #         # Note that this will miss some legimate contacts
    #         if contactPoint[3] in fingerTips and contactPoint[8] < -0.0001 and contactPoint[9] > 0.1:
    #             contacts.append(contactPoint[3])
    #     numUniqueContacts = len(set(contacts))
    #     return numUniqueContacts

    def printInfo(self):
        """Prints info about the robot's links and joints"""
        print("################Link Info################")
        print(f'|{"Index":<5} | {"Name":<30}|')
        linkNameIndex = {self.p.getBodyInfo(2)[0].decode('UTF-8'):-1,}
        for i in range(self.p.getNumJoints(2)):
        	linkName = self.p.getJointInfo(2, i)[12].decode('UTF-8')
        	linkNameIndex[linkName] = i
        for link in linkNameIndex.items():
            print(f'|{link[1]:<5} | {link[0]:<30}|')

        print("")
        print("################Link State################")
        print(f'|{"index":<5} | {"World Position":<17} | {"World Orientation":<23} | {"Linear Velocity":<17} | {"Angular Velocity":<17}|')
        for i in range(0, len(linkNameIndex)-1):
            state = self.p.getLinkState(2, i, computeLinkVelocity=True)
            pos = ' '.join(map('{:5.2f}'.format, state[0]))
            orn = ' '.join(map('{:5.2f}'.format, state[1]))
            posVel = ' '.join(map('{:5.2f}'.format, state[6]))
            ornVel = ' '.join(map('{:5.2f}'.format, state[7]))
            print(f'|{i:<5} | {pos:<17} | {orn:<23} | {posVel:<17} | {ornVel:<17}|')

        print("")
        print("################Joint Info################")
        print(f'|{"Index":<5} | {"Name":<30} | {"Type":<4} | {"Damping":7} | {"Friction":<8} | {"Lower Limit":<11} | {"Upper Limit":<11} | {"Max Force":<9} | {"Max Velocity":<12} | {"Axis":<11} | {"Parent Index":<12}|')
        for i in range(self.p.getNumJoints(2)):
            info = self.p.getJointInfo(2, i)
            pos = ' '.join(map('{:1}'.format, info[13]))
            print(f'|{info[0]:<5} | {str(info[1]):<30} | {info[2]:<4} | {info[6]:<7} | {info[7]:<8} | {info[8]:<11.4} | {info[9]:<11.4} | {info[10]:<9} | {info[11]:<12} | {pos:<11} | {info[16]:<12}|')
        print("")
        print("################Joint State################")
        print(f'|{"index":<5} | {"Position":<8} | {"Velocity":<8} | {"Reaction Forces":<35} | {"Motor Torque":<12}|')
        for i in range(self.p.getNumJoints(2)):
            state = self.p.getJointState(2, i)
            reactionForces = ' '.join(map('{:5.2f}'.format, state[2]))
            print(f'|{i:<5} | {state[0]:<8.2f} | {state[1]:<8.2f} | {reactionForces:<35} | {state[3]:<12.2f}|')
