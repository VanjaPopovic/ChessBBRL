import random
import math

import numpy as np

from behaviour_gym.primitive import primitiveNoIk
from behaviour_gym.utils import quaternion as q


class PushNoIk(primitiveNoIk.PrimitiveNoIk):

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

        super(PushNoIk, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get goal reaching pose
        #objPos, objOrn = self.scene.getPose("lever")
        self.objId = self.scene.getObject("lever")[0]
        objPos = self.p.getLinkState(self.objId, 0)[0]
        objOrn = self.p.getLinkState(self.objId, 0)[1]

        print("obj pos obj orn")
        print(objPos)
        self.goalPos = np.add(objPos, [0, .0, 0.00])
        self.goalOrn = self.robot.restOrn

        # Start near the goal reaching pose
        posNoise = np.random.uniform([-.02, -.02, .01], [.02, .02, .03])
        startPos = np.add(self.goalPos, posNoise)
        #ornNoise = np.random.uniform(-.1, .1, (3,))
        #startOrn = q.rotateGlobal(self.goalOrn, *ornNoise)
        self.robot.reset(self.goalPos, self.goalOrn)

        # Get goal lifting pose
        self.goalLiftPos = np.add(objPos, [0, 10, 0.05])
        self.goalLiftOrn = self.p.getQuaternionFromEuler([0, 0, math.pi/2])

        # Get object id for collision calculation
        self.objId = self.scene.getObject("lever")[0]
        return True


    def _applyAction(self, action):
        # Copy and rescale action
        action = action.copy()
        armAction = action[:6]
        armAction = armAction * self.maxArmAction
        gripAction = action[6:]
        # gripAction = gripAction * self.maxGripAction

        # # Combine current arm velocities with action vector
        # currVel = self.robot.getArmJointsVel()
        # goalVel = np.add(currVel, armAction)
        # goalVel = np.clip(goalVel, -0.1, 0.1)
        #
        # # Update robot motors
        # self.robot.setArmMotorsVel(goalVel)

        currVel = self.robot.getArmJointsVel()
        goalVel = np.add(currVel, armAction)

        # Update robot motors
        self.robot.setArmMotorsVel(goalVel)
        print(goalVel)

        # # Combine current gripper positions with action vector
        # currPos = self.robot.getGripJointAngles()
        # goalPos = np.add(currPos, gripAction)
        #
        # # Update robot motors
        # self.robot.setGripMotorsPos(goalPos)
        print("grip ACTION")
        print(gripAction[0])
        #gripAction[0] = 1000000
        self.robot.applyGripAction(gripAction)


    def _getObs(self):
        # Get object pose and velocities
        objPos, objOrn = self.scene.getPose("lever")
        objLinVel, objAngVel = self.scene.getVelocity("lever")

        # Concatenate and return observation
        obs = np.concatenate((objPos, objOrn, objLinVel, objAngVel,
                              self.goalPos, self.goalOrn,
                              self.goalLiftPos, self.goalLiftOrn,
                              [self.stepCount]))
        return obs


    def _getReward(self, action):
        # Reward for being in a successful state
        successReward = 0
        if self.success:
            successReward = 1

        # Penalty for collisions
        collisionPenalty = 0
        print("collides")
        print(self.collision)
        if self.collision:
            collisionPenalty = self.collisionPenalty

        if self.sparse:
            # If sparse then return sparse rewards for success and collision
            reward = successReward - collisionPenalty
            return reward
        else:
            # Reward for touching block with finger tips
            #numFingers = len(self.robot.getFingerTipLinks())
            #contactReward = self.fingerContacts / numFingers

            #print("contact r")
            #print(contactReward)

            # Reward for reaching grasp pose
            # accuracyReward = 1
            # if not self.objGrasped:
            #     error = self.armPosError + self.armOrnError
            #     accuracyReward = math.exp(25 * -error)
            #
            # # Reward accuracy of lift pose if object grasped
            liftReward = 0
            blockPos, _ = self.scene.getPose("lever")
            if blockPos[1] > 0.245:
                error = self.posError + self.ornError
                liftReward = 10 * math.exp(25 * -error)

            # Penalise mean action magnitude
            actionPenalty = 0.05 * (np.linalg.norm(action, ord=1)/len(action))

            # Return dense reward
            print("reward")
            print( liftReward )
            return liftReward


    def _isDone(self):
        if self.sparse and self.success:
            return True
        elif self.stepCount >= 100:
            return True
        elif self.posError > 0.1 or self.ornError > 0.1:
            # End episode early if error high
            return True
        return False


    def _isSuccess(self):
        if self.posError < self.distanceThreshold and self.ornError < self.distanceThreshold:
            return True
        return False


    def _getNumObs(self):
        return 68 + 7


    def _getNumActions(self):
        # return 17
        return 7


    def _getInfo(self):
        info = {"collisions" : self._collisions,
                "position distance" : self.armPosError,
                "orientation distance" : self.armOrnError,
                "lift position distance": self.posError,
                "lift orientation distance": self.ornError,
                "finger contacts": self.fingerContacts}
        print("IUNFOOO")
        print(info)
        return info


    def _stepCallback(self, action):
        # Update step counter
        self.stepCount += 1
        print("damesa")
        # Track collisions
        if self._isCollision():
            self.collision = True
            self._collisions += 1
        else:
            self.collision = False

        # Track Errors
        blockPos, blockOrn = self.scene.getPose("lever")
        self.posError = self._getDist(blockPos, self.goalLiftPos)
        self.ornError = q.distance(blockOrn, self.goalLiftOrn)
        self.success = self._isSuccess()
        armPos, armOrn = self.robot.getPose()
        self.armPosError = self._getDist(armPos, self.goalPos)
        self.armOrnError = q.distance(armOrn, self.goalOrn)

        # Track contacts
        self.fingerContacts = self._getNumFingerTipContacts()
        self.objGrasped = self.fingerContacts > 1
        self.printInfo()


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
            blockPos, blockOrn = self.scene.getPose("lever")
            self.posError = self._getDist(blockPos, self.goalLiftPos)
            self.ornError = q.distance(blockOrn, self.goalLiftOrn)
            self.success = self._isSuccess()
            armPos, armOrn = self.robot.getPose()
            self.armPosError = self._getDist(armPos, self.goalPos)
            self.armOrnError = q.distance(armOrn, self.goalOrn)

            # Initialise contacts
            self.fingerContacts = self._getNumFingerTipContacts()
            #self.objGrasped = self.fingerContacts > 1
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
        contactPoints = self.p.getContactPoints(self.robot.id)
        for point in contactPoints:
            if point[2] != self.objId:
                if point[3] > 0:
                    return True

        return False


    def _getNumFingerTipContacts(self):
        """Get the number of finger tips in contact with the block."""
        contactPointsBlock = self.p.getContactPoints(self.robot.id)
        print(self.p.getLinkState(self.objId, 0))
        #print("contact points")

        #print(contactPointsBlock)
        fingerTips = self.robot.getFingerTipLinks()
        contacts = []
        # for contactPoint in contactPointsBlock:
        #     if contactPoint[3] in fingerTips:
        #         contacts.append(contactPoint[3])

        for point in contactPointsBlock:
            if point[2] != self.objId:
                if point[3] > 0:
                    contacts.append(contactPoint[3])


        numUniqueContacts = len(set(contacts))
        print(numUniqueContacts)
        return numUniqueContacts

    # def _getNumFingerTipContacts(self):
    #     """Get the number of finger tips in contact with the block."""
    #     contactPointsBlock = self.p.getContactPoints(self.robot.id,
    #                                                  self.objId)
    #     fingerTips = self.robot.getFingerTipLinks()
    #     contacts = []
    #     for contactPoint in contactPointsBlock:
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
