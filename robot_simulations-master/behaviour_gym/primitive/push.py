import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q


class Push(primitive.Primitive):

    def __init__(self, goalObject, goalPosOffset=[0,0,0.0], maxMove=0.01,
                 maxRotation=math.pi/20,
                 posRange=[[0.3,1.0],[-0.42,0.42],[0.9,1.3]],
                 ornRange=[[-math.pi/2, math.pi/2], [-math.pi/2, math.pi/2],
                           [-math.pi/2, math.pi/2]],
                 collisionPenalty=0.02, sparse=False, distanceThreshold=0.05,
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

        super(Push, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get goal reaching pose
        objPos, objOrn = self.scene.getPose("lever")
        self.objId = self.scene.getObject("lever")[0]
        objPos = self.p.getLinkState(self.objId, 0)[0]
        objOrn = self.p.getLinkState(self.objId, 0)[1]

        #print("obj pos obj orn")
        #print(objPos)
        
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

    def _getAction(self, action):
        action = action.copy()
        #print(action)
        # Calculate distances in metres
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
        gripAction = [0.775]

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
            reward = 1 - np.clip(dist, 0, 1) ** 0.4
            buttonId, linkId = self.scene.getObject("lever")
            buttonZ = self.p.getJointState(buttonId, 0)
            normal = abs(buttonZ[0])
           
            clipped = np.clip(normal, 0, 1)
            reward = reward + clipped  
            # Penalise contact
            if self._isContact():
                reward -= self.collisionPenalty
        return reward

    def _isDone(self):
        if self._isSuccess():
            return True
        return False

    def _isSuccess(self):
        goalDist = self._getGoalDist()
        if goalDist < self.distanceThreshold:
            buttonId, linkId = self.scene.getObject("lever")
            buttonZ = self.p.getJointState(buttonId, 0)
            if buttonZ[0] < -0.8:
                print("success")
                return True
        return False

    def _getNumObs(self):
        return self.robot.nObs + 16

    def _getNumActions(self):
        return 6

    def _getInfo(self):
        info = {"collisions": self._collisions,
                "goal distance": self._getGoalDist()}
        return info

    def _stepCallback(self):
        if self._isContact():
            self._collisions += 1
        #print(self._getInfo())
        #print(self.printInfo())
        buttonId, linkId = self.scene.getObject("lever")
        # print(buttonId)
        # print(linkId)
        #print(self.p.getLinkState(buttonId, 0)[4][2])
        self._collisions = 0 
        #print(self.printInfo())
        

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

