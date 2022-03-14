import random
import math

import numpy as np

from behaviour_gym.primitive import primitive
from behaviour_gym.utils import quaternion as q


class ReachWithObject(primitive.Primitive):

    def __init__(self, goalObject, goalPosOffset=[0,0,0.0], maxMove=0.01,
                 maxRotation=math.pi/20,
                 posRange=[[0.3,1.0],[-0.42,0.42],[0.05,0.05]],
                 ornRange=[[-math.pi/2, math.pi/2], [-math.pi/2, math.pi/2],
                           [-math.pi/2, math.pi/2]],
                 collisionPenalty=1, sparse=False, distanceThreshold=0.05,
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
        #pos = np.random.uniform([.4, -.4], [.65, .4])
        self.reachPos = [0.8, 0.4, 0.69]
        

        self._collisions = 0.0

        super(ReachWithObject, self).__init__(*args, **kwargs)


    # Environment Extension Methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        # Get random starting position
   
        pos = np.random.uniform([.4, -.4, 0.95], [.95, .4, 0.65])

        self.goalPos = pos
        pos = np.random.uniform([.4, -.4], [.95, .4])
        self.reachPos = np.append(pos, .69) 
        self.goalOrn = self.p.getQuaternionFromEuler([0, 0, math.pi/2])

        # Start near goal place pose
        posNoise = np.random.uniform([-0.02, -0.02, 0.02], [.02, .02, .04])
        startPos = np.add(self.goalPos, posNoise)
        ornNoise = np.random.uniform(-.1, .1, (3,))
        startOrn = q.rotateGlobal(self.robot.restOrn, *ornNoise)
        self.robot.reset(startPos, self.robot.restOrn)

        # Reset object to be in hand
        self.objId = self.scene.getObject("block")[0]
        objPos, objOrn = self.scene.getPose("block")
        posNoise = np.random.uniform([-0.005, -0.005, -0.015], [.005, .005, 0.01])
        startPos = np.add(startPos, posNoise)
        ornNoise = np.random.uniform(-.1, .1, (3,))
        startOrn = q.rotateGlobal(objOrn, *ornNoise)
        self.p.resetBasePositionAndOrientation(self.objId, startPos,
                                               self.goalOrn)

        # Reset gripper then set motors to continually close
        self.robot.resetGripState([.765])
        self.robot.applyGripState([.775])
        


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
        gripAction = [0.01]

        return goalPosBase, goalOrnBase, gripAction



    def _getObs(self):
        # Get object state
        #objPos, objOrn = self.scene.getPose(self.goalObj)
        #objLinVel, objAngVel = self.scene.getVelocity(self.goalObj)
        
        objPos, objOrn = self.scene.getPose("block")
        objLinVel, objAngVel = self.scene.getVelocity("block")

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
        if self.sparse and self.success:
            return True
        #elif self.stepCount >= 100:
            #return True
        #elif self.posError > 0.1 or self.ornError > 0.2:
            # End episode early if error high
            #return True
        elif self.objDropped:
            # End episode early if object dropped
            return True
        return False


    def _isSuccess(self):
        goalDist = self._getGoalDist()
        if goalDist < self.distanceThreshold:
            
            #self.posError < self.distanceThreshold and self.ornError < self.distanceThreshold
            if not self.objGrasped and not self.objDropped:
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
        #if self._isContact():
            #self._collisions += 1
        #print(self._getInfo())
        #self.stepCount += 1

        # Track collisions
        if self._isContact():
            self.collision = True
            self._collisions += 1
        else:
            self.collision = False

        # Track Errors
        blockPos, blockOrn = self.scene.getPose("block")
        
        #self.posError = self._getDist(blockPos, self.goalPos)
        #self.ornError = q.distance(blockOrn, self.goalOrn)
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
        #posChange = self.prevObjPos[2] - blockPos[2]
        self.objDropped = not self.objGrasped
        self.prevObjPos = blockPos

        # Check if successfully placed
        self.success = self._isSuccess()


    # Helper methods
    # --------------------------------------------------------------------------

    def _getGoalDist(self):
        """Gets the distance between the robot's current and goal poses."""
        armPos, _ = self.robot.getPose()
        objPos, _ = self.scene.getPose(self.goalObj)
        goalPos = np.add(objPos, self.goalPosOffset)
        return self._getDist(armPos, self.reachPos)


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
    
    def _getNumFingerTipContacts(self):
        """Get the number of finger tips in contact with the block."""
        
        self.objId = self.scene.getObject("block")[0]
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
