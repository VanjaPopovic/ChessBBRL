import math

import numpy as np

from behaviour_gym.robot2 import robot
from behaviour_gym.utils import quaternion as q


class Ur103f(robot.Robot):

    def __init__(self, physicsClient, restPos=[0.7, 0.0, 0.4],
                 restOrn=[0.5, 0.5, -0.5, 0.5], restGripState=[0.5],
                 maxGripMove=0.025, maxGripForce=2.0, **kwargs):
        """Initialises ur10 arm with an adaptive 3f gripper.

        Args:
            armRestPos ([float]): position in world space [X,Y,Z] of the arm end
                                  effector's rest position.
            armRestOrn ([float]): orientation in world space [X,Y,Z,W] of the
                                  arm end effector's rest orientation.
            gripRestPos (float): position of the gripper joints in the rest
                                 position.
            maxGripMove (float): maximum change in the gripper position in a
                                 single action.
            maxGripForce (float): maximum force exerted by gripper joints.

        """
        # Control variables
        self.maxGripMove = maxGripMove
        self.maxGripForce = maxGripForce

        # IK Null Space
        self.lowerLimits = [-math.pi/2, -math.pi, 0,
                            -math.pi*2, -math.pi*2, -math.pi*2,
                            -0.16, 0.0, 0.0, 0.0,
                            -0.25, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]
        self.upperLimits = [math.pi/2, 0, math.pi,
                            math.pi*2, math.pi*2, math.pi*2,
                            0.25, math.pi, math.pi, math.pi,
                            0.16, math.pi, math.pi, math.pi,
                            math.pi, math.pi, math.pi]
        self.jointRanges = [math.pi, math.pi, math.pi,
                            math.pi*4, math.pi*4, math.pi*4,
                            0.41, math.pi, math.pi, math.pi,
                            0.41, math.pi, math.pi, math.pi,
                            math.pi, math.pi, math.pi]
        self.restPoses = [0.0, -math.pi/4, math.pi/2,
                          0.0, 0.0, 0.0,
                          0.04, math.pi/2, math.pi/2, math.pi/2,
                          -0.05, math.pi/2, math.pi/2, math.pi/2,
                          math.pi/2, math.pi/2, math.pi/2]

        # Joint IDs
        self.armJoints = [1, 2, 3, 4, 5, 6]
        self.gripJoints = [9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
        self.palmFingerJoints = [9, 13]
        self.firstKnuckleJoints = [10, 14, 18]
        self.secondKnuckleJoints = [11, 15, 19]
        self.thirdKnuckleJoints = [12, 16, 20]

        # Link IDs
        self.fingerTipLinks = [12, 16, 20]

        super(Ur103f, self).__init__(
            physicsClient, urdfFile="ur10_3f/ur10_3f.urdf", eeLink=7,
            restPos=restPos, restOrn=restOrn, restGripState=restGripState,
            nGripActions=1, **kwargs
        )


    # Extension methods
    # --------------------------------------------------------------------------

    def applyGripState(self, gripState):
        # Bring fingers together to pinch grasp
        self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                      jointIndex=9,
                                      controlMode=self.p.POSITION_CONTROL,
                                      targetPosition=-0.15,
                                      maxVelocity=0.2,
                                      force=10)
        self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                      jointIndex=13,
                                      controlMode=self.p.POSITION_CONTROL,
                                      targetPosition=0.15,
                                      maxVelocity=1.5,
                                      force=10)

        # Get goal position for thumb first knuckle
        thumbKnucklePos = gripState[0]

        # Set knuckle positions
        for joint in self.firstKnuckleJoints:
            self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                          jointIndex=joint,
                                          controlMode=self.p.POSITION_CONTROL,
                                          targetPosition=thumbKnucklePos,
                                          maxVelocity=1.5,
                                          force=10)

        # Maintain fixed middle joint positions
        for joint in self.secondKnuckleJoints:
            self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                          jointIndex=joint,
                                          controlMode=self.p.POSITION_CONTROL,
                                          targetPosition=0.0,
                                          maxVelocity=1.5,
                                          force=10)

        # Bend third knuckle joints back as first knuckle joints bend
        thirdKnucklePos = 0.85 - (thumbKnucklePos / 0.95) * 0.85

        # Keep fingertips facing flat towards each other
        for joint in self.thirdKnuckleJoints:
            self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                          jointIndex=joint,
                                          controlMode=self.p.POSITION_CONTROL,
                                          targetPosition=thirdKnucklePos,
                                          maxVelocity=1.5,
                                          force=10)


    def applyGripAction(self, gripAction):
        # Get gripper state - the position of thumb first knuckle joint
        thumbKnucklePos = self.getGripState()[0]
        # Clip goal thumb knuckle position between 0 and 0.9 radians
        thumbKnucklePosOffset = gripAction[0] * self.maxGripMove
        goalThumbKnucklePos = np.clip(thumbKnucklePos + thumbKnucklePosOffset,
                                      0, 0.9)

        # Update gripper motors
        self.applyGripState([goalThumbKnucklePos])


    def resetGripState(self, gripState):
        # Reset fingers together for pinch grasp
        self.p.resetJointState(bodyUniqueId=self.id,
                                jointIndex=9,
                                targetValue=-0.16,
                                targetVelocity=0)
        self.p.resetJointState(bodyUniqueId=self.id,
                                jointIndex=13,
                                targetValue=0.16,
                                targetVelocity=0.0)

        thumbKnucklePos = gripState[0]

        # Reset first knuckle position
        for joint in self.firstKnuckleJoints:
            self.p.resetJointState(bodyUniqueId=self.id,
                                    jointIndex=joint,
                                    targetValue=thumbKnucklePos,
                                    targetVelocity=0.0)

        # Reset middle knuckle joint to their fixed positions
        for joint in self.secondKnuckleJoints:
            self.p.resetJointState(bodyUniqueId=self.id,
                                    jointIndex=joint,
                                    targetValue=0.0,
                                    targetVelocity=0.0)

        # Bend fingertips back as knuckle joint position increases
        thirdKnucklePos = 0.85 - (gripState[0] / 0.95) * 0.85

        # Reset third knuckle joints to face fingertips flat together
        for joint in self.thirdKnuckleJoints:
            self.p.resetJointState(bodyUniqueId=self.id,
                                    jointIndex=joint,
                                    targetValue=thirdKnucklePos,
                                    targetVelocity=0.0)


    def calcIK(self, goalPos, goalOrn, restPoses=None, useNull=True):
        if useNull == False:
            jointPoses = self.p.calculateInverseKinematics(self.id,
                                                           self.eeLink,
                                                           goalPos,
                                                           goalOrn,
                                                           maxNumIterations=1000,
                                                           residualThreshold=1e-20)
        elif restPoses is None:
            # If rest poses not provided then use default and higher max iter
            restPoses = self.restPoses
            jointPoses = self.p.calculateInverseKinematics(self.id,
                                                           self.eeLink,
                                                           goalPos,
                                                           goalOrn,
                                                           self.lowerLimits,
                                                           self.upperLimits,
                                                           self.jointRanges,
                                                           restPoses,
                                                           maxNumIterations=500,
                                                           residualThreshold=1e-20)
        else:
            # Otherwise use restposes and lower max iter
            jointPoses = self.p.calculateInverseKinematics(self.id,
                                                           self.eeLink,
                                                           goalPos,
                                                           goalOrn,
                                                           self.lowerLimits,
                                                           self.upperLimits,
                                                           self.jointRanges,
                                                           restPoses,
                                                           maxNumIterations=100,
                                                           residualThreshold=1e-20)

        # Only return arm joints positions
        armJointPoses = jointPoses[0:6]
        return armJointPoses


    def resetArmJoints(self, positions):
        for i in range(6):
            armJoint = self.armJoints[i]
            pos = positions[i]
            self.p.resetJointState(bodyUniqueId=self.id,
                                   jointIndex=armJoint,
                                   targetValue=pos,
                                   targetVelocity=0)


    def setArmMotorsPos(self, positions):
        # positions = np.clip(positions, self.lowerLimits[0:6],
        #                     self.upperLimits[0:6])
        positions = np.clip(positions, -math.pi*2, math.pi*2)

        for i in range(6):
            armJoint = self.armJoints[i]
            pos = positions[i]

            if armJoint in [1,2]:
                self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                             jointIndex=armJoint,
                                             controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=pos,
                                             maxVelocity=2.16,
                                             force=330)
            elif armJoint == 3:
                self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                             jointIndex=armJoint,
                                             controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=pos,
                                             maxVelocity=3.15,
                                             force=150)
            else:
                self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                             jointIndex=armJoint,
                                             controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=pos,
                                             maxVelocity=3.2,
                                             force=56)


    def setArmMotorsVel(self, velocities):
        velocities = np.clip(velocities, -1, 1)

        for i in range(6):
            armJoint = self.armJoints[i]
            vel = velocities[i]

            if armJoint in [1,2]:
                self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                             jointIndex=armJoint,
                                             controlMode=self.p.VELOCITY_CONTROL,
                                             targetVelocity=vel,
                                             maxVelocity=1,
                                             force=330)
            elif armJoint == 3:
                self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                             jointIndex=armJoint,
                                             controlMode=self.p.VELOCITY_CONTROL,
                                             targetVelocity=vel,
                                             maxVelocity=1,
                                             force=150)
            else:
                self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                             jointIndex=armJoint,
                                             controlMode=self.p.VELOCITY_CONTROL,
                                             targetVelocity=vel,
                                             maxVelocity=1,
                                             force=56)


    def setGripMotorsPos(self, positions):
        positions = np.clip(positions, self.lowerLimits[6:], self.upperLimits[6:])

        for i in range(len(self.gripJoints)):
            gripJoint = self.gripJoints[i]
            pos = positions[i]

            self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                         jointIndex=gripJoint,
                                         controlMode=self.p.POSITION_CONTROL,
                                         targetVelocity=0,
                                         targetPosition=pos)


    def setGripMotorsVel(self, velocities):
        velocities = np.clip(velocities, -1, 1)

        for i in range(len(self.gripJoints)):
            gripJoint = self.gripJoints[i]
            vel = velocities[i]

            self.p.setJointMotorControl2(bodyUniqueId=self.id,
                                         jointIndex=gripJoint,
                                         controlMode=self.p.VELOCITY_CONTROL,
                                         targetVelocity=vel,
                                         maxVelocity=1)


    def getGripObs(self):
        gripPos = []
        for i in range(2):
            gripPos.append(self.p.getJointState(self.id, self.palmFingerJoints[i])[0])
        for i in range(3):
            gripPos.append(self.p.getJointState(self.id,
                                                 self.firstKnuckleJoints[i])[0])
            gripPos.append(self.p.getJointState(self.id,
                                                 self.secondKnuckleJoints[i])[0])
            gripPos.append(self.p.getJointState(self.id,
                                                 self.thirdKnuckleJoints[i])[0])
        return gripPos


    def getArmJointsPos(self):
        jointAngles = []
        for armJoint in self.armJoints:
            jointAngles.append(self.p.getJointState(self.id, armJoint)[0])
        return jointAngles


    def getGripJointsPos(self):
        jointAngles = []
        for gripJoint in self.gripJoints:
            jointAngles.append(self.p.getJointState(self.id, gripJoint)[0])
        return jointAngles


    def getArmJointsVel(self):
        jointVels = []
        for armJoint in self.armJoints:
            jointVels.append(self.p.getJointState(self.id, armJoint)[1])
        return jointVels


    def getGripJointsVel(self):
        jointVels = []
        for gripJoint in self.gripJoints:
            jointVels.append(self.p.getJointState(self.id, gripJoint)[1])
        return jointVels


    def getGripState(self):
        linkState = self.p.getJointState(self.id, self.firstKnuckleJoints[2])
        return [linkState[0]]


    def getFingerTipLinks(self):
        return self.fingerTipLinks
