import math

import numpy as np

from behaviour_gym.robot import robot
from behaviour_gym.utils import quaternion as q

class BBRLRobot(robot.Robot):

    def __init__(self, armRestPos=[0.7, 0.0, 0.5],
                 armRestOrn=[-0.5, 0.5, 0.5, 0.5], gripRestPos=0.3,
                 maxGripMove=0.1, maxGripForce=5.0, **kwargs):
        """
        Initialises ur10 arm with an adaptive 3f gripper.

        Args:
            armRestPos ([float]): position relative to the base link in world
                                  space [X,Y,Z] of the arm end effector's rest
                                  position
            armRestOrn ([float]): orientation in world space [X,Y,Z,W] of the
                                  arm end effector's rest orientation
            gripRestPos (float): position of the gripper joints in the rest
                                 position
            maxGripMove (float): maximum change in the gripper position in a
                                 single action
            maxGripForce (float): maximum force exerted by gripper joints
        """
        super(BBRLRobot, self).__init__(
            urdfFile="ur10_3f/ur10_3f.urdf", nObs=10, nGripActions=2, **kwargs
        )

        # Control variables
        self.armRestPos = np.add(self.startPos, armRestPos)
        self.armRestOrn = armRestOrn
        self.gripRestPos = gripRestPos
        self.maxGripMove = maxGripMove
        self.maxGripForce = maxGripForce

        # IK Null Space
        self.lowerLimits = [-math.pi/2, -math.pi/2, 0,
                            -math.pi*2, -math.pi*2, -math.pi*2,
                            -0.16, 0.0, 0.0, 0.0,
                            -0.25, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]
        self.upperLimits = [math.pi/2, 0, math.pi,
                            math.pi*2, math.pi*2, math.pi*2,
                            0.25, math.pi, math.pi, math.pi,
                            0.16, math.pi, math.pi, math.pi,
                            math.pi, math.pi, math.pi]
        self.jointRanges = [math.pi, math.pi/2, math.pi,
                            math.pi*4, math.pi*4, math.pi*4,
                            0.41, math.pi, math.pi, math.pi,
                            0.41, math.pi, math.pi, math.pi,
                            math.pi, math.pi, math.pi]
        self.restPoses = [-0.4, -1.0, 0.9,
                          1.6, 1.6, -0.4,
                          0.04, math.pi, math.pi, math.pi,
                          -0.05, math.pi, math.pi, math.pi,
                          math.pi, math.pi, math.pi]

        # Link and Joint IDs
        self.armLinks = [1,2,3,4,5,6]
        self.eeLink = 7
        self.palmFingerJoints = [9, 13]
        self.fingerFirstJoints = [10, 14, 18]
        self.fingerMiddleJoints = [11, 15, 19]
        self.fingerTipJoints = [12, 16, 20]

    # Robot extension methods
    # --------------------------------------------------------------------------

    def applyArmPose(self, goalPos, goalOrn):
        # Get IK joint solution
        jointPoses = self._p.calculateInverseKinematics(self.robotId,
                                                        self.eeLink,
                                                        goalPos,
                                                        goalOrn,
                                                        self.lowerLimits,
                                                        self.upperLimits,
                                                        self.jointRanges,
                                                        self.restPoses,
                                                        maxNumIterations=1000,
                                                        residualThreshold=1e-20)

        # Set arm joint positions
        for i in range(6):
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=self.armLinks[i],
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPosition=jointPoses[i],
                                          targetVelocity=0,
                                          maxVelocity=1.5,
                                          force=200)

    def applyArmPoseRelative(self, goalPos, angle):
        # Get goal pose reletive to current pose
        _, orn = self.getArmPose()
        goalOrn = q.rotateQuaternion(orn, 0, 0, angle)

        # Use current joint positions as rest poses for IK null space
        rp = []
        for i in range(self._p.getNumJoints(self.robotId)):
            info = self._p.getJointInfo(self.robotId, i)
            if info[2] != 4:
                state = self._p.getJointState(self.robotId, i)
                rp.append(state[0])

        # Get IK joint solution
        jointPoses = self._p.calculateInverseKinematics(self.robotId,
                                                        self.eeLink,
                                                        goalPos,
                                                        goalOrn,
                                                        self.lowerLimits,
                                                        self.upperLimits,
                                                        self.jointRanges,
                                                        rp,
                                                        maxNumIterations=1000,
                                                        residualThreshold=1e-20)

        # Set arm joint positions
        for i in range(6):
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=self.armLinks[i],
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPosition=jointPoses[i],
                                          targetVelocity=0,
                                          maxVelocity=1.5,
                                          force=200)


    def applyArmRestPose(self):
        self.applyArmPose(self.armRestPos, self.armRestOrn)

    def applyGripAction(self, action):
        gripPos = self.getGripPos()[0]
        gripPosOffset = action * self.maxGripMove
        goalGripPos = np.clip(gripPos + gripPosOffset, 0, 0.9)
        self.applyGripPos(goalGripPos)
        
    def applyGripRestPos(self):
        self.applyGripPos(self.gripRestPos)

    def getObs(self):
        # Arm kinematics
        eeLinkState = self._p.getLinkState(self.robotId, self.eeLink,
                                           computeLinkVelocity=True)
        eeLinkPos = eeLinkState[0]
        eeLinkOrn = eeLinkState[1]
        eeLinkLin = eeLinkState[6]
        eeLinkAng = eeLinkState[7]

        return np.concatenate((eeLinkPos, eeLinkOrn, eeLinkLin, eeLinkAng))

    def getArmPos(self):
        eeLinkState = self._p.getLinkState(self.robotId, self.eeLink)
        return eeLinkState[0]

    def getFingerTipPos(self):
        pos = []
        for i in range(3):
            pos.append(self._p.getLinkState(self.robotId, self.fingerTipJoints[i])[0])
        return pos

    def getFingerTipLinks(self):
        return self.fingerTipJoints

    # Helper methods
    # -------------------------------------------------------------------------
    def getArmPose(self):
        eeLinkState = self._p.getLinkState(self.robotId, self.eeLink)
        return eeLinkState[0], eeLinkState[1]

    def applyGripPos(self, goalPos):
        # Adjust fingertip position based on knuckle position
        fingerMax = 0.85
        fingerTip = fingerMax - (goalPos / 0.95)*fingerMax

        # Bring fingers together to pinch grasp
        self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                      jointIndex=9,
                                      controlMode=self._p.POSITION_CONTROL,
                                      targetPosition=-0.16,
                                      force=self.maxGripForce)

        self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                      jointIndex=13,
                                      controlMode=self._p.POSITION_CONTROL,
                                      targetPosition=0.25,
                                      force=self.maxGripForce)

        # Set knuckle position
        for i in self.fingerFirstJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPosition=goalPos,
                                          force=self.maxGripForce)

        # Maintain fixed middle joint
        for i in self.fingerMiddleJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPosition=0.0,
                                          force=self.maxGripForce)

        # Keep fingertips facing flat
        for i in self.fingerTipJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPosition=fingerTip,
                                          force=self.maxGripForce)

    def getGripPos(self):
        gripPos = []
        for i in range(3):
            gripPos.append(self._p.getJointState(self.robotId,
                                                 self.fingerFirstJoints[i])[0])
            gripPos.append(self._p.getJointState(self.robotId,
                                                 self.fingerMiddleJoints[i])[0])
            gripPos.append(self._p.getJointState(self.robotId,
                                                 self.fingerTipJoints[i])[0])
        return gripPos

    def rotate(self, action):
        eeOrn = self._p.getLinkState(self.robotId, self.eeLink,
                                           computeLinkVelocity=True)[1]

        # euler_angles = list(self._p.getEulerFromQuaternion(eeOrn))

        goalOrn = [0,0,2,0]
         # Get IK joint solution
        jointPoses = self._p.calculateInverseKinematics(self.robotId,
                                                        self.eeLink,
                                                        self.getArmPos(),
                                                        goalOrn,
                                                        self.lowerLimits,
                                                        self.upperLimits,
                                                        self.jointRanges,
                                                        self.restPoses,
                                                        maxNumIterations=1000,
                                                        residualThreshold=1e-9)
        # Set arm joint 6 position to corresponding joint pose only need
        # here so that end-effector only rotates
        self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                        jointIndex=self.armLinks[5],
                                        controlMode=self._p.POSITION_CONTROL,
                                        targetPosition=jointPoses[5],
                                        targetVelocity=0,
                                        force=200)