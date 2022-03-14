import os
import math
import pathlib

import pybullet as p
import pybullet_data
import numpy as np

from behaviour_gym.robot import robot


class Ur10HandLite:

    def __init__(self, physicsClient, startPos=[0,0,0.6], startOrn=[0,0,0],
                 maxFingerForce=100.0, eeOrn=[-0.5, -0.5, 0.5, 0.5]):
        self._p = physicsClient
        self.startPos = startPos
        self.startOrn = self._p.getQuaternionFromEuler(startOrn)
        self.maxFingerForce = maxFingerForce
        self.eeOrn = eeOrn
        self.eeOrn = p.getQuaternionFromEuler([math.pi*0.6,0,math.pi/2])

        # Link and Joint IDs
        self.armLinks = [1,2,3,4,5,6]
        self.eeLink = 10
        self.palmFingerJoints = [9, 13]
        self.fingerFirstJoints = [10, 14, 18]
        self.fingerMiddleJoints = [11, 15, 19]
        self.fingerTipJoints = [12, 16, 20]

        # IK Null Space
        self.lowerLimits = [-math.pi/2, -math.pi/2, 0,
                            -math.pi*2, -math.pi*2, -math.pi*2,
                            -0.349065850399, -0.261799387799, 0.0, 0.0,
                            -0.349065850399, -0.261799387799, 0.0, 0.0,
                            -0.349065850399, -0.261799387799, 0.0, 0.0,
                            -1.0471975512, 0.0, -0.698131700798, -0.261799387799]
        self.upperLimits = [math.pi/2, 0, math.pi,
                            math.pi*2, math.pi*2, math.pi*2,
                            0.349065850399, 1.57079632679, 1.57079632679, 1.57079632679,
                            0.349065850399, 1.57079632679, 1.57079632679, 1.57079632679,
                            0.349065850399, 1.57079632679, 1.57079632679, 1.57079632679,
                            1.0471975512, 1.2217304764, 0.698131700798, 1.57079632679]
        self.jointRanges = [math.pi, math.pi/2, math.pi,
                            math.pi*4, math.pi*4, math.pi*4,
                            0.698131700798, 1.832595714589, 1.57079632679, 1.57079632679,
                            0.698131700798, 1.832595714589, 1.57079632679, 1.57079632679,
                            0.698131700798, 1.832595714589, 1.57079632679, 1.57079632679,
                            2.0943951024, 1.2217304764, 1.396263401596, 1.832595714589]
        self.restPoses = [-0.4, -1.0, 0.9,
                          1.6, 1.6, -0.4,
                          0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0]

        # Load Robot
        self._urdfRoot = str(pathlib.Path(__file__).resolve().parent.parent.parent) + "/models/urdf/"
        robotPath = os.path.join(self._urdfRoot, "ur10_hand_lite.urdf")
        self.robotId = self._p.loadURDF(robotPath, self.startPos, self.startOrn)

        _link_name_to_index = {p.getBodyInfo(self.robotId)[0].decode('UTF-8'):-1,}

        for _id in range(p.getNumJoints(self.robotId)):
        	_name = p.getJointInfo(self.robotId, _id)[12].decode('UTF-8')
        	_link_name_to_index[_name] = _id


        for i in range(p.getNumJoints(self.robotId)):
            print(p.getJointInfo(self.robotId, i))
        print(_link_name_to_index)


        self.reset()


    def reset(self):
        pass


    def applyArmPos(self, goalPos):
        # Get IK joint solution
        jointPoses = self._p.calculateInverseKinematics(self.robotId,
                                                        self.eeLink,
                                                        goalPos,
                                                        self.eeOrn,
                                                        self.lowerLimits,
                                                        self.upperLimits,
                                                        self.jointRanges,
                                                        self.restPoses,
                                                        maxNumIterations=50)
        # jointPoses = self._p.calculateInverseKinematics(self.robotId,
        #                                                 self.eeLink,
        #                                                 goalPos,
        #                                                 self.eeOrn,
        #                                                 maxNumIterations=50)

        # Set arm joint positions
        for i in range(6):
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=self.armLinks[i],
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=jointPoses[i],
                                          targetVelocity=0,
                                          force=200)


    def applyGripPos(self, goalPos):
        # Set positions of all gripper joints
        for i in self.palmFingerJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=0,
                                          force=100)

        # Set rest of finger joints
        for i in self.fingerFirstJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=goalPos,
                                          force=100)

        for i in self.fingerMiddleJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=goalPos,
                                          force=100)

        for i in self.fingerTipJoints:
            self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                          jointIndex=i,
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=goalPos,
                                          force=100)


    # def applyGripPos2(self, action):
    #     currArmPos = []
    #     for armJointId in self.armLinks:
    #         jointPos = self._p.getJointInfo(self.robotId, armJointId)
    #         currArmPos.append(jointPos)
    #
    #     lowerLimits = [currArmPos[0], currArmPos[1], currArmPos[2],
    #                    currArmPos[3], currArmPos[4], currArmPos[5],
    #                    -0.16, 0.0, 0.0, 0.0,
    #                    -0.25, 0.0, 0.0, 0.0,
    #                    0.0, 0.0, 0.0]
    #     upperLimits = [currArmPos[0], currArmPos[1], currArmPos[2],
    #                    currArmPos[3], currArmPos[4], currArmPos[5],
    #                    0.25, math.pi, math.pi, math.pi,
    #                    0.16, math.pi, math.pi, math.pi,
    #                    math.pi, math.pi, math.pi]
    #     jointRanges = [0, 0, 0,
    #                    0, 0, 0,
    #                    0.41, math.pi, math.pi, math.pi,
    #                    0.41, math.pi, math.pi, math.pi,
    #                    math.pi, math.pi, math.pi]
    #     restPoses = [currArmPos[0], currArmPos[1], currArmPos[2],
    #                  currArmPos[3], currArmPos[4], currArmPos[5],
    #                  0.04, math.pi/2, math.pi/2, math.pi/2,
    #                  -0.05, math.pi/2, math.pi/2, math.pi/2,
    #                  math.pi/2, math.pi/2, math.pi/2]
    #
    #     action = np.clip(action, 0, 1) * 0.1
    #     currFingerTipPos = self.getFingerTipPos()[0]
    #     goalFingerTipPos = np.array(currFingerTipPos) + action
    #
    #     # Get IK joint solution
    #     jointPoses = self._p.calculateInverseKinematics(self.robotId,
    #                                                     12,
    #                                                     goalFingerTipPos,
    #                                                     lowerLimits,
    #                                                     upperLimits,
    #                                                     jointRanges,
    #                                                     restPoses,
    #                                                     maxNumIterations=50)
    #
    #     print("jointposes:", jointPoses)
    #
    #     self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
    #                                   jointIndex=9,
    #                                   controlMode=p.POSITION_CONTROL,
    #                                   targetPosition=jointPoses[6],
    #                                   force=100)
    #     self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
    #                                   jointIndex=10,
    #                                   controlMode=p.POSITION_CONTROL,
    #                                   targetPosition=jointPoses[7],
    #                                   force=100)
    #     self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
    #                                   jointIndex=11,
    #                                   controlMode=p.POSITION_CONTROL,
    #                                   targetPosition=jointPoses[8],
    #                                   force=100)
    #     self._p.setJointMotorControl2(bodyUniqueId=self.robotId,
    #                                   jointIndex=12,
    #                                   controlMode=p.POSITION_CONTROL,
    #                                   targetPosition=jointPoses[9],
    #                                   force=100)
    #
    #     for i in range(20):
    #         self._p.stepSimulation()
    #         time.sleep(1./7.)
    #
    #     print("goalFingerTipPos: ", goalFingerTipPos)
    #     print("actual: ", self.getFingerTipPos()[0])


    def getObservation(self):
        eeLinkState = self._p.getLinkState(self.robotId, self.eeLink, computeLinkVelocity=True)
        eeLinkPos = eeLinkState[0]
        eeLinkOrn = eeLinkState[1]
        eeLinkLin = eeLinkState[6]
        eeLinkAng = eeLinkState[7]
        gripperPos = self.getGripperPos()
        return eeLinkPos, eeLinkOrn, eeLinkLin, eeLinkAng, gripperPos


    def getArmPos(self):
        state = self._p.getLinkState(self.robotId, self.eeLink)
        return state[0]


    def getGripperPos(self):
        pos = []
        for i in range(3):
            pos.append(self._p.getJointState(self.robotId, self.fingerFirstJoints[i])[0])
            pos.append(self._p.getJointState(self.robotId, self.fingerMiddleJoints[i])[0])
            pos.append(self._p.getJointState(self.robotId, self.fingerTipJoints[i])[0])
        return pos


    def getFingerTipPos(self):
        pos = []
        for i in range(3):
            [pos.append(self._p.getLinkState(self.robotId, self.fingerTipJoints[i])[0])]
        return pos
