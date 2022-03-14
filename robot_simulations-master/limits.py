import time
import math

import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t
from behaviour_gym.scene import Table
from behaviour_gym.primitive import GraspEasy as Reach

scene = Table(guiMode=True)
robot = scene.loadRobot("ur103f")
robot.resetBasePose(pos=[0,0,0.6])
# robot.resetBasePose([0,0,1.2], [ 0.3662725, 0, 0, 0.9305076 ])
scene.setInit()

reach = Reach("block", [0,0,0.01], scene=scene, robot=robot, timestep=1./30.)
reach.reset()
reach.step([0,0,0,0])

def printObs(obs):
    print("Robot Pos: ", obs[0:3])
    print("Robot Orn: ", obs[3:7])
    print("Robot Lin Vel: ", obs[7:10])
    print("Robot Ang Vel: ", obs[10:13])
    print("Robot Gripper: ", obs[13:24])
    print("Block Pos: ", obs[24:27])
    print("Block Orn: ", obs[27:31])
    print("Block Lin Vel: ", obs[31:34])
    print("Block Ang Vel: ", obs[34:37])
    print("Block-Robot Pos: ", obs[37:40])

# def step(action):
#     obs, reward, done, info = reach.step(action)
#     print("Obs")
#     printObs(obs)
#     print("Reward: ", reward)
#     print("Done: ", done)
#     print("Info: ", info)
_, startOrn = robot.getPose()
def step(action):
    pos, _ = robot.getPose()
    goalPos = np.add(pos, action)
    robot.applyPose(goalPos, startOrn, relative=True)
    scene.step(80, timestep=1./80.)

# def reset():
#     obs = reach.reset()
#     print("Obs")
#     printObs(obs)
def reset():
    scene.reset()
    scene.start()

reset()
step([0,0,-0.05])

step([1,1,1])

reach.reset()
reach.step([1,1,1])

pos, orn = robot.getPose()

poseMatrix = t.getTransformationMatrix(pos, orn)

basePos, baseOrn = robot.getBasePose()
baseMatrix = t.getTransformationMatrix(basePos, baseOrn)

inverse = np.linalg.inv(baseMatrix)
basePosInv, baseOrnInv = t.getPose(inverse)

poseMatrixBase = np.dot(inverse, poseMatrix)
posBase, ornBase = t.getPose(poseMatrixBase)

poseMatrix = np.dot(baseMatrix, poseMatrixBase)
finPos, finOrn = t.getPose(poseMatrix, orn)

def _getDist(startPos, goalPos):
    """Return distance from startPos to goalPos."""
    startPos = np.array(startPos)
    goalPos = np.array(goalPos)
    squared_dist = np.sum((goalPos-startPos)**2, axis=0)
    return np.sqrt(squared_dist)


pyPosBase, pyOrnBase = scene.p.multiplyTransforms(*scene.p.invertTransform(basePos, baseOrn), pos, orn)
pyPosWorld, pyOrnWorld = scene.p.multiplyTransforms(basePos, baseOrn, pyPosBase, pyOrnBase)

accPyPos = _getDist(pyPosWorld, pos)
accPyOrn = q.distance(pyOrnBase, orn)

accPos = _getDist(finPos, pos)
accOrn = q.distance(finOrn, orn)

print()
print("Input - ", pos, orn)
print()
print("EE in Base Frame")
print("PyBullet - ", pyPosBase, pyOrnBase)
print("Mine     - ", posBase, ornBase)
print()
print("EE Final")
print("PyBullet - ", pyPosWorld, pyOrnWorld)
print("Mine - ", finPos, finOrn)

print()
print("Accuracy")
print("         PyBullet Pos - ", accPyPos)
print("         Mine     Pos - ", accPos)
print("         PyBullet Orn - ", accPyOrn)
print("         Mine     Orn - ", accOrn)

goalPos = [pos[0] + 0.1, pos[1], pos[2]]
robot.applyPose(goalPos, orn)
scene.step(10)
linVel, angVel = robot.getVelocity()
vel = [*linVel, *angVel]
print("Orginal Vel - ", vel)

matrix = np.zeros((6,6))
rot = q.getRotationMatrix(baseOrnInv)
matrix[0:3,0:3] = rot
matrix[3:6,3:6] = rot
print("Matrix")
print(matrix)

print("Velocity in Base")
velInBase = np.dot(matrix, vel)
print(velInBase)

matrix = np.zeros((6,6))
rot = q.getRotationMatrix(baseOrn)
matrix[0:3,0:3] = rot
matrix[3:6,3:6] = rot
print("Matrix")
print(matrix)

print("Velocity back in World")
velInWorld = np.dot(matrix, velInBase)
print(velInWorld)


# posV = [pos[0], pos[1], pos[2], 1]
# print(posV)
#
# np.dot(inverse, posV)
#
#
#
# initPos, initOrn = robot.getPose()
# posBase, ornBase= robot.worldToBase(initPos, initOrn)
# pos, orn = robot.baseToWorld(posBase, ornBase)
#
# print("Started: ", initPos, initOrn)
# print("In Base Frame: ", posBase, ornBase)
# print("Back to World: ", pos, orn)
# camera = scene.loadCamera("myntd100050", startPos=[0.5, 1.2, 1.8], startPan=math.pi, startTilt=-0.85)
# scene.setInit()
#
# reach = Reach("block", [0,0,0.01], scene=scene, robot=robot,
#               camera=camera)
#
# reach.reset()
# scene.reset(random=False)
# robot.applyArmPose(robot.armRestPos, robot.armRestOrn)
# scene.start()
#
# def eulDif():
#     pos, orn = robot.getArmPose()
#     difq = scene.p.getDifferenceQuaternion(robot.armRestOrn, orn)
#     difqEul = scene.p.getEulerFromQuaternion(difq)
#     return difqEul
#
# def step(action, steps=1, sec=2):
#     for i in range(steps):
#         _,_,_,_=reach.step(action)
#         time.sleep(sec/steps)
#     return eulDif()
#
# print(eulDif())
# print(step([1,1,1,1,1,1], steps=1, sec=1))
# print(step([-1,-1,-1], steps=1, sec=1))
#
# scene2 = Table(guiMode=False)
# robot2 = scene2.loadRobot("ur103f", startPos=[0,0,0.6], armRestOrn=[0.5, 0.5, -0.5, 0.5])
# camera2 = scene2.loadCamera("myntd100050", startPos=[0.5, 1.2, 1.8], startPan=math.pi, startTilt=-0.85)
# scene2.setInit()
#
# reach2 = Reach("block", [0,0,0.01], physicsSteps=100, scene=scene2, robot=robot2,
#               camera=camera2)
#
# reach2.reset()
# scene2.reset(random=False)
# robot2.applyArmPose(robot.armRestPos, robo.armRestOrn)
# scene2.start()
#
# def state():
#     for i in range(7):
#         state = scene.p.getJointState(robot.robotId, i)
#         print(state[0])
#
# def eul():
#     pos, orn = robot.getArmPose()
#     ornEul = scene.p.getEulerFromQuaternion(orn)
#     return ornEul
#
# def eulDif():
#     pos, orn = robot.getArmPose()
#     difq = scene.p.getDifferenceQuaternion(robot.armRestOrn, orn)
#     difqEul = scene.p.getEulerFromQuaternion(difq)
#     return difqEul
#
# def step(action):
#     for i in range(10):
#         _,_,_,_ = reach.step([0,0,0,*action])
#         time.sleep(1./5.)
#     return robot.getArmPose(), eulDif(), state()
#
#
# def state2():
#     for i in range(7):
#         state = scene.p.getJointState(robot.robotId, i)
#         print(state[0])
#
# def eul2():
#     pos, orn = robot2.getArmPose()
#     ornEul = scene2.p.getEulerFromQuaternion(orn)
#     return ornEul
#
# def eulDif2():
#     pos, orn = robot2.getArmPose()
#     difq = scene2.p.getDifferenceQuaternion(robot2.armRestOrn, orn)
#     difqEul = scene2.p.getEulerFromQuaternion(difq)
#     return difqEul
#
# def step2(action):
#     for i in range(9):
#         _,_,_,_ = reach2.step([0,0,0,*action])
#         time.sleep(1./5.)
#     return robot2.getArmPose(), eulDif2(), state2()
#
# pose, dif, state = step([1,0,0])
# print(dif)
# pose, dif, state = step([0,1,0])
# print(dif)
# pose, dif, state = step([0,0,1])
# print(dif)
#
# pose, dif, state = step2([1,1,1])
# print(dif)
#
# reach2.render(mode="Human")
#
# for i in range(9):
#     _,_,_,_ = reach.step([0,0,0,1,0,0])
#     time.sleep(1./5.)
#
# print(robot.getArmPose())
# print("eulDif: ", eulDif())
#
# for i in range(5):
#     _,_,_,_ = reach.step([0,0,0,0,1,0])
#     time.sleep(1./5.)
#
# print(robot.getArmPose())
# print("eulDif: ", eulDif())
#
# for i in range(5):
#     _,_,_,_ = reach.step([0,0,0,0,0,1])
#     time.sleep(1./5.)
#
# print(robot.getArmPose())
# print("eulDif: ", eulDif())
#
#
# # for i in range(5):
# #     _,_,_,_ = reach.step([0,0,0,0,0,-1])
# #     time.sleep(1./5.)
# #
# # print(robot.getArmPose())
# #
# # for i in range(5):
# #     _,_,_,_ = reach.step([0,0,0,0,-1,0])
# #     time.sleep(1./5.)
# #
# # print(robot.getArmPose())
# #
# # for i in range(5):
# #     _,_,_,_ = reach.step([0,0,0,-1,0,0])
# #     time.sleep(1./5.)
#
# # print(robot.getArmPose())
# # print("eulDif: ", eulDif())
#
# for i in range(5):
#     _,_,_,_ = reach.step([0,0,0,-1,0,0])
#     time.sleep(1./5.)
#
# print(robot.getArmPose())
# print("eulDif: ", eulDif())
#
# for i in range(5):
#     _,_,_,_ = reach.step([0,0,0,0,-1,0])
#     time.sleep(1./5.)
#
# print(robot.getArmPose())
# print("eulDif: ", eulDif())
#
# for i in range(5):
#     _,_,_,_ = reach.step([0,0,0,0,0,-1])
#     time.sleep(1./5.)
#
# print(robot.getArmPose())
# print("eulDif: ", eulDif())
# time.sleep(2)
#
# reach.reset()
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,1,1,1])
#     time.sleep(1./10.)
#
# print(robot.getArmPose())
# time.sleep(2)
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,-1])
#     time.sleep(1./10.)
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,-1,0])
#     time.sleep(1./10.)
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,-1,0,0])
#     time.sleep(1./10.)
#
# print(robot.getArmPose())
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,1])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,-1,0,0])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,-1])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,1,0,0])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,-1])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,0])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,1])
#     time.sleep(1./10.)
#
# print("euler: ", eul())
# print("eulDif: ", eulDif())
#
#
#
#
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,1,0,0])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(20):
#     _,_,_,_ = reach.step([0,0,0,-1,0,0])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,1,0,0])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,-1,0])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,-1,0])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(20):
#     _,_,_,_ = reach.step([0,0,0,0,1,0])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,1])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(20):
#     _,_,_,_ = reach.step([0,0,0,0,0,-1])
#
# print("restOrn, orn: ", eulDif())
#
# for i in range(10):
#     _,_,_,_ = reach.step([0,0,0,0,0,1])
#
# print("restOrn, orn: ", eulDif())
#
# scene.reset(random=False)
# robot.applyArmPoseRelative(0,0,0,0,0,0)
# scene.start()
# armPos, armOrn = robot.getArmPose()
# goalOrn = q.rotateQuaternion(armOrn, -1.57, 0, 0)
# eulGoalOrn = scene.p.getEulerFromQuaternion(goalOrn)
# eulOrn = scene.p.getEulerFromQuaternion(armOrn)
# robot.applyArmPoseRelative(0,0,0,-1.57,0,0)
# scene.step(480, timestep=1./120.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
# print("Goal Euler: ", eulGoalOrn)
#
# scene.reset(random=False)
# robot.applyArmPoseRelative(0,0,0,0,0,0)
# scene.start()
# armPos, armOrn = robot.getArmPose()
# goalOrn = q.rotateQuaternion(armOrn, 0, -1.57, 0)
# eulGoalOrn = scene.p.getEulerFromQuaternion(goalOrn)
# eulOrn = scene.p.getEulerFromQuaternion(armOrn)
# robot.applyArmPoseRelative(0,0,0,0,-1.57,0)
# scene.step(480, timestep=1./120.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
# print("Goal Euler: ", eulGoalOrn)
#
# scene.reset(random=False)
# robot.applyArmPoseRelative(0,0,0,0,0,0)
# scene.start()
# armPos, armOrn = robot.getArmPose()
# goalOrn = q.rotateQuaternion(armOrn, 0.0, 0.0, -1.57)
# eulGoalOrn = scene.p.getEulerFromQuaternion(goalOrn)
# eulOrn = scene.p.getEulerFromQuaternion(armOrn)
# robot.applyArmPoseRelative(0,0,0,0,0,-1.57)
# scene.step(480, timestep=1./120.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
# print("Goal Euler: ", eulGoalOrn)
#
# scene.reset(random=False)
# robot.applyArmPoseRelative(0,0,0,0,0,0)
# scene.start()
# armPos, armOrn = robot.getArmPose()
# eulOrn = scene.p.getEulerFromQuaternion(armOrn)
# robot.applyArmPoseRelative(0,0,0,0,-0.2,0)
# scene.step(480, timestep=1./120.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
#
# scene.reset(random=False)
# robot.applyArmPoseRelative(0,0,0,0,0,0)
# scene.start()
# armPos, armOrn = robot.getArmPose()
# eulOrn = scene.p.getEulerFromQuaternion(armOrn)
# robot.applyArmPoseRelative(0,0,0,0,0,-0.2)
# scene.step(480, timestep=1./120.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
#
#
# # reach = Reach("block", [0,0,0.01], physicsSteps=100, scene=scene, robot=robot,
# #               camera=camera)
# # reach.reset()
#
# scene.reset(random=False)
# scene.start()
# pos, orn = robot.getArmPose()
# eulOrn = scene.p.getEulerFromQuaternion(orn)
# robot.applyArmPoseRelative(0,0,0,0.5,0,0)
# scene.step(480, timestep=1./240.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
#
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
#
# scene.reset(random=False)
# scene.start()
# pos, orn = robot.getArmPose()
# eulOrn = scene.p.getEulerFromQuaternion(orn)
# robot.applyArmPoseRelative(0,0,0,0,-0.5,0)
# scene.step(480, timestep=1./240.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
#
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
#
# scene.reset(random=False)
# scene.start()
# pos, orn = robot.getArmPose()
# eulOrn = scene.p.getEulerFromQuaternion(orn)
# robot.applyArmPoseRelative(0,0,0,0,0,-0.5)
# scene.step(480, timestep=1./240.)
# newPos, newOrn = robot.getArmPose()
# newEulOrn = scene.p.getEulerFromQuaternion(newOrn)
#
# print("Old Euler: ", eulOrn)
# print("New Euler: ", newEulOrn)
#
# # reach.render("human")
#
# reach.step([0,-1,0])
# reach.render("human")
#
#
# scene.start()
# scene.reset(random=True)
# scene.reset(random=False)
# # robot = scene.setup("ur103f", startPos=[0,0,0.6])
