import behaviour_gym
from behaviour_gym.utils import factory
from behaviour_gym.utils import quaternion as q

import gym
import math
import time


env = gym.make("Grasp-Ur103f-v0", renders=True, timestep=1./30.)
# _ = env.reset()
camera1 = factory.CameraFactory().createCamera("myntd100050", physicsClient=env.env._p, cameraStartPos=[0.5, 1.2, 1.8], startPan=math.pi, startTilt=-0.85)
_ = camera1.getRgbd()
_ = camera.getRgbd()
for i in range(5):
    camera.moveRelative(0,-0.01,0,0,0)
    _ = camera.getRgbd()

env.render(mode="rgb_array")
initPos = env.robot.getArmPos()

for i in range(5):
    startPos = env.robot.getArmPos()
    env.robot.applyArmPoseRelative(0,0,0, 0, 0, 0)
    env.robot.applyGripAction(1)
    for i in range(100):
        env.env._p.stepSimulation()
        time.sleep(1./30.)
    print("Start Pos:", startPos)
    armPos = env.robot.getArmPos()
    print("End Pos:", armPos)
    diff = (abs(startPos[0] - armPos[0]), abs(startPos[1] - armPos[1]), abs(startPos[2] - armPos[2]))
    print("Difference:", diff)
    print("")

totalDrift = (abs(initPos[0] - armPos[0]), abs(initPos[1] - armPos[1]), abs(initPos[2] - armPos[2]))
print("Total Diff:", diff)

env.close()
env = gym.make("Grasp-Ur103f-v0", renders=True, timestep=1./1000.)
initPos = env.robot.getArmPos()

for i in range(5):
    startPos, startOrn = env.robot.getArmPose()
    # goalOrn = q.rotateQuaternion(startOrn, 0, 0, math.pi/12)
    goalOrn = startOrn
    env.robot.applyArmPose(startPos, goalOrn)
    for i in range(100):
        env.env._p.stepSimulation()
    print("Start Pos:", startPos)
    armPos = env.robot.getArmPos()
    print("End Pos:", armPos)
    diff = (abs(startPos[0] - armPos[0]), abs(startPos[1] - armPos[1]), abs(startPos[2] - armPos[2]))
    print("Difference:", diff)
    print("")

totalDrift = (abs(initPos[0] - armPos[0]), abs(initPos[1] - armPos[1]), abs(initPos[2] - armPos[2]))
print("Total Diff:", diff)

time.sleep(5)
