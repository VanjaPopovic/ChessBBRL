import time
import math

import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t
from behaviour_gym.scene import ChessScene

scene = Chess(guiMode=True)
robot = scene.loadRobot("ur103f")
robot.resetBasePose(pos=[0,0,0.6])
# camera = scene.loadCamera("myntd100050", restPos=[0.5, 1.2, 1.8],
#                           restPan=math.pi, restTilt=-0.85)
camera = None
scene.setInit()
scene.start()
startPos, startOrn = robot.getPose()
# camera.getRgbd()


time.sleep(5)
while true:
    print("hello")

