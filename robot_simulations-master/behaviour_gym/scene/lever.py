import os
import random
import time
import math

import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import gym
from gym.utils import seeding

from behaviour_gym.robot import utils
from behaviour_gym.scene import scene


class Lever(scene.Scene):
    """Base class for all gym environments."""

    def __init__(self, blockPosRange=[[0.4,0.95],[-0.4,0.4]], *sceneArgs,
                 **sceneKwargs):
        """Initialises the table environment.

        Args:
            blockPosRange [float]: range for randomising the block's position.
                                   [[xMin,xMax], [yMin,yMax]]
            sceneArgs: arguments passed to the Scene constructor.
            sceneKwargs: keyword arguments passed to the Scene constructor.

        """
        super(Lever, self).__init__(*sceneArgs, **sceneKwargs)
        self.blockPosMinX = blockPosRange[0][0]
        self.blockPosMaxX = blockPosRange[0][1]
        self.blockPosMinY = blockPosRange[1][0]
        self.blockPosMaxY = blockPosRange[1][1]


    # Scene extension methods
    # --------------------------------------------------------------------------

    def _loadScene(self):
        # Load ground plane
        path = os.path.join(self.pybulletModels, "plane.urdf")
        self.p.loadURDF(path)

        # Load table
        path = os.path.join(self.objectModels, "table/table.urdf")
        tableId = self.p.loadURDF(path, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        # Load block
        path = os.path.join(self.objectModels, "block/block.urdf")
        #self.blockId = self.p.loadURDF(path, [0.6, 0.0, 0.64],
         #                              [0.0, 0.0, 0.0, 1.0])
        path = os.path.join(self.objectModels, "lever/lever.sdf")
        self.buttonId = self.p.loadSDF(path)

        #print(self.buttonId[0])
        #print(var)

        #self.p.resetBasePositionAndOrientation(self.buttonId[0],[0.6,0.0,0.64],[0.0,0.0,0.0,1.0])
        objects = {}
        objects["lever"] = [self.buttonId[0], -1]

        return objects


    def _randomise(self):
        # Get a random starting pose for the block
        blockPos = [random.uniform(self.blockPosMinX, self.blockPosMaxX),
                    random.uniform(self.blockPosMinY, self.blockPosMaxY),
                    0.64]
        yOrn = 0
        # if random.random() > 0.5:
        #     yOrn = math.pi/2
        # blockOrn = [0, yOrn, random.uniform(0, math.pi*2)]
        blockOrn = [0, 0, math.pi/2]
        blockOrn = self.p.getQuaternionFromEuler(blockOrn)

        # Reset block to the random pose
        self.p.resetBasePositionAndOrientation(self.buttonId[0], blockPos, blockOrn)
