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


class Vanja(scene.Scene):
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
        super(Vanja, self).__init__(*sceneArgs, **sceneKwargs)
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
        self.blockId = self.p.loadURDF(path, [0.6, 0.3, 0.64],
                                       [0.0, 0.0, 1.0, 1.0])
                                       
        # Load lever
        path = os.path.join(self.objectModels, "lever/lever.sdf")
        self.leverId = self.p.loadSDF(path)
        leverOrn = [0, 0, math.pi/2]
        leverOrn = self.p.getQuaternionFromEuler(leverOrn)
        self.p.resetBasePositionAndOrientation(self.leverId[0],[0.6,0.0,0.625], leverOrn)
        
        path = os.path.join(self.objectModels, "button/button.sdf")
        self.buttonId = self.p.loadSDF(path)
        self.p.resetBasePositionAndOrientation(self.buttonId[0],[0.6,-0.3,0.64],[0.0,0.0,0.0,1.0])
        # Return dictionary of object names to model and link indices
        objects = {}
        objects["block"] = [self.blockId, -1]
        objects["lever"] = [self.leverId[0], -1]
        objects["button"] = [self.buttonId[0], -1]

        return objects


    def _randomise(self):
        return
        # Get a random starting pose for the block
        #blockPos = [random.uniform(self.blockPosMinX, self.blockPosMaxX),
                    #random.uniform(self.blockPosMinY, self.blockPosMaxY),
                    #0.64]
        #yOrn = 0
        # if random.random() > 0.5:
        #     yOrn = math.pi/2
        # blockOrn = [0, yOrn, random.uniform(0, math.pi*2)]
        #blockOrn = [0, 0, math.pi/2]
        #blockOrn = self.p.getQuaternionFromEuler(blockOrn)

        # Reset block to the random pose
        
        #self.p.resetBasePositionAndOrientation(self.blockId, blockPos, blockOrn)
        #self.p.resetBasePositionAndOrientation(self.leverId[0], blockPos, blockOrn)
        
        #self.p.resetBasePositionAndOrientation(self.buttonId[0], blockPos, blockOrn)
        
