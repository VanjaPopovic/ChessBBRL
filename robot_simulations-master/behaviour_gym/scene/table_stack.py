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
from behaviour_gym.utils import quaternion as q


class TableStack(scene.Scene):
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
        super(TableStack, self).__init__(*sceneArgs, **sceneKwargs)


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
        ornFlat = [0,0,0,1]
        ornFlatTwist = q.rotateGlobal(ornFlat, 0, 0, math.pi/4)
        ornFlatNegTwist = q.rotateGlobal(ornFlat, 0, 0, -math.pi/4)
        ornFlat90 = q.rotateGlobal(ornFlat, 0, 0, math.pi/2)
        ornUp = q.rotateGlobal(ornFlat, 0, math.pi/2, 0)
        ornUpTwist = q.rotateGlobal(ornUp, 0, 0, math.pi/4)
        self.block1Id = self.p.loadURDF(path, [0.4, -0.4, 0.68],
                                        ornUp)
        self.block2Id = self.p.loadURDF(path, [0.4, -0.2, 0.64],
                                        ornFlat)
        self.block3Id = self.p.loadURDF(path, [0.4, 0.0, 0.64],
                                        ornFlatTwist)
        self.block4Id = self.p.loadURDF(path, [0.4, 0.2, 0.68],
                                        ornUpTwist)
        self.block5Id = self.p.loadURDF(path, [0.6, -0.4, 0.64],
                                        ornFlat90)
        self.block6Id = self.p.loadURDF(path, [0.6, -0.2, 0.64],
                                        ornFlatTwist)
        self.block7Id = self.p.loadURDF(path, [0.6, 0.0, 0.64],
                                        ornFlatNegTwist)
        self.block8Id = self.p.loadURDF(path, [0.6, 0.2, 0.64],
                                        ornFlat)

        # Return dictionary of object names to model and link indices
        objects = {}
        objects["block1"] = [self.block1Id, -1]
        objects["block2"] = [self.block2Id, -1]
        objects["block3"] = [self.block3Id, -1]
        objects["block4"] = [self.block4Id, -1]
        objects["block5"] = [self.block5Id, -1]
        objects["block6"] = [self.block6Id, -1]
        objects["block7"] = [self.block7Id, -1]
        objects["block8"] = [self.block8Id, -1]

        return objects


    def _randomise(self):
        pass
