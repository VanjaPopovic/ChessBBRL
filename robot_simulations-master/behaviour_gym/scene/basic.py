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


class Basic(scene.Scene):
    """Base class for all gym environments."""

    def __init__(self, *sceneArgs, **sceneKwargs):
        """Initialises the table environment.

        Args:
            sceneArgs: arguments passed to the Scene constructor.
            sceneKwargs: keyword arguments passed to the Scene constructor.

        """
        super(Basic, self).__init__(*sceneArgs, **sceneKwargs)


    # Scene extension methods
    # --------------------------------------------------------------------------

    def _loadScene(self):
        return {}


    def _randomise(self):
        pass
