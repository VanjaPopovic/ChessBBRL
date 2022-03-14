import os
import time
import math
import pathlib

import cv2
import pkgutil
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import gym
from gym.utils import seeding

from behaviour_gym.robot import utils


class Environment(gym.Env):
    """Base class for all gym environments."""

    def __init__(self, robotName, physicsSteps=30, renders=False,
                 timestep=1./240., **robotKwargs):
        """
        Initialises base environment.

        Args:
            robotName (string): name of the robot to load in
            physicsSteps (int): number of steps through physics engine per
                                environment step
            renders (boolean): whether or not to render environment
            timestep (int): if renders, time to sleep after each step through
                            the physics
            robotKwargs (dict): key word arguments for the robot constructor
        """
        # Env Variables
        self._physicsSteps = physicsSteps

        # Rendering Variables
        self.renders = renders
        self.timestep = timestep
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.timestep))
        }

        # Physics client
        if self.renders:
            self._p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        # Spawn robot
        robotKwargs['physicsClient'] = self._p
        self.robot = utils.RobotFactory().createRobot(robotName, **robotKwargs)

        # Move to rest pose
        self.robot.applyArmRestPose()
        self.robot.applyGripRestPos()
        for i in range(300):
            self._p.stepSimulation()

        # Spawn rest of scene
        self._urdfRoot = pybullet_data.getDataPath()
        self._p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"))
        root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        urdfPath = root + "/models/objects/table/table.urdf"
        self._tableId = self._p.loadURDF(urdfPath, [0.5, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 1.0])
        self._blockId = None
        self._sphereId = None
        # Save starting state
        self.seed()
        self.startStateId = self._p.saveState()

        # Environment spaces
        nObs = self._getNumObs()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(nObs,),
                                                dtype='float32')
        nActions = self._getNumActions()
        self.action_space = gym.spaces.Box(-1., 1., shape=(nActions,),
                                           dtype='float32')


    # Methods
    # --------------------------------------------------------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        # Restore starting state
        if self._blockId is not None:
            self._p.removeBody(self._blockId)
        if self._sphereId is not None:
            self._p.removeBody(self._sphereId)
        self._p.restoreState(self.startStateId)

        # Reset environment
        reset_success = False
        while not reset_success:
            reset_success = self._reset()

        return np.array(self._getObs())


    def step(self, action):
        # Set action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step through physics environment
        self._setAction(action)
        for i in range(self._physicsSteps):
            self._p.stepSimulation()
            if self.renders:
                time.sleep(self.timestep)
        self._stepCallback()

        # Get step data and training info
        obs = self._getObs()
        reward = self._getReward()
        done = self._isDone()
        info = {
            'is_success': self._isSuccess()
        }
        info.update(self._getInfo())

        return np.array(obs), reward, done, info


    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        view_matrix = self._p.computeViewMatrix(cameraEyePosition=[0.5, 0.8, 1.6],
                                                cameraTargetPosition=[0.5, 0, 0.4],
                                                cameraUpVector=[0, 0, 1.0])
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=50,
                                                         aspect=1280.0 / 720.0,
                                                         nearVal=0.02,
                                                         farVal=50.0)

        width, height, rgb, depth, seg = self._p.getCameraImage(width=1280,
                                                                height=720,
                                                                viewMatrix=view_matrix,
                                                                projectionMatrix=proj_matrix,
                                                                lightDirection=[-1.5, -1.5, 2.5],
                                                                lightDistance=4.0,
                                                                shadow=True,
                                                                renderer=self._p.ER_TINY_RENDERER,
                                                                **args)

        rgba_array = np.array(rgb, dtype=np.uint8)
        rgba_array = np.reshape(rgb_array, (720, 1280, 4))
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2RGB)
        trueDepth = 50.0 * 0.02 / (50.0 - (50.0 - 0.02) * depth)
        depthMinMax = (depth - depth.min()) / (depth.max() - depth.min())
        return rgb_array

    def close(self):
        self._p.disconnect()


    # Extension methods
    # --------------------------------------------------------------------------

    def _reset(self):
        """
        Reset environment after reloading initial state.
        Return True if successful.
        """
        raise NotImplementedError()


    def _setAction(self, action):
        """Set action before stepping through physics."""
        raise NotImplementedError()


    def _getObs(self):
        """Return the observation."""
        raise NotImplementedError()


    def _getReward(self):
        """Return the reward."""
        raise NotImplementedError()


    def _isDone(self):
        """Return if done."""
        raise NotImplementedError()


    def _isSuccess(self):
        """Return if episode successful."""
        raise NotImplementedError()


    def _getNumObs(self):
        """Return the number of observations."""
        raise NotImplementedError()


    def _getNumActions(self):
        """Return the number of actions."""
        raise NotImplementedError()


    def _getInfo(self):
        """Return dictionary of additional info."""
        return {}


    def _stepCallback(self):
        """A custom callback that is called after stepping the simulation. Can
        be used to enforce additional constraints on the simulation state.
        """
        pass


    # Helper methods
    # --------------------------------------------------------------------------


    def _getDist(self, startPos, goalPos):
        """Return distance from startPos to goalPos."""
        startPos = np.array(startPos)
        goalPos = np.array(goalPos)
        squared_dist = np.sum((goalPos-startPos)**2, axis=0)
        return np.sqrt(squared_dist)


    def _getQuaternionDist(self, q1, q2):
        """Return angle of rotation to get from q1 to q2."""
        q1 = np.array(q1)
        q2 = np.array(q2)
        # Small bug here, I (Nikos) think, where sometimes inner > 1
        # This causes the arc cosine function to be undefined since the domain
        # is -1 <= x <= 1
        inner = np.clip(np.inner(q1, q2), -1, 1)
        dist = math.acos(2 * inner**2 - 1)
        return dist


    def _isContact(self):
        """Check if contact made with table or block."""
        contactPointsTable = self._p.getContactPoints(self.robot.robotId,
                                                      self._tableId)
        contactPointsBlock = 0
        if self._blockId is not None:
            contactPointsBlock = self._p.getContactPoints(self.robot.robotId,
                                                          self._blockId)
        if len(contactPointsTable) > 1:
            return True
        if len(contactPointsBlock) > 0:
            return True
        return False
