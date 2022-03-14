import math

import cv2
import gym
from gym.utils import seeding
import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t

import time

class Primitive(gym.Env):

    def __init__(self, scene, robot, camera=None, timestep=None,
                 physicsSteps=40, interpolation=1):
        """Initiliases the base primitive.

        Args:
            scene (Scene): the scene object to act within.
            robot (Robot): the robot object to interact with.
            camera (Camera): the camera object to interact with.
            timestep (float): how long to sleep between each physics step. If
                              None then does not sleep. Only useful if the Scene
                              was created with guiMode=True.
            physicsSteps (int): number of physics steps per step of the
                                environment.
            interpolation (int): number of poses to interpolate between the
                                 current and goal end-effector pose. If 1 then
                                 no interpolation performed. If interpolating,
                                 tweak physicsSteps to get desired behaviour.

        """
        # Physics Client
        self.p = scene.getPhysicsClient()
        self.physicsSteps = physicsSteps
        self.interpolation = interpolation

        # Scene setup
        self.scene = scene
        self.robot = robot
        self.camera = camera
        self.timestep = timestep

        # Track step data
        self.lastPrevObs = None
        self.lastObs = None
        self.lastAction = None
        self.lastReward = None
        self.lastInfo = None

        # Gym metadata and spaces
        self.metadata = {'render.modes': ['human', 'rgbd_array']}
        nObs = self._getNumObs()
        self.observation_space = gym.spaces.Box(np.float32(-np.inf),
                                                np.float32(np.inf),
                                                shape=(nObs,),
                                                dtype=np.float32)
        nActions = self._getNumActions()
        self.action_space = gym.spaces.Box(np.float32(-1), np.float32(1),
                                           shape=(nActions,),
                                           dtype=np.float32)


    # Gym Methods
    # --------------------------------------------------------------------------

    def seed(self, seed=None):
        """Seeds the random number generator with a strong seed.

        Note that if seed=None then an OS specific source of randomness is used.

        Args:
            seed (int): a positive int value or None.

        Returns:
            the seed in an array

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        """Resets the environment to a random starting state."""
        # Randomise the environment
        randomiseSuccess = False
        while not randomiseSuccess:
            self.scene.reset(random=True)
            randomiseSuccess = self._randomise()

        # Start the simulation with a dummy action
        self.scene.start()
        robPos, robOrn = self.robot.getPose()
        self.robot.applyPose(robPos, robOrn, relative=True)
        self.scene.step(30)

        # Optional post initialisation callback
        self._resetCallback()

        # Get initial observation
        obs = self.getObs()

        # Track step data
        self.lastPrevObs = None
        self.lastObs = obs
        self.lastAction = None
        self.lastReward = None
        self.lastInfo = None
        self._prevRobotOrnBase = None

        return obs


    def step(self, action):
        """Executes the action and steps through the environment."""
        # Get robot actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        goalPos, goalOrn, gripAction = self._getAction(action)

        # If manipulating gripper then update gripper motors
        if gripAction is not None:
            self.robot.applyGripAction(gripAction)

        if self.interpolation > 1:
            # Interpolate poses
            pos, orn = self.robot.getPose()
            poses = t.interpolate(pos, orn, goalPos, goalOrn, self.interpolation)

            # Calculate physics steps per pose
            interpSteps = self.physicsSteps - 20
            interpMod = interpSteps % self.interpolation
            stepsPerPose = int((interpSteps - interpMod) / self.interpolation)

            # Step through poses
            for i in range(self.interpolation):
                self.robot.applyPose(poses[i][0], poses[i][1], relative=True)
                self.scene.step(stepsPerPose, self.timestep)

            # Extra steps to fully reach goal pose
            self.scene.step(20 + interpMod, self.timestep)
        else:
            # If not interpolating then go straight to final pose
            self.robot.applyPose(goalPos, goalOrn, relative=True)
            self.scene.step(self.physicsSteps, self.timestep)

        # Optional post step callback
        self._stepCallback()

        # Calc step accuracy
        pos, orn = self.robot.getPose()
        posError = self._getDist(pos, goalPos)
        ornError = q.distance(orn, goalOrn)
        errorTotal = posError + ornError
        #if errorTotal > 0.001:
            # time.sleep(1)
            #print("Step error")
            #print("Pos: ", posError)
            #print("Orn: ", ornError)
            #print("Total: ", errorTotal)

        # Get step data
        obs = self.getObs()
        reward = self._getReward(action)
        done = self._isDone()
        info = {
            'is_success': self._isSuccess()
        }
        info.update(self._getInfo())

        # Track step data
        self.lastPrevObs = self.lastObs
        self.lastObs = obs
        self.lastAction = action
        self.lastReward = reward
        self.lastInfo = info
        return obs, reward, done, info


    def render(self, mode="rgbd_array"):
        """Renders the environment.

        Requires that a camera is loaded into the environment.

        Args:
            mode (string): either rgbd_array or human.
                           human  - displays rgb and depth images.
                           rgbd_array - returns an rgbd array.

        Returns:
            rgbd array if in rgbd_array mode else empty array

        """
        if self.camera is not None:
            # Get camera image
            rgbdImg = self.camera.getRgbd()

            if mode.lower() == "human":
                # Display rgb and depth images
                rgbImg = rgbdImg[:,:,:3]
                bgrImg = cv2.cvtColor(rgbImg.astype('float32'), cv2.COLOR_RGB2BGR)
                depthImg = rgbdImg[:,:,3]
                cv2.imshow("RGB Image", bgrImg)
                cv2.imshow("Depth Image", depthImg)
                # Wait for keypress then return empty array
                cv2.waitKey(10)
                return np.array([])

            elif mode.lower() == "rgbd_array":
                # Return the rgbd array
                return rgbdImg

        # If no camera or unrecognised mode then return empty array
        return np.array([])


    def close(self):
        """Does nothing but left in for compatibility.

        Use scene.close() to shutdown the simulation. A single simulation could
        be used in multiple environments so left to the user to decide when to
        exit the simulation.

        """
        pass


    # Methods
    # --------------------------------------------------------------------------

    def getObs(self):
        """Returns an observation of the environment."""
        # Get robot observation
        robotObs = self.robot.getObs()

        # Concatenate with the rest of the observations
        obs = np.concatenate((robotObs, self._getObs()))

        # Return flattened vector
        return obs.flatten()


    # Extension methods
    # --------------------------------------------------------------------------

    def _randomise(self):
        """Randomises the environment.

        Only call before starting the simulation.

        Returns:
            True if successful. Otherwise False.

        """
        raise NotImplementedError()


    def _getAction(self, action):
        """Returns actions for manipulating the robot.

        Returns:
            goalPos ([float]): goal position in world space [X,Y,Z] metres,
            goalOrn ([float]): goal orientation in world space [X,Y,Z,W],
            gripAction ([float]): vector feed into robot.applyGripAction. If
                                  None then gripper motors left unchanged.

        """
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


    def _stepCallback(self, action):
        """A custom callback that is called after stepping the simulation.

        Args:
            action [float]: action vector that has just been applied.

        User can access state transition and info of the previous step using
        self.lastObs, self.lastReward etc.

        """
        pass


    def _resetCallback(self):
        """A custom callback that is called after resetting the simulation."""
        pass


    # Helper methods
    # --------------------------------------------------------------------------

    def _getDist(self, startPos, goalPos):
        """Return distance from startPos to goalPos."""
        startPos = np.array(startPos)
        goalPos = np.array(goalPos)
        squared_dist = np.sum((goalPos-startPos)**2, axis=0)
        return np.sqrt(squared_dist)


    def getManhattanDist(self, startPos, goalPos):
        startPos = np.array(startPos)
        goalPos = np.array(goalPos)
        return np.sum(abs(goalPos-startPos))


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
