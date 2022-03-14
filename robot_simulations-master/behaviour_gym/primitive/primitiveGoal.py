import math

import cv2
import gym
from gym.utils import seeding
import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



class PrimitiveGoal(gym.GoalEnv):

    def __init__(self, scene, robot, camera=None, maxSteps=50,
                 timestep=None):
        """Initiliases the base primitive.z

        Args:
            scene (Scene): the scene object to act within.
            robot (Robot): the robot object to interact with.
            camera (Camera): the camera object to interact with.
            maxSteps (int): the maximum number of steps to take in a single
                            episode.
            timestep (float): how long to sleep between each physics step. If
                              None then does not sleep. Only useful if the Scene
                              was created with guiMode=True.

        """
        # Physics Client
        self.p = scene.getPhysicsClient()

        # Scene setup
        self.scene = scene
        self.robot = robot
        self.camera = camera

        # Maximum steps through environment
        self.maxSteps = maxSteps
        self._step = 0

        # Time to sleep between each p.stepSimulation
        self.timestep = timestep

        # Gym metadata
        self.metadata = {'render.modes': ['human', 'rgbd_array']}

        # Environment spaces
        nObs = self._getNumObs()
        self.observation_space = gym.spaces.Dict({
            'observation' : gym.spaces.Box(-np.inf, np.inf, shape=(nObs,), dtype='float32'),
            'achieved_goal' : gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype='float32'),
            'desired_goal' : gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype='float32'),
        })

        # self.observation_space = gym.spaces.Box(-np.inf, np.inf,
        #                                         shape=(nObs,),
        #                                         dtype='float32')
        nActions = self._getNumActions()
        self.action_space = gym.spaces.Box(-1., 1., shape=(nActions,),
                                           dtype='float32')

        # Track robot's relative orientation to base to prevent sign flipping
        self._prevRobotOrnBase = None


    # Methods
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
        """Resets the environment ro a random starting state."""
        # Reset and randomise the scene
        self.scene.reset(random=True)

        # Reset step counter
        self._step = 0

        # Reset robot's relative orientation tracker
        self._prevRobotOrnBase = None

        # Randomise the environment/robot
        randomiseSuccess = False
        while not randomiseSuccess:
            randomiseSuccess = self._randomise()

        # Start the simulation
        self.scene.start()

        # After simulation starts do a dummy action to better initiliase
        robPos, robOrn = self.robot.getPose()
        self.robot.applyPose(robPos, robOrn, relative=True)
        self.scene.step(30)

        # Optional post initialisation callback
        self._resetCallback()

        return self.getObs()


    def step(self, action):
        """Executes the action and steps through the environment."""
        self._step = self._step + 1

        # Get action for manipulating robot
        action = np.clip(action, self.action_space.low, self.action_space.high)
        goalPos, goalOrn, gripAction = self._getAction(action)

        # If manipulating gripper then update gripper motors
        if gripAction is not None:
            self.robot.applyGripAction(gripAction)

        # Interpolate poses
        pos, orn = self.robot.getPose()
        poses = t.interpolate(pos, orn, goalPos, goalOrn, 2)

        # Step through poses
        for i in range(2):
            # Calculate IK solution
            self.robot.applyPose(poses[i][0], poses[i][1], relative=True)

            # Step through scene 1/8 of a second
            self.scene.step(30, self.timestep)

        # Step through scene for 1/12 of a second to reach final pose
        self.scene.step(20, self.timestep)

        # Get step data and training info
        obs = self.getObs()
        reward = self.compute_reward(self._getAchievedGoal(), self._getDesiredGoal(), {})
        done = self._isDone()
        info = {
            'is_success': self._isSuccess()
        }
        info.update(self._getInfo())

        # If max number of actions executed then episode is done
        if self._step >= self.maxSteps:
            done = True

        # Optional post step callback
        self._stepCallback()

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

        Use scene.close() to close the simulation. A single scene could be
        used in multiple environments so left to user to decide when to
        exit the the simulation.

        """
        pass


    def getObs(self):
        """Returns an observation of the environment."""
        # Get robot observation
        robotObs = self.getRobotObs()

        # Concatenate with the rest of the observations
        obs = np.concatenate((robotObs, self._getObs())).flatten()
        achievedGoal = self._getAchievedGoal()
        actualGoal = self._getDesiredGoal()

        # Return flattened vector
        return {
            'observation': obs.copy(),
            'achieved_goal': achievedGoal.copy(),
            'desired_goal': actualGoal.copy()
        }


    def getRobotObs(self):
        """Returns an vector observation of the robot."""
        # # Get robot pose in base frame
        # posWorld, ornWorld = self.robot.getPose()
        # posBase, ornBase = self.robot.toBasePose(posWorld, ornWorld,
        #                                          self._prevRobotOrnBase)
        #
        # # Update previous orientation
        # self._prevRobotOrnBase = ornBase
        #
        # # Get robot velocities in base frame
        # linVelWorld, angVelWorld = self.robot.getVelocity()
        # linVelBase, angVelBase = self.robot.toBaseVelocity(linVelWorld,
        #                                                    angVelWorld)
        #
        # # Get gripper observation
        # gripObs = self.robot.getGripObs()
        #
        # return np.concatenate((posBase, ornBase, linVelBase, angVelBase,
        #                        gripObs))
        # Get robot pose in base frame
        posWorld, ornWorld = self.robot.getPose()
        posBase, ornBase = self.robot.toBasePose(posWorld, ornWorld,
                                                 self._prevRobotOrnBase)

        # Update previous orientation
        self._prevRobotOrnBase = ornBase

        # Get robot velocities in base frame
        linVelWorld, angVelWorld = self.robot.getVelocity()
        linVelBase, angVelBase = self.robot.toBaseVelocity(linVelWorld,
                                                           angVelWorld)

        # Get gripper observation
        gripObs = self.robot.getGripObs()

        return np.concatenate((posWorld, ornWorld, linVelWorld, angVelWorld,
                               gripObs))


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


    def compute_reward(self, achieved_goal, desired_goal, info):
        """Return the reward."""
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.sparse:
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


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
        """A custom callback that is called after stepping the simulation.

        E.g., can be used to check for collisions after step through the
        environment and tracking other variables.

        """
        pass


    def _resetCallback(self):
        """A custom callback that is called after resetting the simulation.

        E.g., can be used to get the initial positions of objects after the
        simulation has started.

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
