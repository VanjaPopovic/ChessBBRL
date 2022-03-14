import os
import pathlib
import time
import string
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

from behaviour_gym.utils import factory


class Scene:
    """Base class for all pybullet scenes."""

    def __init__(self, startSteps=240, gravity=[0, 0, -10], guiMode=False):
        """setups the base scene.

        Args:
            startSteps (int): number of steps to take through the physics
                              engine when initially starting the simulation.
                              Typically used to let models come to a rest.
            gravity ([float]): gravity along the X, Y and Z world axes.
            guiMode (bool): whether to connect in GUI or DIRECT mode.

        """
        # Scene Parameters
        self.startSteps = startSteps
        self.gravity = gravity
        self.guiMode = guiMode
        self.robots = []
        self.cameras = []

        # Scene internal parameters
        self._initState = None
        self._isRunning = False

        # Set up physics client
        self.guiMode = guiMode
        if self.guiMode:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        # self.p.setPhysicsEngineParameter(numSolverIterations=1000)
        # self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        # self.p.setPhysicsEngineParameter(contactERP=0.3)
        self.p.setGravity(*gravity)

        # Set up factories
        self.robotFactory = factory.RobotFactory2()
        self.cameraFactory = factory.CameraFactory()

        # Set up model folders for future loading
        self.pybulletModels = pybullet_data.getDataPath()
        behaviourGymRoot = pathlib.Path(
            __file__).resolve().parent.parent.parent
        self.objectModels = str(behaviourGymRoot) + "/models/objects/"

        # Load the world objects
        self.objects = self._loadScene()
        self.debugObjects = []

    # Methods
    # --------------------------------------------------------------------------

    def setInit(self):
        """Sets the scene's initial state for future reloading.

        This should be called only once under the following conditions:
            After loading in all models. Do not load or delete models after
            calling this method.

            Before stepping through the physics engine with p.stepSimulation().
            Randomisation may make use of functions that override the physics
            simulation which may cause simulation artifacts if the simulation
            is in progress.

        """
        if not self.isInitiliased():
            self._initState = self.p.saveState()

    def reset(self, random=True):
        """Resets the scene.

        Cannot reset the scene until the initial state has been set.

        Args:
            random (bool): If true then randomises the scene before starting.
                           Otherwise it starts from its initial state.

        """
        # If scene not initialised then don't start simulation
        if self.isInitiliased():
            # Set simulation in running state
            self._isRunning = False

            for object in self.debugObjects:
                self.p.removeBody(object)

            # Restore the initial state
            # self.p.restoreState(self._initState)

            # Reset all robots to their rest configuration
            for robot in self.robots:
                robot.reset()

            # Reset all cameras to their rest configuration
            for camera in self.cameras:
                camera.reset()

            # Randomise the scene
            if random:
                self.objects = self._randomise()

    def start(self):
        """Starts the simulation.

        Cannot be called until the initial state has been set.

        """
        if self._isRunning:
            print("Error - Simulation has already started.")
        elif not self.isInitiliased():
            print("Error - Simulation has not been initialised.")
        else:
            self._isRunning = True
            self.step(self.startSteps)

    def step(self, steps, timestep=None):
        """Steps through the physics simulation.

        Args:
            steps (int): number of steps to take through the environment.
                         240=1 second of simulation time.
            timestep (float): if not none waits this many seconds between
                              physics steps.

        """
        if self._isRunning:
            for i in range(steps):
                self.p.stepSimulation()
                if timestep is not None:
                    time.sleep(timestep)
        else:
            print("Error - Simulation has not been started yet.")

    def close(self):
        """Closes the physics simulation."""
        self.p.disconnect()

    def loadRobot(self, robotName, **robotKwargs):
        """Loads a robot into the scene.

        Cannot load a robot after calling setInit.

        Args:
            robotName (string): name of the robot to load into the scene.
            robotKwargs (dict): key word arguments for the robot constructor.

        Returns:
            Robot object.

        """
        if not self.isInitiliased():
            # Create the desired robot
            robot = self.robotFactory.createRobot(robotName,
                                                  physicsClient=self.p,
                                                  **robotKwargs)

            # Reset to rest config
            robot.reset(*robot.getRest())

            # Add to list of robots
            self.robots.append(robot)

            # Return robot
            return robot
        else:
            print("Error - Do not load additional models after calling setInit().")

    def loadCamera(self, cameraName, **cameraKwargs):
        """Loads a camera into the scene.

        Args:
            cameraName (string): name of the camera to load from the
                                 CameraFactory.
            cameraKwargs: key word arguments passed to the Camera's constructor.

        Returns:
            Camera object.

        """
        # Create desired camera
        camera = self.cameraFactory.createCamera(cameraName,
                                                 physicsClient=self.p,
                                                 **cameraKwargs)

        # Add to list of cameras
        self.cameras.append(camera)

        # Return camera
        return camera

    def visualisePose(self, pos, orn):
        # Load axes
        path = os.path.join(self.objectModels, "axes/axes.urdf")
        axisId = self.p.loadURDF(path, pos, orn)
        self.debugObjects.append(axisId)
        return axisId

    def getPhysicsClient(self):
        """Returns the scene's physics client."""
        return self.p

    def getObjects(self):
        """Returns a dictionary of objects that can be interacted with.

        Returns:
            {<string>: <[int]>} : Dictionary of objects in the scene that can be
                                  interacted with. Each object has a unique name
                                  which indexes to an array containing
                                  [modelId, linkId]. Model's with no links
                                  should use -1 as their link index.

        """
        return self.objects

    def getObject(self, name):
        """Returns the objects ID (and link ID if applicable).

        Returns:
            object ID (int)
            link ID (int), -1 if using base

        """
        return self.objects[name].model

    def getPose(self, name):
        """Returns the position and orientation of an object in the scene.

        Args:
            name (string): name of the object as it appears in the objects
                           dictionary.

        Returns:
            world position [x,y,z],
            world orientation as quaternion [x,y,z,w].

        """
        print("NAME IN HEREEEEEEEEEEEEEEEEEEE", name)
        modelId = self.objects[name].model[0]
        linkId = -1

        if linkId == -1:
            # If model has only a single link use base position and orientation
            objectPos, objectOrn = self.p.getBasePositionAndOrientation(
                modelId)
        else:
            # Otherwise use the designated link
            state = self.p.getLinkState(modelId, linkId)
            objectPos = state[0]
            objectOrn = state[1]

        return np.array(objectPos), np.array(objectOrn)

    def getVelocity(self, name):
        """Returns the linear and angular velocity of an object in the scene.

        Args:
            name (string): name of the object as it appears in the objects
                           dictionary.

        Returns:
            world linear velocity [x,y,z],
            world angular velocity [x,y,z].

        """
        # Get the model and link indices
        modelId = self.objects[name].model[0]
        linkId = -1

        if linkId == -1:
            # If model has only a single link use base velocity
            objectLinVel, objectAngVel = self.p.getBaseVelocity(modelId)
        else:
            # Otherwise use the designated link
            state = self.p.getLinkState(
                modelId, linkId, computeLinkVelocity=True)
            objectLinVel = state[6]
            objectAngVel = state[7]

        return np.array(objectLinVel), np.array(objectAngVel)

    def isInitiliased(self):
        """Returns true if the scene has been initialised."""
        if self._initState is not None:
            return True
        return False

    # Extension methods
    # --------------------------------------------------------------------------

    def _loadScene(self):
        """Loads the scenes objects.

        Returns:
            {<string>: <[int]>} : Dictionary of objects in the scene that can be
                                  interacted with. Each object has a unique name
                                  which indexes to an array containing
                                  [modelId, linkId]. Model's with no joints
                                  should use -1 as their link index.

        """
        raise NotImplementedError()

    def _randomise(self):
        """Randomises the scene."""
        raise NotImplementedError()
