import os
import math
import random
import pybullet_data

from behaviour_gym.scene import scene
from robot import UR103F


class ChessScene(scene.Scene):
    """Base class for all gym environments."""

    def __init__(self, blockPosRange=[[0.4, 0.95], [-0.4, 0.4]], *sceneArgs,
                 **sceneKwargs):
        """Initialises the table environment.

        Args:
            blockPosRange [float]: range for randomising the block's position.
                                   [[xMin,xMax], [yMin,yMax]]
            sceneArgs: arguments passed to the Scene constructor.
            sceneKwargs: keyword arguments passed to the Scene constructor.

        """
        self.blockPosMinX = blockPosRange[0][0]
        self.blockPosMaxX = blockPosRange[0][1]
        self.blockPosMinY = blockPosRange[1][0]
        self.blockPosMaxY = blockPosRange[1][1]
        super(PickPlaceScene, self).__init__(*sceneArgs, **sceneKwargs)

    # Scene extension methods
    # --------------------------------------------------------------------------

    def loadRobot(self, **robotKwargs):
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
            robot = UR103F(physicsClient=self.p, **robotKwargs)

            # Reset to rest config
            robot.reset(*robot.getRest())

            # Add to list of robots
            self.robots.append(robot)

            # Return robot
            return robot
        else:
            print("Error - Do not load additional models after calling setInit().")

    def _loadScene(self):
        # Load ground plane
        path = os.path.join(self.pybulletModels, "plane.urdf")
        self.p.loadURDF(path)

        # Load table
        path = os.path.join(self.objectModels, "table/table.urdf")
        self.tableId = self.p.loadURDF(
            path, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        # Load block
        path = os.path.join(self.objectModels, "block/block.urdf")
        # blockOrn = [0, 0, random.uniform(0, math.pi / 2)]
        blockOrn = [0, 0, math.pi/2]
        blockOrn = self.p.getQuaternionFromEuler(blockOrn)
        self.blockId = self.p.loadURDF(path, [0.6, 0.0, 0.64],
                                       blockOrn)

        # Use same constraints for goal
        self.target = [random.uniform(self.blockPosMinX, self.blockPosMaxX), random.uniform(
            self.blockPosMinY, self.blockPosMaxY), 0.7]
        file_path = os.path.join(
            pybullet_data.getDataPath(), "sphere_smooth.obj")
        meshScale = [0.02, 0.02, 0.02]
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        self.visualShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_MESH, fileName=file_path, rgbaColor=[
                                                      1, 1, 1, 1], specularColor=[0.4, .4, 0], meshScale=meshScale)
        self.sphereId = self.p.createMultiBody(
            baseMass=0.0, baseVisualShapeIndex=self.visualShapeId, basePosition=self.target)

        # Return dictionary of object names to model and link indices
        objects = {}
        objects["block"] = [self.blockId, -1]

        return objects

    def _randomise(self):
        # Get a random starting pose for the block
        blockPos = [random.uniform(self.blockPosMinX, self.blockPosMaxX),
                    random.uniform(self.blockPosMinY, self.blockPosMaxY),
                    0.64]
        # blockOrn = [0, 0, random.uniform(0, math.pi / 2)]
        blockOrn = [0, 0, math.pi/2]
        blockOrn = self.p.getQuaternionFromEuler(blockOrn)

        # Reset block to the random pose
        self.p.resetBasePositionAndOrientation(
            self.blockId, blockPos, blockOrn)

        # Reset goal to random position
        self.target = [random.uniform(self.blockPosMinX, self.blockPosMaxX), random.uniform(
            self.blockPosMinY, self.blockPosMaxY), 0.7]
        self.p.resetBasePositionAndOrientation(
            self.sphereId, self.target, [0, 0, 0, 1])

        armStartPos = [random.uniform(self.blockPosMinX,
                                      self.blockPosMaxX),
                       random.uniform(self.blockPosMinY,
                                      self.blockPosMaxY),
                       random.uniform(0.9, 1.3)]

        self.robots[0].reset(armStartPos)

    # Helper functions

    def getNumFingerTipContacts(self, robot_index, block_name):
        """Get the number of finger tips in contact with the block."""
        contactPointsBlock = self.p.getContactPoints(self.robots[robot_index].id,
                                                     self.objects[block_name][0])
        fingerTips = self.robots[robot_index].getFingerTipLinks()
        contacts = []
        for contactPoint in contactPointsBlock:
            if contactPoint[3] in fingerTips:
                contacts.append(contactPoint[3])

        numUniqueContacts = len(set(contacts))
        return numUniqueContacts

    def getTarget(self):
        return self.target
