import pathlib

import numpy as np


class Robot:
    """Base class for all robots."""

    def __init__(self, physicsClient, urdfFile, nObs, nGripActions,
                 startPos=[0,0,0], startOrn=[0,0,0]):
        """Initialises base robot.

        Args:
            physicsClient (obj): physics client for loading and controlling
                                 the robot.
            urdfFile (string): filepath of the urdf file to load from the
                               models/robots/ directory.
            nGripActions (int): the number of actions for controlling the
                                gripper
            nObs (int): number of observations in array returned by getObs().
            nGripActions (int): number of actions for manipulating the gripper
                                using applyGripActions().
            startPos ([float]): position [X,Y,Z] in world space of the base
                                link (NOTE: not the centre of mass).
            startOrn ([float]): orientation in world space [X,Y,Z] or [X,Y,Z,W]
                                of the base link (NOTE: not centre of mass)

        """
        # Client and base link pose info
        self._p = physicsClient
        self.startPos = startPos
        if len(startOrn) == 3:
            self.startOrn = self._p.getQuaternionFromEuler(startOrn)
        else:
            self.startOrn = startOrn

        # Load Robot
        root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        urdfPath = root + "/models/robots/" + urdfFile
        self.robotId = self._p.loadURDF(urdfPath, self.startPos, self.startOrn)

        # Number of observations and gripper actions
        self.nObs = nObs
        self.nGripActions = nGripActions


    # Extension methods
    # --------------------------------------------------------------------------

    def applyArmPose(self, goalPos, goalOrn):
        """Sets arm joint positions to move the end-effector to the goal pose.

        Args:
            goalPos ([float]): position [X,Y,Z] in world space.
            goalOrn ([float]): quaternion [X,Y,Z,W] relative to end-effector
                               axes.

        """
        raise NotImplementedError()


    def applyArmPoseRelative(self, x, y, z, roll, pitch, yaw):
        """Sets joint positions to move the end-effector relative to start pose.

        Args:
            x (float): metres to move in the X axis.
            y (float): metres to move in the Y axis.
            z (float): metres to move in the Z axis.
            roll (float): radians to rotate around the end-effector's X axis.
            pitch (float): radians to rotate around the end-effector's Y axis.
            yaw (float): radians to rotate around the end-effector's Z axis.

        """
        raise NotImplementedError()


    def applyArmRestPose(self):
        """Sets arm joints to move the end-effector to its rest pose."""
        raise NotImplementedError()


    def applyGripAction(self, action):
        """Sets gripper joints to execute the action.

        Args:
            action ([float]): action to manipulate the gripper, specifics depend
                              on the choice of gripper and implementation.

        """
        raise NotImplementedError()


    def applyGripRestPos(self):
        """Sets gripper joints to their rest positions."""
        raise NotImplementedError()


    def resetToRest(self):
        """Resets the joints positions to rest instantly and updates motors.

        Overrides the physics engine so only to be used at the start of a
        simulation, i.e., before stepSimulation has been called.

        """
        raise NotImplementedError()


    def rest(self, goalPos, goalOrn, gripAction):
        """Resets the joint positions adnd motors instantly.

        Overrides the physics engine so only to be used at the start of a
        simulation, i.e., before stepSimulation has been called.

        Args:
            goalPos ([float]): position [X,Y,Z] in world space.
            goalOrn ([float]): quaternion [X,Y,Z,W] relative to end-effector
                               axes.
            gripAction ([float]): action to manipulate the gripper, specifics
                                  depend on the choice of gripper and
                                  implementation.

        """
        raise NotImplementedError()


    def getObs(self):
        """Returns observation of the robot as a numpy array."""
        raise NotImplementedError()


    def getArmPose(self):
        """Returns end-effector's position [X,Y,Z] and orientation [X,Y,Z,W]."""
        raise NotImplementedError()


    def getFingerTipPos(self):
        """Returns position [X,Y,Z] of the gripper's finger tips."""
        raise NotImplementedError()


    def getFingerTipLinks(self):
        """Returns the list of ids for the gripper's finger tip links."""
        raise NotImplementedError()


    # Helper methods
    # --------------------------------------------------------------------------

    def printInfo(self):
        """Prints info about the robot's links and joints"""
        print("################Link Info################")
        print(f'|{"Index":<5} | {"Name":<30}|')
        linkNameIndex = {self._p.getBodyInfo(self.robotId)[0].decode('UTF-8'):-1,}
        for i in range(self._p.getNumJoints(self.robotId)):
        	linkName = self._p.getJointInfo(self.robotId, i)[12].decode('UTF-8')
        	linkNameIndex[linkName] = i
        for link in linkNameIndex.items():
            print(f'|{link[1]:<5} | {link[0]:<30}|')

        print("")
        print("################Link State################")
        print(f'|{"index":<5} | {"World Position":<17} | {"World Orientation":<23} | {"Linear Velocity":<17} | {"Angular Velocity":<17}|')
        for i in range(0, len(linkNameIndex)-1):
            state = self._p.getLinkState(self.robotId, i, computeLinkVelocity=True)
            pos = ' '.join(map('{:5.2f}'.format, state[0]))
            orn = ' '.join(map('{:5.2f}'.format, state[1]))
            posVel = ' '.join(map('{:5.2f}'.format, state[6]))
            ornVel = ' '.join(map('{:5.2f}'.format, state[7]))
            print(f'|{i:<5} | {pos:<17} | {orn:<23} | {posVel:<17} | {ornVel:<17}|')

        print("")
        print("################Joint Info################")
        print(f'|{"Index":<5} | {"Name":<30} | {"Type":<4} | {"Damping":7} | {"Friction":<8} | {"Lower Limit":<11} | {"Upper Limit":<11} | {"Max Force":<9} | {"Max Velocity":<12} | {"Axis":<11} | {"Parent Index":<12}|')
        for i in range(self._p.getNumJoints(self.robotId)):
            info = self._p.getJointInfo(self.robotId, i)
            pos = ' '.join(map('{:1}'.format, info[13]))
            print(f'|{info[0]:<5} | {str(info[1]):<30} | {info[2]:<4} | {info[6]:<7} | {info[7]:<8} | {info[8]:<11.4} | {info[9]:<11.4} | {info[10]:<9} | {info[11]:<12} | {pos:<11} | {info[16]:<12}|')
        print("")
        print("################Joint State################")
        print(f'|{"index":<5} | {"Position":<8} | {"Velocity":<8} | {"Reaction Forces":<35} | {"Motor Torque":<12}|')
        for i in range(self._p.getNumJoints(self.robotId)):
            state = self._p.getJointState(self.robotId, i)
            reactionForces = ' '.join(map('{:5.2f}'.format, state[2]))
            print(f'|{i:<5} | {state[0]:<8.2f} | {state[1]:<8.2f} | {reactionForces:<35} | {state[3]:<12.2f}|')
