import pathlib

import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t


class Robot:
    """Base class for all robots."""

    def __init__(self, physicsClient, urdfFile, eeLink, restPos, restOrn,
                 restGripState, nGripActions, basePos=[0,0,0], baseOrn=[0,0,0,1]):
        """Initialises base robot.

        Args:
            physicsClient (obj): physics client for loading and controlling
                                 the robot.
            urdfFile (string): filepath of the urdf file to load from the
                               models/robots/ directory.
            eeLink (int): id of the end-effector link.
            restPos ([float]): position [X,Y,Z] (metres) in base coordinate
                               frame of the end-effector's rest position.
            restOrn ([float]): orientation [X,Y,W,Z] in base coordinate frame
                               of the end-effector's rest orientation.
            restGripState ([float]): vector describining the robot gripper's
                                     rest state. Specifics depend on choice of
                                     gripper and implementation.
            nGripActions (int): number of actions for manipulating the gripper.
            basePos ([float]): position [X,Y,Z] (metres) in world space of the
                               base link.
            baseOrn ([float]): orientation in world space [X,Y,Z,W] of the base
                               link.

        """
        # Physics Client
        self.p = physicsClient

        # End effector link
        self.eeLink = eeLink

        # Rest pose and gripper configuration
        self.restPos = restPos
        self.restOrn = restOrn
        self.restGripState = restGripState

        # Load Robot
        root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
        urdfPath = root + "/models/robots/" + urdfFile
        self.id = self.p.loadURDF(urdfPath)

        # Move to base position
        self.resetBasePose(basePos, baseOrn)

        # Reset robot to rest configuraration
        restPos, restOrn, restGripState = self.getRest()
        self.reset(restPos, restOrn, restGripState)

        # Number of observations and gripper actions
        self.nObs = len(self.getObs())
        self.nGripActions = nGripActions


    # Methods
    # --------------------------------------------------------------------------

    def applyPose(self, goalPos, goalOrn, relative=True):
        """Sets the arm joint motors to move the end-effector to the goal pose.

        Args:
            goalPos ([float]): position [X,Y,Z] in world space.
            goalOrn ([float]): quaternion [X,Y,Z,W] relative to end-effector
                               axes.
            relative (bool): if True then use current joint positions as the
                             rest positions for the IK solver.

        """
        # If relative get current joint positions as rest poses for IK
        rp = None
        if relative:
            rp = self.getJointsPos()

        # Calculate IK solution
        jointPoses = self.calcIK(goalPos, goalOrn, rp)

        # Update arm motors
        self.setArmMotorsPos(jointPoses)


    def reset(self, goalPos=None, goalOrn=None, gripState=None):
        """Resets joint positions instantly and updates motors.

        Do not call during a running simulation (i.e., p.stepSimulation() has
        been called at least once) as it overrides the physics.

        Args:
            goalPos ([float]): position [X,Y,Z] in world space.
            goalOrn ([float]): quaternion [X,Y,Z,W] relative to end-effector
                               axes.
            gripState ([float]): vector to manipulate the gripper. Each element
                                 should range between -1 to 1. Specifics on the
                                 effect of each element depends on the type of
                                 gripper and implementation.

        """
        # Use rest configuration for missing parameters
        restPos, restOrn, restGrip = self.getRest()
        if goalPos is None:
            goalPos = restPos
        if goalOrn is None:
            goalOrn = restOrn
        if gripState is None:
            gripState = restGrip

        # Calculate IK solution
        jointPoses = self.calcIK(goalPos, goalOrn)

        # Reset arm joint positions instantly
        self.resetArmJoints(jointPoses)

        # Update arm motors
        self.setArmMotorsPos(jointPoses)

        # Reset gripper state instantly
        self.resetGripState(gripState)

        # Update gripper motors
        self.applyGripState(gripState)


    def resetBasePose(self, pos=None, orn=None):
        """Resets the position and orientation of the base.

        Do not call during a running simulation (i.e., p.stepSimulation() has
        been called at least once) as it overrides the physics.

        Args:
            pos ([float]): world position [x,y,z],
            orn ([float]): world orientation as quaternion [x,y,z,w].

        """
        # Use current base pose for missing parameters
        basePos, baseOrn = self.getBasePose()
        if pos is None:
            pos = basePos
        if orn is None:
            orn = baseOrn

        # Reset base position
        self.p.resetBasePositionAndOrientation(self.id, pos, orn)

        # Reset robot
        self.reset()


    def getObs(self):
        """Returns a vector describing the robot.

        Index:
            [0:3] - position [x,y,z] in world frame.
            [3:7] - orientation [x,y,z,w] in world frame.
            [7:10] - linear velocity [x,y,z] in world frame.
            [10:13] - angular velocity [x,y,z] in world frame.
            [13:13+armJoints] - arm joint angles.
            [13+armJoints:] - gripper observation.

        Returns:
            vector of float values.

        """
        # Get pose in base frame
        pos, orn = self.getPose()

        # Get velocity in base frame
        linVel, angVel = self.getVelocity()

        # Get arm joint angles
        armJointAngles = self.getArmJointsPos()

        # Get vector observation of the gripper
        gripObs = self.getGripObs()

        return np.concatenate((pos, orn, linVel, angVel, armJointAngles,
                               gripObs))


    def getPose(self):
        """Returns the world position and orientation of the end-effector.

        Args:
            name (string): name of the object as it appears in the objects
                           dictionary.

        Returns:
            world position [x,y,z],
            world orientation as quaternion [x,y,z,w].

        """
        eeLinkState = self.p.getLinkState(self.id, self.eeLink)
        return eeLinkState[0], eeLinkState[1]



    def getVelocity(self):
        """Returns the world linear and angular velocity of the end-effector.

        Returns:
            world linear velocity [x,y,z],
            world angular velocity [x,y,z].

        """
        eeLinkState = self.p.getLinkState(self.id, self.eeLink,
                                           computeLinkVelocity=True)
        return eeLinkState[6], eeLinkState[7]


    def getBasePose(self):
        """Returns the position and orientation of the base.

        Returns:
            world position [x,y,z],
            world orientation as quaternion [x,y,z,w].

        """
        return self.p.getBasePositionAndOrientation(self.id)


    def getRest(self):
        """Returns the rest configuration in world space.

        Returns:
            world position [X,Y,W],
            world orientation [X,Y,W,Z],
            grip vector.

        """
        # Move rest pose from base to world frame
        restPosWorld, restOrnWorld = self.toWorldPose(self.restPos,
                                                      self.restOrn)
        return restPosWorld, restOrnWorld, self.restGripState


    def getJointsPos(self):
        """Returns list of joint positions.

        Ignores fixed joints.

        Returns:
            list of joint positions.

        """
        pos = []
        # Loop through every joint
        for i in range(self.p.getNumJoints(self.id)):
            info = self.p.getJointInfo(self.id, i)
            # If joint not fixed add position to list
            if info[2] != 4:
                state = self.p.getJointState(self.id, i)
                pos.append(state[0])

        return pos


    def getJointsVel(self):
        """Returns list of joint velocities.

        Ignores fixed joints.

        Returns:
            list of joint velocities.

        """
        vel = []
        # Loop through every joint
        for i in range(self.p.getNumJoints(self.id)):
            info = self.p.getJointInfo(self.id, i)
            # If joint not fixed add position to list
            if info[2] != 4:
                state = self.p.getJointState(self.id, i)
                vel.append(state[1])

        return vel


    def setRest(self, restPos, restOrn, restGripState):
        """Sets the rest configuration.

        Rest configuration automatically converted from world to base coordinate
        frame to make rest configuration agnostic to base pose.

        Args:
            restPos ([float]): world position [X,Y,Z] metres.
            restOrn ([float]): world orientation [X,Y,W,Z].
            restGripState ([float]): vector to control grip. Specifics depend on
                                     gripper choice and implementation.

        """
        # Move pose from world to base frame
        restPosBase, restOrnBase = self.toBasePose(restPos, restOrn)

        # Update rest configuration
        self.restPos = restPosBase
        self.restOrn = restOrnBase
        self.restGripState = restGripState


    # Extension methods
    # --------------------------------------------------------------------------

    def applyGripState(self, gripState):
        """Sets the gripper joint motors according to the gripper state vector.

        Args:
            gripState ([float]): vector describing the gripper state. Specifics
                                 depend on the type of gripper and
                                 implementation.

        """
        raise NotImplementedError()


    def applyGripAction(self, gripAction):
        """Sets the gripper joint motors to execute the gripper action.

        Args:
            gripState ([float]): vector to manipulate the gripper. Each element
                                 should range between -1 to 1. Specifics on the
                                 effect of each element depends on the type of
                                 gripper and implementation.

        """
        raise NotImplementedError()


    def resetGripState(self, gripState):
        """Resets the gripper joint positions instantly and update motors.

        Do not call during a running simulation (i.e., p.stepSimulation() has
        been called at least once) as it overrides the physics.

        Args:
            gripState ([float]): vector describing the gripper state. Specifics
                                 depend on the type of gripper and
                                 implementation.

        """
        raise NotImplementedError()


    def getGripError(self):
        """Returns the error between the grip state and the gripper joints."""
        raise NotImplementedError()


    def calcIK(self, goalPos, goalOrn, restPoses=None, useNull=True):
        """Returns an IK solution for the arm joints.

        Args:
            goalPos ([float]): world position [X,Y,Z].
            goalOrn ([float]): world orientation [X,Y,Z,W].
            restPoses ([float]): rest poses for IK null space. If none then
                                 should use a default and probably increase
                                 the IK iterations.
            useNull (bool): whether to use null space or not. If not using
                            null space then the IK iterations should be pretty
                            high.

        Returns:
            list of joint positions for each non-fixed arm joint.

        """
        raise NotImplementedError()


    def resetArmJoints(self, positions):
        """Resets arm joints instantly using list of joint poses.

        Do not call during a running simulation (i.e., p.stepSimulation() has
        been called at least once) as it overrides the physics.

        Args:
            positions ([float]): list of joint positions for each non-fixed arm
                                 joint, e.g. from calcIk.

        """
        raise NotImplementedError()


    def setArmMotorsPos(self, positions):
        """Sets arm motors using list of joint poses.

        Args:
            positions ([float]): list of joint positions for each non-fixed
                                 arm joint, e.g. from calcIk.

        """
        raise NotImplementedError()


    def setArmMotorsVel(self, velocities):
        """Sets arm motors using list of joint velocities.

        Args:
            velocities ([float]): list of joint velocities for each non-fixed
                                  arm joint.

        """
        raise NotImplementedError()


    def setGripMotorsVel(self, velocities):
        """Sets gripper motors using list of joint velocities.

        Args:
            velocities ([float]): list of joint velocities for each non-fixed
                                  gripper joint.

        """
        raise NotImplementedError()


    def getArmJointsPos(self):
        """Returns the arm joint angles

        Returns:
            vector of arm joint angles.

        """
        raise NotImplementedError()


    def getArmJointsVel(self):
        """Returns the arm joint velocities

        Returns:
            vector of arm joint velocities.

        """
        raise NotImplementedError()


    def getGripJointsVel(self):
        """Returns the gripper joint velocities

        Returns:
            vector of gripper joint velocities.

        """
        raise NotImplementedError()


    def getGripObs(self):
        """Returns a vector describing the gripper.

        Returns:
            vector observation of the gripper. Specifics depend on choice of
            gripper and implementation.

        """
        raise NotImplementedError()


    def getGripState(self):
        """Returns the gripper state.

        Returns:
            vector describing the gripper state.

        """
        raise NotImplementedError()


    def getFingerTipLinks(self):
        """Returns the list of ids for the gripper's finger tip links."""
        raise NotImplementedError()


    # Helper methods
    # --------------------------------------------------------------------------

    def toBasePose(self, posWorld, ornWorld, prevOrnBase=None):
        """Transforms the pose from the world to the base coordinate frame.

        Args:
            posWorld ([float]): position [X,Y,Z] metres in world frame.
            ornWorld ([float]): orientation [X,Y,W,Z] in world frame.
            prevOrnBase ([float]): optional argument specifying the previous
                                   orientation in the base frame. Used to
                                   prevent sign flipping.

        Returns:
            pos relative to base,
            orn relative to base.

        """
        # Get base pose in world frame as a matrix
        basePosWorld, baseOrnWorld = self.getBasePose()
        baseMatrixWorld = t.getTransformationMatrix(basePosWorld, baseOrnWorld)

        # Invert matrix
        worldMatrixBase = t.invertMatrix(baseMatrixWorld)

        # Get pose in world as matrix
        poseMatrixWorld = t.getTransformationMatrix(posWorld, ornWorld)

        # Transform pose matrix to the base frame
        poseMatrixBase = np.dot(worldMatrixBase, poseMatrixWorld)

        # Extract pose in base frame from matrix
        posBase, ornBase = t.getPose(poseMatrixBase, prevOrnBase)

        return posBase, ornBase


    def toBaseVelocity(self, linVelWorld, angVelWorld):
        """Transforms the pose from the world to the base coordinate frame.

        Args:
            posWorld ([float]): position [X,Y,Z] metres in world frame.
            ornWorld ([float]): orientation [X,Y,W,Z] in world frame.
            prevOrnBase ([float]): optional argument specifying the previous
                                   orientation in the base frame. Used to
                                   prevent sign flipping.

        Returns:
            pos relative to base,
            orn relative to base.

        """
        # Get rotation matrix in world frame
        basePosWorld, baseOrnWorld = self.getBasePose()
        baseRotWorld = t.getRotationMatrix(baseOrnWorld)

        # Invert rotation matrix
        worldRotBase = t.invertMatrix(baseRotWorld)

        # Construct matrix to transform the velocity vector
        matrix = np.zeros((6,6))
        matrix[0:3,0:3] = worldRotBase
        matrix[3:6,3:6] = worldRotBase

        # Construct velocity vector
        velWorld = [*linVelWorld, *angVelWorld]

        # Transform velocity vector
        velBase = np.dot(matrix, velWorld)

        # Return linear and angular velocity
        return velBase[:3], velBase[3:]


    def toWorldPose(self, posBase, ornBase, prevOrnWorld=None):
        """Transforms the pose from the base to the world coordinate frame.

        Args:
            posBase ([float]): position [X,Y,Z] metres in base frame.
            ornBase ([float]): orientation [X,Y,W,Z] in base frame.
            prevOrnWorld ([float]): optional argument specifying the previous
                                    orientation in the world frame. Used to
                                    prevent sign flipping.


        Returns:
            pos relative to world,
            orn relative to world.

        """
        # Get base pose in the world coordinate frame
        basePosWorld, baseOrnWorld = self.getBasePose()
        baseMatrixWorld = t.getTransformationMatrix(basePosWorld, baseOrnWorld)

        # Get pose in world as matrix
        poseMatrixBase = t.getTransformationMatrix(posBase, ornBase)

        # Transform pose in base frame to the world frame
        poseMatrixWorld = np.dot(baseMatrixWorld, poseMatrixBase)

        # Extract pose in world frame from the matrix
        posWorld, ornWorld = t.getPose(poseMatrixWorld, prevOrnWorld)

        return posWorld, ornWorld


    def toWorldVelocity(self, linVelBase, angVelBase):
        """Transforms the pose from the world to the base coordinate frame.

        Args:
            posWorld ([float]): position [X,Y,Z] metres in world frame.
            ornWorld ([float]): orientation [X,Y,W,Z] in world frame.
            prevOrnBase ([float]): optional argument specifying the previous
                                   orientation in the base frame. Used to
                                   prevent sign flipping.

        Returns:
            pos relative to base,
            orn relative to base.

        """
        # Get rotation matrix in world frame
        basePosWorld, baseOrnWorld = self.getBasePose()
        baseRotWorld = t.getRotationMatrix(baseOrnWorld)

        # Invert rotation matrix
        worldRotBase = t.invertMatrix(baseRotWorld)

        # Construct matrix to transform the velocity vector
        matrix = np.zeros((6,6))
        matrix[0:3,0:3] = worldRotBase
        matrix[3:6,3:6] = worldRotBase

        # Construct velocity vector
        velWorld = [*linVelWorld, *angVelWorld]

        # Transform velocity vector
        velWorld = np.dot(matrix, velWorld)

        # Return linear and angular velocity
        return velWorld[:3], velWorld[3:]


    def printInfo(self):
        """Prints info about the robot's links and joints"""
        print("################Link Info################")
        print(f'|{"Index":<5} | {"Name":<30}|')
        linkNameIndex = {self.p.getBodyInfo(self.id)[0].decode('UTF-8'):-1,}
        for i in range(self.p.getNumJoints(self.id)):
        	linkName = self.p.getJointInfo(self.id, i)[12].decode('UTF-8')
        	linkNameIndex[linkName] = i
        for link in linkNameIndex.items():
            print(f'|{link[1]:<5} | {link[0]:<30}|')

        print("")
        print("################Link State################")
        print(f'|{"index":<5} | {"World Position":<17} | {"World Orientation":<23} | {"Linear Velocity":<17} | {"Angular Velocity":<17}|')
        for i in range(0, len(linkNameIndex)-1):
            state = self.p.getLinkState(self.id, i, computeLinkVelocity=True)
            pos = ' '.join(map('{:5.2f}'.format, state[0]))
            orn = ' '.join(map('{:5.2f}'.format, state[1]))
            posVel = ' '.join(map('{:5.2f}'.format, state[6]))
            ornVel = ' '.join(map('{:5.2f}'.format, state[7]))
            print(f'|{i:<5} | {pos:<17} | {orn:<23} | {posVel:<17} | {ornVel:<17}|')

        print("")
        print("################Joint Info################")
        print(f'|{"Index":<5} | {"Name":<30} | {"Type":<4} | {"Damping":7} | {"Friction":<8} | {"Lower Limit":<11} | {"Upper Limit":<11} | {"Max Force":<9} | {"Max Velocity":<12} | {"Axis":<11} | {"Parent Index":<12}|')
        for i in range(self.p.getNumJoints(self.id)):
            info = self.p.getJointInfo(self.id, i)
            pos = ' '.join(map('{:1}'.format, info[13]))
            print(f'|{info[0]:<5} | {str(info[1]):<30} | {info[2]:<4} | {info[6]:<7} | {info[7]:<8} | {info[8]:<11.4} | {info[9]:<11.4} | {info[10]:<9} | {info[11]:<12} | {pos:<11} | {info[16]:<12}|')
        print("")
        print("################Joint State################")
        print(f'|{"index":<5} | {"Position":<8} | {"Velocity":<8} | {"Reaction Forces":<35} | {"Motor Torque":<12}|')
        for i in range(self.p.getNumJoints(self.id)):
            state = self.p.getJointState(self.id, i)
            reactionForces = ' '.join(map('{:5.2f}'.format, state[2]))
            print(f'|{i:<5} | {state[0]:<8.2f} | {state[1]:<8.2f} | {reactionForces:<35} | {state[3]:<12.2f}|')
