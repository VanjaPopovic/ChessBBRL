"""Module containing Factory classes."""
from behaviour_gym.robot import ur10_3f, ur10_hand_lite, bbrl_robot
from behaviour_gym.robot2 import ur10_3f as ur10_3f2
from behaviour_gym.camera import mynt


class RobotFactory:
    """Factory class for loading in Robot implementations."""

    def createRobot(self, robotName, **robotKwargs):
        """Loads and returns a reference to the desired robot.

        Args:
            robotName (string): name of the robot to load
            robotKwargs (dict): keyword arguments for the robot constructor

        Returns:
            the Robot object

        """
        if robotName.upper() == "UR103F":
            return ur10_3f.Ur103f(**robotKwargs)
        elif robotName.upper() == "UR10HANDLITE":
            return ur10_hand_lite.Ur10HandLite(**robotKwargs)
        elif robotName.upper() == "BBRL":
            return bbrl_robot.BBRLRobot(**robotKwargs)
        else:
            raise ValueError(robotName)


class CameraFactory:
    """Factory class for loading in Camera implementations."""

    def createCamera(self, cameraName, **cameraKwargs):
        """Loads and returns a reference to the desired camera.

        Args:
            cameraName (string): name of the camera to load
            cameraKwargs (dict): keyword arguments for the camera constructor

        Returns:
            the Camera object

        """
        if cameraName.upper() == "MYNTD100050":
            return mynt.D100050(**cameraKwargs)
        elif cameraName.upper() == "MYNTD1000120":
            return mynt.D1000120(**cameraKwargs)
        else:
            raise ValueError(cameraName)


class RobotFactory2:
    """Factory class for loading in Robot2 implementations."""

    def createRobot(self, robotName, **robotKwargs):
        """Loads and returns a reference to the desired robot.

        Args:
            robotName (string): name of the robot to load
            robotKwargs (dict): keyword arguments for the robot constructor

        Returns:
            the Robot object

        """
        if robotName.upper() == "UR103F":
            return ur10_3f2.Ur103f(**robotKwargs)
        else:
            raise ValueError(robotName)
