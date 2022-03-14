from behaviour_gym.robot import ur10_3f, ur10_hand_lite, bbrl_robot


class RobotFactory:
    """Factory class for loading in and returning Robot implementations."""

    def createRobot(self, robotName, **robotKwargs):
        """
        Loads and returns reference to the desired robot.
        Args:
            robotName (string): name of the robot to load
            physicsClient (obs): the physics client to use
            robotKwargs (dict): keyword arguments for the robot constructor
        """
        if robotName.upper() == "UR103F":
            return ur10_3f.Ur103f(**robotKwargs)
        elif robotName.upper() == "UR10HANDLITE":
            return ur10_hand_lite.Ur10HandLite(**robotKwargs)
        elif robotName.upper() == "BBRL":
            return bbrl_robot.BBRLRobot(**robotKwargs)
        else:
            raise ValueError(robot)
