"""Module containing helper functions for working with matrices."""
import numpy as np
from pyquaternion import Quaternion

from behaviour_gym.utils import quaternion as q


def getRotationMatrix(quaternion):
    """Returns the quaternion as a 3x3 rotation matrix.

    Args:
        quaternion ([float]): quaternion.

    Returns:
        3x3 rotation matrix.

    """
    # Wrap quaternion
    quaternion = q.wrap(quaternion)

    # Return rotation matrix
    return quaternion.rotation_matrix


def getTransformationMatrix(pos, orn):
    """Returns the pose as a 4x4 homogeneous transformation matrix.

    Args:
        pos ([float]): [X,Y,Z] translation.
        orn ([float]): quaternion.

    Returns:
        4x4 homogeneous transformation matrix.

    """
    # Wrap quaternion
    quaternion = q.wrap(orn)

    # Get transformation matrix
    matrix = quaternion.transformation_matrix

    # Add position translation
    matrix[0:3,3] = pos

    # Return transformation matrix
    return matrix


def getPose(transMatrix, prevQuaternion=None):
    """Returns the 4x4 homogeneous transformation matrix as a pose.

    Note - a single rotation matrix can have multiple equivalent quaternions,
           i.e., q == -q. Converting quaternions to matrix form and back may
           result in sign swapping. This may be problematic for differentation
           where small changes in rotation should not result in large changes
           in the 4D quaternion space. See prevQuaternion arg for how to
           combat this.

    Args:
        transMatrix ([float]): 4x4 matrix to extract pose from.
        prevQuaternion ([float]): optional argument to prevent quaternion sign
                                  flipping. Ensure this quaternion is in the
                                  same coordinate frame as the transMatrix.

    Returns:
        4x4 homogeneous transformation matrix.

    """
    # Get position from matrix
    pos = transMatrix[0:3,3]

    # Get quaternion from matrix
    orn = Quaternion(matrix=transMatrix)

    # If given previous quaternion do sign flipping check
    if prevQuaternion is not None:
        # Wrap previous quaternion
        orn2 = q.wrap(prevQuaternion)

        # Measure distance from prev orn to orn and -orn
        dist = Quaternion.sym_distance(orn, orn2)
        distFlipped = Quaternion.sym_distance(-orn, orn2)

        # Flip orientation if -orn closer to prev orn
        if distFlipped < dist:
            orn = -orn

    return pos, q.unwrap(orn)


def invertMatrix(matrix):
    """Returns the inverse of the input matrix.

    Args:
        matrix ([float]): matrix to find inverse of.

    Returns:
        inverse of matrix.

    """
    return np.linalg.inv(matrix)


def interpolate(startPos, startOrn, endPos, endOrn, n):
    q1 = q.wrap(startOrn)
    q2 = q.wrap(endOrn)

    amounts = np.linspace(0, 1, n+1)

    poses = []
    for amount in amounts[1:]:
        pos = [(1 - amount)*startPos[0] + amount*endPos[0],
               (1 - amount)*startPos[1] + amount*endPos[1],
               (1 - amount)*startPos[2] + amount*endPos[2]]
        # pos = (1 - amount) * startPos + endPos
        orn = Quaternion.slerp(q1, q2, amount)
        poses.append([pos, q.unwrap(orn)])

    return poses
