"""Module containing helper functions for working with quaternion.

A quaternion is defined as the array [X,Y,Z,W] following PyBullet's convention.

Converts between PyBullet and PyQuaternion implementations.
"""
import math

import numpy as np
from pyquaternion import Quaternion


def wrap(pybulletQuaternion):
    """Wraps the PyBullet quaternion into a unit PyQuaternion.

    Args:
        pybulletQuaternion ([float]): Quaternion from PyBullet [X,Y,Z,W].

    Returns:
        same quaterntion wrapped in a PyQuaternion.

    """
    # If already a PyQuaternion return unit quaternion
    if isinstance(pybulletQuaternion, Quaternion):
        return pybulletQuaternion.normalised

    # Else convert to PyQuaternion and return unit quaternion
    q = Quaternion(pybulletQuaternion[3], *pybulletQuaternion[:3])
    return q.normalised


def unwrap(pyQuaternion):
    """Unwraps the PyQuaternion into a PyBullet unit quaternion.

    Args:
        pybulletQuaternion ([float]): Quaternion from PyBullet [X,Y,Z,W].

    Returns:
        same quaterntion wrapped in a PyQuaternion.

    """
    # If not a PyQuaternion then return
    if not isinstance(pyQuaternion, Quaternion):
        return pyQuaternion

    # Else normalise and return in pybullet form
    q = pyQuaternion.normalised
    return  [q[1], q[2], q[3], q[0]]


def rotateLocal(quaternion, xAngle, yAngle, zAngle):
    """Rotates a quaternion along the x, y and z axis by the given angles.

    Applies the rotation as a local rotation, i.e, Quaternion*Rotation.

    Args:
        quaternion ([float]): the quaternion to be rotated.
        xAngle (float): radians to rotate around the x axis.
        yAngle (float): radians to rotate around the y axis.
        zAngle (float): radians to rotate around the z axis.

    Returns:
        [float]: the rotated quaternion.

    """
    # Wrap Quaternion
    q = wrap(quaternion)

    # Construct rotation quaternions from axis angles
    xRot = Quaternion(axis=(1.0, 0.0, 0.0), radians=xAngle)
    xRot = xRot.normalised
    yRot = Quaternion(axis=(0.0, 1.0, 0.0), radians=yAngle)
    yRot = yRot.normalised
    zRot = Quaternion(axis=(0.0, 0.0, 1.0), radians=zAngle)
    zRot = zRot.normalised

    # Calculate total rotation
    rot = xRot * yRot * zRot

    # Apply rotation
    q = q * rot

    return unwrap(q)


def rotateGlobal(quaternion, xAngle, yAngle, zAngle):
    """Rotates a quaternion along the x, y and z axis by the given angles.

    Applies the rotation as a global rotation, i.e, Rotation*Quaternion.

    Args:
        quaternion ([float]): the quaternion to be rotated.
        xAngle (float): radians to rotate around the x axis.
        yAngle (float): radians to rotate around the y axis.
        zAngle (float): radians to rotate around the z axis.

    Returns:
        [float]: the rotated quaternion.

    """
    # Wrap Quaternion
    q = wrap(quaternion)

    # Construct rotation quaternions from axis angles
    xRot = Quaternion(axis=(1.0, 0.0, 0.0), radians=xAngle)
    xRot = xRot.normalised
    yRot = Quaternion(axis=(0.0, 1.0, 0.0), radians=yAngle)
    yRot = yRot.normalised
    zRot = Quaternion(axis=(0.0, 0.0, 1.0), radians=zAngle)
    zRot = zRot.normalised

    # Calculate total rotation
    rot = zRot * yRot * xRot

    # Apply rotation
    q = rot * q

    return unwrap(q)


def distance(quaternion0, quaternion1):
    """Computes the distance between two quaternions.

    Args:
        quaternion0 ([float]): first quaternion.
        quaternion1 ([float]): second quaternion.

    Returns:
        positive scalar corresponding to the chord of the shortest path/arc that
        connects q1 to q2.

    """
    # Wrap Quaternions
    q0 = wrap(quaternion0)
    q1 = wrap(quaternion1)

    # Return distance
    return Quaternion.absolute_distance(q0, q1)


def random():
    """Returns a random quaternion sampled from a uniform distribution."""
    return unwrap(Quaternion.random())
