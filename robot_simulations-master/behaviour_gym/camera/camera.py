import cv2
import numpy as np


class Camera:
    """Base class for all Cameras."""

    def __init__(self, physicsClient, restPos, restPan, restTilt, fov,
                 width, height, nearVal, farVal, minDepth, maxDepth):
        """Initialises base camera.

        Args:
            physicsClient (obj): physics client for loading and controlling
                                 the camera.
            restPos ([float]): position [X,Y,Z] metres in world space of
                               the camera.
            restPan (float): initial pan in radians.
            restTilt (float): initial tilt in radians.
            fov (float): the cameras field of view.
            width (int): the width of camera images.
            height (int): the height of camera images.
            nearVal (int): minimum range (metres) for rendering.
            farVal (int): maximum range (metres) for rendering.
            minDepth (float): minimum depth range (metres).
            maxDepth (float): maximum depth range (metres).

        """
        # Physics Client
        self._p = physicsClient

        # Rest Configuration
        self.restPos = restPos
        self.restPan = restPan
        self.restTilt = restTilt

        # Current configuration
        self.pos = restPos
        self.pan = restPan
        self.tilt = restTilt

        # Camera parameters
        self.fov = fov
        self.width = width
        self.height = height
        self.nearVal = nearVal
        self.farVal = farVal
        self.minDepth = minDepth
        self.maxDepth = maxDepth

        # Compute projection matrix once then reuse
        self.projMatrix = self._p.computeProjectionMatrixFOV(fov=self.fov,
                                                             aspect=self.width / self.height,
                                                             nearVal=self.nearVal,
                                                             farVal=self.farVal)

        # Finish setting up camera
        self.reset()


    # Methods
    # --------------------------------------------------------------------------

    def reset(self):
        """Resets camera to its start pose and parameters."""
        self.pos = self.restPos
        self.pan = self.restPan
        self.tilt = self.restTilt
        self._updateViewMatrix()


    def move(self, x, y, z, pan, tilt):
        """Moves camera to designated world pose.

        Args:
            x (float): metres from origin along the X axis.
            y (float): metres from origin along the Y axis.
            z (float): metres from origin along the Z axis.
            pan (float): camera pan in radians.
            tilt (float): camera tilt in radians.

        """
        self.pos = [x, y, z]
        self.pan = pan
        self.tilt = tilt
        self._updateViewMatrix()


    def moveRelative(self, x, y, z, pan, tilt):
        """Moves camera pose relative to its current pose.

        Args:
            x (float): metres to move in the X axis.
            y (float): metres to move in the Y axis.
            z (float): metres to move in the Z axis.
            pan (float): radians to pan the camera.
            tilt (float): radians to tilt the camera.

        """
        self.pos = np.add(self.pos, [x, y, z])
        self.pan = self.pan + pan
        self.tilt = self.tilt + tilt
        self._updateViewMatrix()


    def getPose(self):
        """Returns the camera's pose as a numpy array.

        Returns:
            float array with values [x, y, z, pan, tilt]

        """
        pose = [self.pos[0],
                self.pos[1],
                self.pos[2],
                self.pan,
                self.tilt]
        return np.array(pose)


    def getRest(self):
        """Returns the camera's rest pose.

        Returns:
            float array with values [X,Y,Z,pan,tilt].

        """
        pose = [self.restPos[0],
                self.restPos[1],
                self.restPos[2],
                self.restPan,
                self.restTilt]
        return np.array(pose)


    def setRest(self, pos, pan, tilt):
        """Sets the camera's rest configuration.

        Args:
            pos ([float]): [X,Y,Z] position in world frame.
            pan (float): pan in radians.
            tilt (float): tilt in radians.

        """
        self.restPos = pos
        self.restPan = pan
        self.restTilt = tilt


    # Extension Methods
    # --------------------------------------------------------------------------

    def getRgbd(self):
        """Returns image data in RGBD numpy array.

        Number of returned channels dependent on camera implementation.
        Monocular return 4 channels (RGB and Depth).
        Stereo returns 7 (RGB*2 and single Depth from the left or right camera).

        Returns:
            float array of shape [height, width, channels]

        """
        raise NotImplementedError()


    def _updateViewMatrix(self):
        """Updates the view matrix to using the camera's current parameters."""
        raise NotImplementedError()


    # Helper Methods
    # --------------------------------------------------------------------------

    def getRgbImg(self, rgba):
        """Converts an RGBA image to a RGB.

        Args:
            rgba ([float]): the rgba image returned by p.getCameraImage().

        Returns:
            numpy array [height, width, channels(3)] in range the range [0,1].

        """
        rgbaImg = np.array(rgba, dtype=np.uint8)
        rgbaImg = np.reshape(rgbaImg, (self.height, self.width, 4))
        rgbImg = cv2.cvtColor(rgbaImg, cv2.COLOR_RGBA2RGB)
        return rgbImg/255


    def getDepthImg(self, depthBuffer):
        """Clamps true depth values between camera's range.

        Args:
            depthBuffer ([float]): the depth buffer returned by
                                   p.getCameraImage().

        Returns:
            numpy array [height, width] in the range [0,1].

        """
        trueDepth = self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depthBuffer)
        clampedDepth = np.clip(trueDepth, self.minDepth, self.maxDepth)
        depthImg = (clampedDepth - self.minDepth) / (self.maxDepth - self.minDepth)
        return depthImg
