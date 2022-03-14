import math

import numpy as np
import cv2

from behaviour_gym.camera import camera


class Monocular(camera.Camera):
    """Base class for all Monocular Cameras."""

    def __init__(self, *cameraArgs, **rendererKwargs):
        """Initialises base Monocular Camera.

        Args:
            *cameraArgs: arguments passed to Camera constructor
            **renderKwargs: key word arguments passed to p.getCameraImage().
                            Note that the majority only work with tiny renderer.

        """
        super(Monocular, self).__init__(*cameraArgs)
        self.rendererKwargs = rendererKwargs


    # Camera Extension Methods
    # --------------------------------------------------------------------------

    def getRgbd(self):
        width, height, rgba, depth, seg = self._p.getCameraImage(width=self.width,
                                                                 height=self.height,
                                                                 viewMatrix=self.viewMatrix,
                                                                 projectionMatrix=self.projMatrix,
                                                                 **self.rendererKwargs)

        rgbImg = self.getRgbImg(rgba)
        depthImg = self.getDepthImg(depth)
        rgbdImg = np.dstack((rgbImg, depthImg))
        return rgbdImg


    def _updateViewMatrix(self):
        self.viewMatrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.pos,
                                                                     distance=1e-4,
                                                                     yaw=self.pan * 180/math.pi,
                                                                     pitch=self.tilt * 180/math.pi,
                                                                     roll=0.0,
                                                                     upAxisIndex=2)
