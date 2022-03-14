"""Parameters based on available information from
https://www.mynteye.com/pages/mynt-eye-d.
"""
from behaviour_gym.camera import monocular


class D100050(monocular.Monocular):

    def __init__(self, physicsClient, restPos, restPan, restTilt,
                     **rendererKwargs):
        """Initialises Mynt D1000-50 camera as a Monocular Camera."""

        super(D100050, self).__init__(
            physicsClient, restPos, restPan, restTilt, 50, 1280, 720, 0.02,
            50.0, 0.5, 15, renderer=physicsClient.ER_TINY_RENDERER,
            **rendererKwargs
        )


class D1000120(monocular.Monocular):

    def __init__(self, physicsClient, restPos, restPan, restTilt,
                     **rendererKwargs):
        """Initialises Mynt D1000-120 camera as a Monocular Camera."""

        super(D1000120, self).__init__(
            physicsClient, restPos, restPan, restTilt, 120, 1280, 720, 0.02,
            50.0, 0.3, 10, renderer=physicsClient.ER_TINY_RENDERER,
            **rendererKwargs
        )
