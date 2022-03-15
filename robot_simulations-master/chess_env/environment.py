import gym
import math
import cv2
import numpy as np
import random
from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t
from stockfish import Stockfish
import chess
from Chessnut import Game
import os


class PickAndPlace(gym.Env):
    """
    Constructs a gym environment using robot_simulations adapted for BBRL.
    """

    def __init__(
        self,
        scene,
        robot,
        step_fn,
        max_arm_move=0.05,
        max_arm_rot=math.pi / 10,
        camera=None,
        max_steps=240,
        timestep=None,
        reactive_net=None,
        device=None,
    ):
        """Initialises a PickAndPlace environment"""

        self.p = scene.getPhysicsClient()
        self.curr_fen = scene.current_fen_string
        self.prev_fen = scene.prev_fen_string
        self.posDict = scene.posDict
        self.scene = scene
        self.robot = robot
        self.camera = camera
        self.step_fn = step_fn
        self._goal_object = "a1"
        self._dest_object = "a1"
        self._current_behaviour = "approach"
        self.max_steps = max_steps
        self._max_move = max_arm_move
        self._max_rot = max_arm_rot
        self._step = 0
        self._is_sparse = True
        self._target_dist_threshold = 0.03
        if reactive_net is not None:
            self._target_dist_threshold = 0.03
        self._approach_dist_threshold = 0.03
        self._place_dist_threshold = 0.03
        self.reactive_net = reactive_net
        self.device = device
        self.has_approached = False
        self.has_grasped = False
        self.has_retracted = False
        self.has_placed = False
        self.robot.resetBasePose(pos=[0, 0, 0.6])
        self._setup_observation_space()
        self.timestep = timestep
        if os.name == 'nt':
            self.stockfish = Stockfish(
                r'C:\Users\fream\Downloads\robot_simulations-master\robot_simulations-master\chess_env\stockfish2.exe')
        else:
            self.stockfish = Stockfish(os.path.abspath(
                "/home/pitsill0s/Desktop/ChessBBRL/robot_simulations-master/chess_env/stockfish_14.1_linux_x64"))
        self.action_space = self._setup_action_space(self._get_num_actions())

        self.metadata = {"render.modes": ["human", "rgbd_array"]}

    def reset(self):
        """Resets the evnironment"""
        self._step = 0
        self._current_behaviour = "approach"
        self.has_approached = False
        self.has_grasped = False
        self.has_retracted = False
        self.has_placed = False
        self._reset_scene()

        self.obs = self._get_obs()
        return self.obs

    def step(self, action):
        """
        Executes the action in the environment
        This method uses a step function that is defined in the script
        that calls the environment.

        This enables switching between executing low level actions from
        the reactive network, expert behaviours or high level behaviours
        that make use of the reactive network as part of the environment.
        """
        self._step += 1
        obs, reward, done, info = self.step_fn(self, action)
        return obs, reward, done, info

    def render(self, mode="rgbd_array"):
        """Renders the environment.

        Requires that a camera is loaded into the environment.

        Args:
            mode (string): either rgbd_array or human.
                           human  - displays rgb and depth images.
                           rgbd_array - returns an rgbd array.

        Returns:
            rgbd array if in rgbd_array mode else empty array

        """
        if self.camera is not None:
            # Get camera image
            rgbdImg = self.camera.getRgbd()

            if mode.lower() == "human":
                # Display rgb and depth images
                rgbImg = rgbdImg[:, :, :3]
                bgrImg = cv2.cvtColor(rgbImg.astype(
                    "float32"), cv2.COLOR_RGB2BGR)
                depthImg = rgbdImg[:, :, 3]
                cv2.imshow("RGB Image", bgrImg)
                cv2.imshow("Depth Image", depthImg)
                # Wait for keypress then return empty array
                cv2.waitKey(10)
                return np.array([])

            elif mode.lower() == "rgbd_array":
                # Return the rgbd array
                return rgbdImg

        # If no camera or unrecognised mode then return empty array
        return np.array([])

    def close(self):
        pass

    def approached(self):
        """
        Checks whether approach is successful and set corresponding variable
        """
        if not self.has_approached:
            robot_pos, _ = self.robot.getPose()
            block_pos, _ = self.scene.getPose(self._goal_object)
            # Target for approach is 5cm above object
            block_pos[2] += 0.05
            self.has_approached = (
                self._get_dist(
                    robot_pos, block_pos) <= self._approach_dist_threshold
            )

    def retracted(self):
        """
        Checks whether retract is successful and set corresponding variable
        """
        if not self.has_retracted:
            block_pos = self.robot.getGripperObs().copy()[:3]
            target_pos = self.scene.getDestTarget().copy()
            target_pos[2] = 0.7
            self.has_retracted = (
                np.linalg.norm(
                    target_pos - block_pos) <= self._approach_dist_threshold
            )

    def grasped(self):
        """
        Checks whether grasp is successful and set corresponding variable
        """
        if not self.has_grasped:
            self.has_grasped = (
                self.scene.getNumFingerTipContacts(0, self._goal_object) == 3
            )

    def placed(self):
        """
        Checks whether place is successful and set corresponding variable
        """
        if not self.has_placed:
            block_pos, _ = self.scene.getPose(self._goal_object)
            final_target = self.scene.getDestTarget()
            block_is_in_place = (
                self._get_dist(
                    block_pos, final_target) <= self._place_dist_threshold
            )
            self.has_placed = block_is_in_place

    def _get_obs(self):
        """Returns the observation from the environment"""

        if self.camera is None:
            # Vanilla BBRL that uses object and end effector information
            robot_obs = self.robot.getGripperObs()
            # print("Getting position for ",self._goal_object)

            block_pos, block_orn = self.scene.getPose(self._goal_object)
            block_lin_vel, block_ang_vel = self.scene.getVelocity(
                self._goal_object)
            relative_pos = block_pos - robot_obs[:3]
            gripper_state = 0
            if self.has_grasped:
                gripper_state = 1

            obs = np.concatenate(
                (
                    robot_obs,
                    block_pos,
                    block_orn,
                    block_lin_vel,
                    block_ang_vel,
                    relative_pos,
                    self.scene.getTarget(),
                    self.scene.getDestTarget(),
                    [gripper_state],
                )
            )
            # print(obs)
            # print(robot_obs)
            # print(block_pos)
            # print(block_orn)
            # print(block_lin_vel)
            # print(block_ang_vel)
            # print(relative_pos)
            # print(self.scene.getTarget())
            #print("OBS DEST TARGET",self.scene.getDestTarget())

        else:
            # Return image from camera and the joint states of the robot
            robot_obs = self.robot.getJointStates()
            image = self.camera.getRgbd()

            obs = {"robot": robot_obs, "image": image}
        return obs

    def _get_reward(self):
        if self._is_success():
            return 1
        return 0

    def _is_done(self):
        if self._is_success():
            return True
        elif self._step == self.max_steps:
            return True
        return False

    def _is_success(self):
        if self.has_approached and self.has_grasped and self.has_retracted and self.has_placed:
            return True
        return False

    def _get_info(self):
        info = {"target": self.scene.getTarget(
        ), "is_success": self._is_success()}

        return info

    def _setup_action_space(self, n_actions):
        """
        Sets up the actions space
        """
        if self.reactive_net is not None:
            return gym.spaces.Discrete(n_actions)
        return gym.spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")

    def _setup_observation_space(self):
        """
        Get observation space dimensions
        """
        obs_shape = self._get_obs_shape()
        if self.camera is None:
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(obs_shape,), dtype="float32"
            )
        else:
            robot, image = obs_shape
            robot_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(robot,), dtype="float32"
            )
            image_space = gym.spaces.Box(
                low=0, high=255, shape=image, dtype=np.uint8)
            self.observation_space = gym.spaces.Tuple(
                (robot_space, image_space))

    def _get_num_actions(self):
        """
        Get num actions
        """
        if self.reactive_net is not None:
            return 3
        return 5

    def _get_obs_shape(self):
        """Returns the environment's observation shape"""

        obs = self._get_obs()
        if self.camera is None:
            return obs.shape[0]

        return obs["robot"].shape[0], obs["image"]

    def _reset_scene(self):
        """Resets the scene by placing the robot, block and target at random positions"""

        self.scene.reset(random=True)

        self.stockfish.set_fen_position(self.scene.current_fen_string)
        fen = self.stockfish.get_fen_position()
        print("fen", fen)
        board = chess.Board(fen)
        print(self.stockfish.get_board_visual())

        move = self.stockfish.get_best_move()
        chessMove = chess.Move.from_uci(move)
        if board.is_castling(chessMove):
            self.move_piece(move, castling=True,
                            kingSideCastle=board.is_kingside_castling(chessMove))
        else:
            self.move_piece(move, castling=False,
                            kingSideCastle=board.is_kingside_castling(chessMove))

        print("Chess Move", chessMove)
        print("Origin", self._goal_object)
        print("Destination", self._dest_object)
        self.scene.setInit()
        self.scene.start()
        # After simulation starts do a dummy action to better initiliase
        print(self.scene.objects[self._goal_object].model)
        robPos, robOrn = self.robot.getPose()
        self.robot.applyPose(robPos, robOrn, relative=True)
        self.scene.step(30)

    def _get_dist(self, robot_pos, goal_pos):
        """Return distance from robot_pos to goal_pos."""
        robot_pos = np.array(robot_pos)
        goal_pos = np.array(goal_pos)
        squared_dist = np.sum((goal_pos - robot_pos) ** 2, axis=0)

        return np.sqrt(squared_dist)

    def move_piece(self, pieceString, castling, kingSideCastle):
        origin, destination = pieceString[:2], pieceString[2:][:2]
        globalOrn = [0, 0, 0]
        globalOrn = self.p.getQuaternionFromEuler(globalOrn)
        self._goal_object = origin
        self._dest_object = destination
        self.scene.target = self.posDict[origin].pos
        self.scene.destTarget = self.posDict[destination].pos
        print("Setting destination position ", self.scene.destTarget)

        if castling == True:
            dest_1, dest_2 = destination[:1], destination[1:]
            if kingSideCastle == True:
                rook_from = 'h' + dest_2
                rook_to = 'f' + dest_2
                self.posDict[rook_to].model = self.posDict[rook_from].model
                self.posDict[rook_from].model = None
                self.p.resetBasePositionAndOrientation(
                    self.posDict[rook_to].model[0], self.posDict[rook_to].pos, globalOrn
                )
            else:
                rook_from = 'a' + dest_2
                rook_to = 'd' + dest_2
                self.posDict[rook_to].model = self.posDict[rook_from].model
                self.posDict[rook_from].model = None
                self.p.resetBasePositionAndOrientation(
                    self.posDict[rook_to].model[0], self.posDict[rook_to].pos, globalOrn
                )

        if self.posDict[destination].model != None:
            print("Theres a piece at destination we need to remove it ")
            self.p.resetBasePositionAndOrientation(
                self.posDict[destination].model[0], [0, 0, 0], globalOrn
            )

        # self.p.resetBasePositionAndOrientation(
        #                 self.posDict[origin].model[0], self.posDict[destination].pos, globalOrn
        #                  )

        # self.posDict[destination].model = self.posDict[origin].model
        # self.posDict[origin].model = None

    def update_board_configuration(self, pieceString):
        print("Updating board", pieceString)
        split = 2
        origin, destination = pieceString[:split], pieceString[split:]
        print("THIS IS THE HYPOTHETHICAL POSTION",
              self.posDict[origin].model[0])
        print("THIS IS THE ACTUAL POSTION", self.posDict[destination].pos)

        self.posDict[destination].model = self.posDict[origin].model
        self.posDict[origin].model = None
