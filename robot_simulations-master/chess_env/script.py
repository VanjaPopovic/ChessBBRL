import math
import time
import random
import numpy as np
import os
import cv2
from stockfish import Stockfish
import chess
from Chessnut import Game
class ChessSquare:
    def __init__(self, pos, model, name):
        self.name = name 
        self.pos = pos
        self.model = model

if __name__ == "__main__":

    from environment import PickAndPlace
    from pick_place import PickPlaceScene
    from utils import *
    if os.name == 'nt':
        stockfish = Stockfish(
            r'C:\Users\fream\Downloads\robot_simulations-master\robot_simulations-master\chess_env\stockfish2.exe')
    else:
        stockfish = Stockfish(os.path.abspath(
            "/home/pitsill0s/Desktop/ChessBBRL/robot_simulations-master/chess_env/stockfish_14.1_linux_x64"))
    scene = PickPlaceScene(False, guiMode=True)
    robot = scene.loadRobot()
    env = PickAndPlace(scene, robot, step_fn_expert_behaviours)

    view_matrix = env.p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=1.58,
        yaw=91.20,
        pitch=-50.20,
        roll=0,
        upAxisIndex=2)

    projectionMatrix = env.p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=100.0)
    for i in range(100):
        level = random.uniform(1, 15)
        stockfish.set_skill_level(level)
        scene.stockfish.set_skill_level(level)
        stockfish.set_fen_position(scene.start_fen)
        board = chess.Board(scene.start_fen)

        move = stockfish.get_best_move()
        chessMove = chess.Move.from_uci(move)
        env.reset()
        # im_rgb = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
        # img = cv2.imshow("Image",im_rgb)
        # cv2.imwrite("file.jpg",im_rgb)
        # cv2.waitKey()
        done = False
        while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()):
            if env._current_behaviour == "approach":
                action = 0
            elif env._current_behaviour == "grasp":
                action = 1
            elif env._current_behaviour == "retract":
                action = 2
            else:
                action = 3
            env.step(action)
            if env.has_placed:
                env.reset()
    # width, height, rgbImg, depthImg, segImg = env.p.getCameraImage(
    #     width=1280,
    #     height=1024,
    #     viewMatrix=view_matrix,
    #     projectionMatrix=projectionMatrix)

