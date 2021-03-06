import math
import time
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
            "/home/pitsill0s/Downloads/robot_simulations-master/chess_env/stockfish_14.1_linux_x64"))
    scene = PickPlaceScene(True, guiMode=True)
    robot = scene.loadRobot()
    env = PickAndPlace(scene, robot, step_fn_expert_behaviours, False)

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
    stockfish.set_fen_position(scene.current_fen_string)
    board = chess.Board(scene.current_fen_string)
    print(stockfish.get_board_visual())

    move = stockfish.get_best_move()
    chessMove = chess.Move.from_uci(move)
    while True:
        env.reset()
        # im_rgb = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
        # img = cv2.imshow("Image",im_rgb)
        # cv2.imwrite("file.jpg",im_rgb)
        # cv2.waitKey()
        if env._is_done():
            print("YAAAAAAAAAAAAAAAAAAAAAAAAAAAAAY")
            env.update_board_configuration(move)

        move = stockfish.get_best_move()

        print("PLAYING MOVE", move)
        chessMove = chess.Move.from_uci(move)

        # width, height, rgbImg, depthImg, segImg = env.p.getCameraImage(
        #     width=1280,
        #     height=1024,
        #     viewMatrix=view_matrix,
        #     projectionMatrix=projectionMatrix)

        while not env.has_approached:
            env.step(0)
            if env.has_approached:
                continue
        # while not env.has_grasped:
        #     env.step(1)
        # while not env.has_retracted:

        #     # print("HAS RETRACTED, ", env.has_retracted)
        #     env.step(2)

        # while not env.has_placed:

        #     print(env.has_placed)
        #     env.step(3)

        # board.push(chessMove)
        # print(board)
        # stockfish.set_fen_position(board.fen())
        # print(board.fen())