import os
import math
import random
import pybullet_data
import cv2

from behaviour_gym.scene import scene
from robot import UR103F
from stockfish import Stockfish
import chess
from Chessnut import Game
import os


class ChessSquare:
    def __init__(self, pos, model, name):
        self.name = name
        self.pos = pos
        self.model = model


class PickPlaceScene(scene.Scene):
    """Base class for all gym environments."""

    def __init__(
        self, isTraining, blockPosRange=[[0.4, 0.95], [-0.4, 0.4]], *sceneArgs, **sceneKwargs
    ):
        """Initialises the table environment.

        Args:
            blockPosRange [float]: range for randomising the block's position.
                                   [[xMin,xMax], [yMin,yMax]]
            sceneArgs: arguments passed to the Scene constructor.
            sceneKwargs: keyword arguments passed to the Scene constructor.

        """
        self.blockPosMinX = blockPosRange[0][0]
        self.blockPosMaxX = blockPosRange[0][1]
        self.blockPosMinY = blockPosRange[1][0]
        self.blockPosMaxY = blockPosRange[1][1]
        super(PickPlaceScene, self).__init__(*sceneArgs, **sceneKwargs)
        self.isTraining = isTraining
        if os.name == 'nt':
            self.stockfish = Stockfish(
                r'C:\Users\fream\Downloads\robot_simulations-master\robot_simulations-master\chess_env\stockfish2.exe')
        else:
            self.stockfish = Stockfish(os.path.abspath(
                "/home/pitsill0s/Desktop/ChessBBRL/robot_simulations-master/chess_env/stockfish_14.1_linux_x64"))
        self.start_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    # Scene extension methods
    # --------------------------------------------------------------------------

    def loadRobot(self, **robotKwargs):
        """Loads a robot into the scene.

        Cannot load a robot after calling setInit.

        Args:
            robotName (string): name of the robot to load into the scene.
            robotKwargs (dict): key word arguments for the robot constructor.

        Returns:
            Robot object.

        """
        if not self.isInitiliased():
            # Create the desired robot
            robot = UR103F(physicsClient=self.p, **robotKwargs)

            # Reset to rest config
            robot.reset(*robot.getRest())

            # Add to list of robots
            self.robots.append(robot)

            # Return robot
            return robot
        else:
            print("Error - Do not load additional models after calling setInit().")

    def _loadScene(self):
        # Load ground plane
        path = os.path.join(self.pybulletModels, "plane.urdf")
        self.p.loadURDF(path)

        # Load table
        path = os.path.join(self.objectModels, "table/table.urdf")
        self.tableId = self.p.loadURDF(
            path, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.7])
        globalOrn = [0, 0, 0]
        globalOrn = self.p.getQuaternionFromEuler(globalOrn)

        path = os.path.join(self.objectModels,
                            "flat_chessboard/flat_chessboard.sdf")
        self.chessboard = self.p.loadSDF(path)

        chessboardOrn = [0, 0, -1.56]
        chessboardOrn = self.p.getQuaternionFromEuler(chessboardOrn)

        self.p.changeDynamics(self.chessboard[0], linkIndex=-1, mass=100.0)
        self.p.resetBasePositionAndOrientation(
            self.chessboard[0], [0.6, 0.0, 0.627], chessboardOrn
        )

        self.black_pieces = []
        self.white_pieces = []
        self.positions = []
        self.posDict = dict()
        self.pieceDict = dict()

        for i in range(8):
            col = []
            for j in range(8):
                col.append([0.835 - (j * 0.067), -0.23 + (i * 0.067), 0.625])
                idx = chr(97+i) + str(1+j)
                item = ChessSquare(
                    pos=[0.835 - (j * 0.067), -0.23 + (i * 0.067), 0.625], model=None, name=idx)
                self.posDict[idx] = item
                print(self.posDict[idx].pos, self.posDict[idx].name)
            self.positions.append(col)

        print(self.posDict)

        self.current_fen_string = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        self.prev_fen_string = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        self.setup_with_fen(self.current_fen_string)

        # Use same constraints for goal
        self.target = [
            random.uniform(self.blockPosMinX, self.blockPosMaxX),
            random.uniform(self.blockPosMinY, self.blockPosMaxY),
            0.7,
        ]

        self.destTarget = [
            random.uniform(self.blockPosMinX, self.blockPosMaxX),
            random.uniform(self.blockPosMinY, self.blockPosMaxY),
            0.7,
        ]

        objects = self.posDict
        return objects

    def setup_with_fen(self, start_fen_string):
        globalOrn = [0, 0, 0]
        globalOrn = self.p.getQuaternionFromEuler(globalOrn)
        print("Initializing board with fen", start_fen_string)
        print(start_fen_string.split(" ")[0].split("/")[0])

        fen_rows = start_fen_string.split(" ")[0].split("/")
        blackPawn = 0
        whitePawn = 0
        whiteRook = 0
        blackRook = 0
        whiteKnight = 0
        blackKnight = 0
        whiteBishop = 0
        blackBishop = 0
        whiteQueen = 0
        blackQueen = 0
        for i in range(len(fen_rows)):
            if fen_rows[i] == "8":
                continue
            else:
                for j in range(len(fen_rows[i])):
                    idx = chr(97+j) + str(8-i)
                    print("i", i)
                    print("j", j)
                    print("idx", idx)
                    print("idx in that pos is", fen_rows[i][j])
                    print(fen_rows[i])
                    if fen_rows[i][j] == "P":
                        self.new_method(globalOrn, whitePawn,
                                        idx, "pawn_white/model.sdf", "P")
                        whitePawn += 1
                    if fen_rows[i][j] == "R":
                        self.new_method(globalOrn, whiteRook,
                                        idx, "rook_white/model.sdf", "R")
                        whiteRook += 1
                    if fen_rows[i][j] == "N":
                        self.new_method(globalOrn, whiteKnight,
                                        idx, "knight_white/model.sdf", "N")
                        whiteKnight += 1
                    if fen_rows[i][j] == "B":
                        self.new_method(globalOrn, whiteBishop,
                                        idx, "bishop_white/model.sdf", "B")
                        whiteBishop += 1
                    if fen_rows[i][j] == "Q":
                        self.new_method(globalOrn, whiteQueen, idx,
                                        "queen_white/model.sdf", "Q")
                        whiteQueen += 1
                    if fen_rows[i][j] == "K":
                        self.new_method(globalOrn, "", idx,
                                        "king_white/model.sdf", "K")
                    if fen_rows[i][j] == "p":
                        self.new_method(globalOrn, blackPawn,
                                        idx, "pawn_black/model.sdf", "p")
                        blackPawn += 1
                    if fen_rows[i][j] == "r":
                        self.new_method(globalOrn, blackRook,
                                        idx, "rook_black/model.sdf", "r")
                        blackRook += 1
                    if fen_rows[i][j] == "n":
                        self.new_method(globalOrn, blackKnight,
                                        idx, "knight_black/model.sdf", "n")
                        blackKnight += 1
                    if fen_rows[i][j] == "b":
                        self.new_method(globalOrn, blackBishop,
                                        idx, "bishop_black/model.sdf", "b")
                        blackBishop += 1
                    if fen_rows[i][j] == "q":
                        self.new_method(globalOrn, blackQueen, idx,
                                        "queen_black/model.sdf", "q")
                        blackQueen += 1
                    if fen_rows[i][j] == "k":
                        self.new_method(globalOrn, "", idx,
                                        "king_black/model.sdf", "k")
        self.add_extra(globalOrn, 2, "bishop_black/model.sdf", "b")
        self.add_extra(globalOrn, 2, "bishop_white/model.sdf", "B")
        self.add_extra(globalOrn, 2, "rook_black/model.sdf", "r")
        self.add_extra(globalOrn, 2, "rook_white/model.sdf", "R")
        self.add_extra(globalOrn, 2, "knight_black/model.sdf", "n")
        self.add_extra(globalOrn, 2, "knight_white/model.sdf", "N")
        self.add_extra(globalOrn, 1, "queen_white/model.sdf", "Q")
        self.add_extra(globalOrn, 1, "queen_black/model.sdf", "q")
        self.add_extra(globalOrn, 2, "queen_white/model.sdf", "Q")
        self.add_extra(globalOrn, 2, "queen_black/model.sdf", "q")


        

    def new_method(self, globalOrn, item, idx, modelName, modelSymbol):
        path = os.path.join(
            self.objectModels, modelName)
        model = self.p.loadSDF(path)
        self.p.resetBasePositionAndOrientation(
            model[0], self.posDict[idx].pos, globalOrn
        )
        name = modelSymbol + str(item)
        self.pieceDict[name] = model
        self.posDict[idx].model = model

    def add_extra(self, globalOrn, item, modelName, modelSymbol):
        path = os.path.join(
            self.objectModels, modelName)
        model = self.p.loadSDF(path)
        self.p.resetBasePositionAndOrientation(
            model[0], [0,0,0], globalOrn
        )
        name = modelSymbol + str(item)
        self.pieceDict[name] = model

    def randomize_with_fen(self, start_fen_string):
        globalOrn = [0, 0, 0]
        globalOrn = self.p.getQuaternionFromEuler(globalOrn)
        print("Reseting board with fen", start_fen_string)
        fen_rows = start_fen_string.split(" ")[0].split("/")
        blackPawn = 0
        whitePawn = 0
        whiteRook = 0
        blackRook = 0
        whiteKnight = 0
        blackKnight = 0
        whiteBishop = 0
        blackBishop = 0
        whiteQueen = 0
        blackQueen = 0
        for key, value in self.pieceDict.items():
            self.p.resetBasePositionAndOrientation(
                value[0], [0, 0, 0], globalOrn
            )
        for key, value in self.posDict.items():
            value.model = None

        for i in range(len(fen_rows)):
            if fen_rows[i] == "1":
                continue
            else:
                for j in range(len(fen_rows[i])):
                    idx = chr(97+j) + str(8-i)
                    if fen_rows[i][j] == "P":
                        self.setupStuff(globalOrn, whitePawn, idx, "P")
                        whitePawn += 1
                    if fen_rows[i][j] == "R":
                        self.setupStuff(globalOrn, whiteRook, idx, "R")
                        whiteRook += 1
                    if fen_rows[i][j] == "N":
                        self.setupStuff(globalOrn, whiteKnight, idx, "N")
                        whiteKnight += 1
                    if fen_rows[i][j] == "B":
                        self.setupStuff(globalOrn, whiteBishop, idx, "B")
                        whiteBishop += 1
                    if fen_rows[i][j] == "Q":
                        self.setupStuff(globalOrn, whiteQueen, idx, "Q")
                        whiteQueen +=1
                    if fen_rows[i][j] == "K":
                        self.setupStuff(globalOrn, "", idx, "K")
                    if fen_rows[i][j] == "p":
                        self.setupStuff(globalOrn, blackPawn, idx, "p")
                        blackPawn += 1
                    if fen_rows[i][j] == "r":
                        self.setupStuff(globalOrn, blackRook, idx, "r")
                        blackRook += 1
                    if fen_rows[i][j] == "n":
                        self.setupStuff(globalOrn, blackKnight, idx, "n")
                        blackKnight += 1
                    if fen_rows[i][j] == "b":
                        self.setupStuff(globalOrn, blackBishop, idx, "b")
                        blackBishop += 1
                    if fen_rows[i][j] == "q":
                        self.setupStuff(globalOrn, blackQueen, idx, "q")
                        blackQueen +=1
                    if fen_rows[i][j] == "k":
                        self.setupStuff(globalOrn, "", idx, "k")
        print("board updated")
        self.objects = self.posDict

    def setupStuff(self, globalOrn, item, idx, string):
        name = string + str(item)
        model = self.pieceDict[name]
        self.p.resetBasePositionAndOrientation(
            model[0], self.posDict[idx].pos, globalOrn
        )

        self.posDict[idx].model = model
        # print("adding" + name + "to " + idx )

    def _randomizePositionFromFen(self):
        lines = open('randomFen.txt').read().splitlines()
        myline = random.choice(lines)

        a, b = myline.split(' ', 1)
        d = {"2": "11", "3": "111", "4": "1111", "5": "11111",
             "6": "111111", "7": "1111111", "8": "11111111"}
        for x, y in d.items():
            a = a.replace(x, y)
        newString = a + " " + b
        return newString

    def _randomise(self):

        armStartPos = [
            random.uniform(self.blockPosMinX, self.blockPosMaxX),
            random.uniform(self.blockPosMinY, self.blockPosMaxY),
            random.uniform(0.9, 1.3),
        ]

        self.robots[0].reset(armStartPos)

        chessboardOrn = [0, 0, -1.56]
        chessboardOrn = self.p.getQuaternionFromEuler(chessboardOrn)
        self.p.resetBasePositionAndOrientation(
            self.chessboard[0], [0.6, 0.0, 0.627], chessboardOrn
        )
        isCheckMate = True
        while isCheckMate:
            self.current_fen_string = self._randomizePositionFromFen()
            self.stockfish.set_fen_position(self.current_fen_string)
            fen = self.stockfish.get_fen_position()
            board = chess.Board(fen)
            if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
                self.current_fen_string = self._randomizePositionFromFen()
            else:
                isCheckMate = False

        if self.isTraining:
            self.randomize_with_fen(self.current_fen_string)
        else:
            self.stockfish.set_fen_position(self.current_fen_string)
            board = chess.Board(self.current_fen_string)
            move = self.stockfish.get_best_move()
            chessMove = chess.Move.from_uci(move)
            board.push(chessMove)
            self.stockfish.set_fen_position(board.fen())
            self.current_fen_string = self._simplify_fen(board.fen())
            self.randomize_with_fen(self.current_fen_string)
        self.objects = self.posDict
        return self.posDict

    def _simplify_fen(myline):
        a, b = myline.split(' ', 1)
        d = {"2": "11", "3": "111", "4": "1111", "5": "11111",
             "6": "111111", "7": "1111111", "8": "11111111"}
        for x, y in d.items():
            a = a.replace(x, y)
        newString = a + " " + b
        return newString
    # Helper functions

    def getNumFingerTipContacts(self, robot_index, block_name):
        """Get the number of finger tips in contact with the block."""
        contactPointsBlock = self.p.getContactPoints(
            self.robots[robot_index].id, self.objects[block_name].model[0]
        )
        fingerTips = self.robots[robot_index].getFingerTipLinks()
        contacts = []
        for contactPoint in contactPointsBlock:
            if contactPoint[3] in fingerTips:
                contacts.append(contactPoint[3])

        numUniqueContacts = len(set(contacts))
        return numUniqueContacts

    def getTarget(self):
        return self.target

    def getDestTarget(self):
        return self.destTarget
