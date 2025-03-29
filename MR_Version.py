import cv2
import cv2.aruco as aruco

import json

from inference_sdk import InferenceHTTPClient
import numpy as np

import matplotlib.pyplot as plt

import chess
import chess.svg
import requests

import base64

def read_api_key(filepath=".secrets"):
    with open(filepath, "r") as file:
        return file.read().strip()

API_KEY = read_api_key()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

WIDTH = 1280
HEIGHT = 960

### Board corners processing

def getBoardCorners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        print(corners)
        print(ids)
        print("Not all ArUco markers detected.")
        return []

    ids = ids.flatten()
    marker_map = {id_: corner for id_, corner in zip(ids, corners)}

    def get_center(corner):
        pts = corner[0]
        return pts.mean(axis=0)

    corner_points = [
        get_center(marker_map[0]),  # top-left
        get_center(marker_map[1]),  # top-right
        get_center(marker_map[2]),  # bottom-left
        get_center(marker_map[3])   # bottom-right
    ]

    return corner_points
    

### Inference block

def parse_data(data):
    res = []
    heighCoef = 2.65

    for prediction in data.get("predictions", []):
        x, y, confidence, class_name = prediction["x"], prediction["y"] + \
            prediction["height"] / \
            heighCoef, prediction["confidence"], prediction["class"]
        res.append([x, y, confidence, class_name])

    return res

def converPointToPosition(point):
    dictionary = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
    }

    width_section = int(WIDTH / 8)
    height_section = int(HEIGHT / 8)

    row = point[1] // height_section
    row = abs(7 - row)
    col = point[0] // width_section

    letter = dictionary.get(int(col), '?')

    if (int(row+1) < 1):
        return "-1"

    res = f"{letter}{int(row+1)}"

    return res

def converToFen(data):
    board = [['1' for _ in range(8)] for _ in range(8)]

    piece_map = {
        'black-pawn': 'p',
        'black-knight': 'n',
        'black-bishop': 'b',
        'black-rook': 'r',
        'black-queen': 'q',
        'black-king': 'k',
        'white-pawn': 'P',
        'white-knight': 'N',
        'white-bishop': 'B',
        'white-rook': 'R',
        'white-queen': 'Q',
        'white-king': 'K',
    }

    for tile, piece in data:
        col = ord(tile[0].upper()) - ord('A')
        row = 8 - int(tile[1])
        board[row][col] = piece_map[piece]

    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen_piece_placement = "/".join(fen_rows)

    fen = f"{fen_piece_placement} w - - 0 1"

    return fen

def warpedToOriginalCoords(x, y, inverse_matrix):
    point = np.array([[[x, y]]], dtype=np.float32)
    original_point = cv2.perspectiveTransform(point, inverse_matrix)
    return original_point[0][0] 

def convertPosToOriginCoordinates(move, inverse_matrix):
    dictionary = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}
    
    width_section = int(WIDTH / 8)
    height_section = int(HEIGHT / 8)

    x_pos = int(dictionary[move[0]] * width_section + width_section / 2)
    y_pos = int((abs(8 - int(move[1]))) * height_section + height_section / 2)

    return warpedToOriginalCoords(x_pos, y_pos, inverse_matrix)


def post_chess_api(data):
    url = "https://chess-api.com/v1"
    headers = {"Content-Type": "application/json"}
    payload = {"fen": data}

    try:
        response = requests.post(url, headers=headers, json=payload)

        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def getBestMove(image, corners):

    result = CLIENT.infer(image, model_id="chessscanner-w3y0i/2")
    data = parse_data(result)

    # Chnaging perspectie of the board

    imageCopy = image.copy()

    src_points = np.array(corners, dtype=np.float32)

    dst_points = np.array([
        [0, 0],
        [WIDTH - 1, 0],
        [0, HEIGHT - 1],
        [WIDTH - 1, HEIGHT - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_matrix = np.linalg.inv(matrix)

    warped_image = cv2.warpPerspective(imageCopy, matrix, (WIDTH, HEIGHT))

    # Mapping piece to new positions

    original_points = []

    for d in data:
        if d[0] > 0 and d[1] > 0:
            original_points.append([d[0], d[1]])

    original_points = np.array(original_points, dtype=np.float32)
    original_points = np.array([original_points])

    if (len(original_points[0]) < 1):
        print("ERROR: No pieces detected")
        return -1

    mapped_points = cv2.perspectiveTransform(original_points, matrix)
    
    finalImage = warped_image.copy()

    for i, mp in enumerate(mapped_points[0]):
        cv2.circle(finalImage, (int(mp[0]), int(mp[1])), 10, (0, 255, 0), -1) 

    label = data[i][3]  
    cv2.putText(
        finalImage, label, (int(mp[0]) + 15, int(mp[1]) - 10),  
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
    )

    ### Creating FEN Board

    chessBoardData = []

    for i in range(len(data)):
        cvt = converPointToPosition(mapped_points[0][i])
        
        if cvt[0] == '?' or cvt == '-1':
            print(mapped_points[0][i])
            continue

        chessBoardData.append([cvt, data[i][3]])

    #### Converting pos to fen and sending to the API

    fenData = converToFen(chessBoardData)
    board = chess.Board(fenData)
    response_data = post_chess_api(fenData)

    if not response_data:
        print("Error processing the chess_api")
        return -1

    response = {"piece":response_data.get("piece"), "to":response_data.get("to"), "eval":response_data.get("eval"), "move": response_data.get("move")}
    best_move = response["move"]

    if not best_move:
        print("No best move detected")
        return -1
    
    move = chess.Move.from_uci(best_move)

    originPoint = convertPosToOriginCoordinates(response["to"], inverse_matrix)

    response["x"] = int(originPoint[0])
    response["y"] = int(originPoint[1])

    if move in board.legal_moves:
        board.push(move)

    board_svg = chess.svg.board(board=board, lastmove=move, size=720)

    svg_base64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")

    response["board"] = f"data:image/svg+xml;base64,{svg_base64}"

    with open("board.svg", "w") as f:
        f.write(board_svg)

    return response