import cv2

from inference_sdk import InferenceHTTPClient
import numpy as np

import chess
import chess.svg
import requests


from IPython.display import clear_output

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
boardCorners = []

### Board corners processing

def extractBoardCorners(points, pattern_size):
    points = points.reshape(-1, 2)

    bottom_left = [int(points[0][0]),  int(points[0][1])]
    top_left = [int(points[pattern_size[0] - 1][0]),
                   int(points[pattern_size[0] - 1][1])]
    bottom_right = [int(points[-pattern_size[0]][0]),
                 int(points[-pattern_size[0]][1])]
    top_right = [int(points[-1][0]),  int(points[-1][1])]

    bottomCellSide = (bottom_right[0] - bottom_left[0]) // 6
    bottom_right[0] += int(bottomCellSide * 1.3)
    bottom_right[1] += int(bottomCellSide * 0.85)
    bottom_left[0] -= int(bottomCellSide * 1.3)
    bottom_left[1] += int(bottomCellSide * 0.85)

    topCellSide = (top_right[0] - top_left[0]) // 6
    top_right[0] += int(topCellSide * 0.95)
    top_right[1] -= int(topCellSide * 0.55)
    top_left[0] -= int(topCellSide * 0.95)
    top_left[1] -= int(topCellSide * 0.55)

    return [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]

def getBoardCorners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (7, 7)

    res, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if res:
        corner_points = extractBoardCorners(corners, pattern_size)
        return corner_points
    else:
        print("Chessboard corners not found.")
        return []
    

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

def converPointToPosition(point, height, width):
    dictionary = {
        7: 'A',
        6: 'B',
        5: 'C',
        4: 'D',
        3: 'E',
        2: 'F',
        1: 'G',
        0: 'H',
    }

    width_section = int(width / 8)
    height_section = int(height / 8)

    row = (point[0] // width_section)
    col = point[1] // height_section

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
        cvt = converPointToPosition(mapped_points[0][i], HEIGHT, WIDTH)
        
        if cvt[0] == '?' or cvt == '-1':
            print(mapped_points[0][i])
            continue

        chessBoardData.append([cvt, data[i][3]])

    print(chessBoardData)

    # Converting pos to fen and sending to the API

    fenData = converToFen(chessBoardData)

    board = chess.Board(fenData)
    board_svg = chess.svg.board(board=board)
    

    response_data = post_chess_api(fenData)
    if response_data:
        best_move = response_data.get("move")
        print("Best move is: ", best_move)

        if best_move:
            move = chess.Move.from_uci(best_move)
            if move in board.legal_moves:
                board.push(move)

            board_svg = chess.svg.board(board=board, lastmove=move, size=720)

            with open("board.svg", "w") as f:
                f.write(board_svg)

        return []
    else:
        print("Error processing the chess_api")
        return -1

    
if __name__ == "__main__":
    image = cv2.imread("test/board.jpg")
    corners = getBoardCorners(image)

    image2 = cv2.imread("test/1.jpg")

    res = getBestMove(image, corners)
    print(res)