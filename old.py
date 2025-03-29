import cv2
import matplotlib.pyplot as plt

from inference_sdk import InferenceHTTPClient
import numpy as np

import chess
import chess.svg
import requests

from IPython.display import clear_output


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)


def draw_rectangles(image, data):
    for prediction in data.get("predictions", []):
        x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        confidence = prediction["confidence"]
        class_name = prediction["class"]

        top_left = (int(x - width / 2), int(y - height / 2))
        bottom_right = (int(x + width / 2), int(y + height / 2))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        label = f"({x};{y})"
        text_position = (top_left[0], top_left[1] - 10)
        cv2.putText(image, label, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def parse_data(data):
    res = []
    heighCoef = 2.65

    for prediction in data.get("predictions", []):
        x, y, confidence, class_name = prediction["x"], prediction["y"] + \
            prediction["height"] / \
            heighCoef, prediction["confidence"], prediction["class"]
        res.append([x, y, confidence, class_name])

    return res


def findBoardCorners(points, pattern_size):
    points = points.reshape(-1, 2)

    bottom_right = [int(points[0][0] + 150),  int(points[0][1] + 100)]
    bottom_left = [int(points[pattern_size[0] - 1][0] - 150),
                   int(points[pattern_size[0] - 1][1] + 100)]
    top_right = [int(points[-pattern_size[0]][0] + 75),
                 int(points[-pattern_size[0]][1] - 50)]
    top_left = [int(points[-1][0] - 75),  int(points[-1][1] - 50)]

    return [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]


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

# Global variabes
boardCorners = []

def getBestMove(image):    
    width = 1280
    height = 960

    if (len(boardCorners) != 4):
        print("No corners defined. Looking for empty board now")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern_size = (7, 7)

        res, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if res:
            boardCorners = findBoardCorners(corners, pattern_size)
            print("corners are ", boardCorners)
            return ""
        else:
            print("Chessboard corners not found.")

    

        

    result = CLIENT.infer(image, model_id="chessscanner-w3y0i/2")
    data = parse_data(result)

    # Chnaging perspectie of the board

    print("Beginning to change board perspective")

    imageCopy = image.copy()

    width = len(image[0])
    height = len(image)

    src_points = np.array(boardCorners, dtype=np.float32)

    src_points[0] = [src_points[0][0]-0, src_points[0][1]-10]
    src_points[1] = [src_points[1][0]-0, src_points[1][1]-10]
    src_points[2] = [src_points[2][0]-0, src_points[2][1]-0]
    src_points[3] = [src_points[3][0]-0, src_points[3][1]-0]

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(imageCopy, matrix, (width, height))

    print("Board perspective changed")

    # Mapping piece to new positions

    original_points = []

    for d in data:
        if d[0] > 0 and d[1] > 0:
            original_points.append([d[0], d[1]])

    original_points = np.array(original_points, dtype=np.float32)
    original_points = np.array([original_points])

    print("len ", len(original_points[0]))
    if (len(original_points[0]) < 1):
        print("No pieces detected")
        continue

    print(original_points)
    mapped_points = cv2.perspectiveTransform(original_points, matrix)

    
    finalImage = warped_image.copy()



    for i, mp in enumerate(mapped_points[0]):
        cv2.circle(finalImage, (int(mp[0]), int(mp[1])), 10, (0, 255, 0), -1) 

    label = data[i][3]  
    cv2.putText(
        finalImage, label, (int(mp[0]) + 15, int(mp[1]) - 10),  
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
    )

    print("Points mapped")

    # Creating board

    chessBoardData = []

    for i in range(len(data)):
        cvt = converPointToPosition(mapped_points[0][i], 720, 1280)
        
        if cvt[0] == '?' or cvt == '-1':
            print(mapped_points[0][i])
            continue

        chessBoardData.append([cvt, data[i][3]])

    print(chessBoardData)

    # Converting pos to fen and sending to the API

    fenData = converToFen(chessBoardData)

    board = chess.Board(fenData)

    board_svg = chess.svg.board(board=board)

    plt.figure(figsize=(7, 7))
    plt.imshow(cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    response_data = post_chess_api(fenData)
    if response_data:
        best_move = response_data.get("move")
        print("Best move is: ", best_move)
        if best_move:
            # Apply the best move to the board
            move = chess.Move.from_uci(best_move)
            if move in board.legal_moves:
                board.push(move)

            # Highlight the move on the board
            # Adjust size (default is 720)
            board_svg = chess.svg.board(board=board, lastmove=move, size=360)
            clear_output(wait=True)  # Clear the output


            plt.show()            

            display(SVG(board_svg))

    cap.release()
