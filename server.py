from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from datetime import datetime

import MR_Version as mr

import uuid


app = Flask(__name__)

# Directory to save received images
SAVE_DIR = "received_images_test"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Variables
boardCorners = None



@app.route('/getBestMove', methods=['POST'])
def getBestMove():
    if(boardCorners is None):
        return "Cheesboard not defined", 500
    
    if 'file' not in request.files:
        return "No file part", 500
    
    file = request.files['file']

    # Convert the received image file to an OpenCV format
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    try:
        response = mr.getBestMove(image, boardCorners)

        if(response == -1):
            return jsonify({"error":"see server logs"}), 500

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/getChessboard', methods=['POST'])
def getChessboard():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']

    # Convert the received image file to an OpenCV format
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    try:
        boardCorners = mr.getBoardCorners(image)

        result_json = {"points": [list(point) for point in boardCorners]}
        return jsonify(result_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Allow external access


