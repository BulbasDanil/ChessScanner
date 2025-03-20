from flask import Flask, request
import os
import numpy as np
import cv2
from datetime import datetime

app = Flask(__name__)

# Directory to save received images
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    
    # Save the received image
    file = request.files['file']
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    file.save(filepath)

    # Convert the received image file to an OpenCV format (without saving it)
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    

    return f"Image received and saved as {filename}", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Allow external access
