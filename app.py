import cv2
import numpy as np
from flask import Flask
import multiprocessing

app = Flask(__name__)

BILATERAL_FILTER = False  # Reduce number of lines with bilateral filter
DOWNLOAD_IMAGES = False  # Download each rendered frame automatically (works best in firefox)
USE_L2_GRADIENT = False  # Creates less edges but is still accurate (leads to faster renders)
SHOW_GRID = True  # Show the grid in the background while rendering

frame = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock=False)
width = multiprocessing.Value('i', 0, lock=False)
frame_latex = 0


def get_contours(filename, nudge=.33):
    image = cv2.imread(filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if BILATERAL_FILTER:
        median = max(10, min(245, np.median(gray)))
        lower = int(max(0, (1 - nudge) * median))
        upper = int(min(255, (1 + nudge) * median))
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        edged = cv2.Canny(filtered, lower, upper, L2gradient=True)
    else:
        edged = cv2.Canny(gray, 30, 200)

    with frame.get_lock():
        frame.value += 1
        height.value = max(height.value, image.shape[0])
        width.value = max(width.value, image.shape[1])

    return edged[::-1]


@app.route('/')
def hello_world():  # put application's code here
    return ('Im not sure'
            'what im doing right now'

            )


if __name__ == '__main__':
    app.run()
