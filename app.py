import cv2
import potrace
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import multiprocessing
import os
import time

app = Flask(__name__)

FRAME_DIR = 'frames'
FILE_EXT = 'png'
COLOUR = '#2464b4'
SCREENSHOT_SIZE = [None, None]
SCREENSHOT_FORMAT = 'png'
OPEN_BROWSER = True

BILATERAL_FILTER = False
DOWNLOAD_IMAGES = False
USE_L2_GRADIENT = False
SHOW_GRID = True

frame = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock=False)
width = multiprocessing.Value('i', 0, lock=False)
frame_latex = 0

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_contours(filename, low_threshold=50, high_threshold=150):
    start_time = time.time()
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nudge = .33

    median = max(10, min(245, np.median(gray)))
    lower = int(max(0, int(1 - nudge) * median))
    upper = int(min(255, int(1 + nudge) * median))
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    print(lower, upper)
    # Flip the image vertically
    gray = cv2.flip(gray, 0)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, lower, upper, L2gradient=USE_L2_GRADIENT)

    with frame.get_lock():
        frame.value += 1
        height.value = max(height.value, image.shape[0])
        width.value = max(width.value, image.shape[1])

    print("Time taken to get contours: " , time.time() - start_time)
    return edges


def get_trace(data, epsilon_factor=0.00):
    # Use RETR_TREE to find all contours, including internal edges
    contours, _ = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    simplified_contours = np.zeros_like(data)
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(simplified_contours, [approx], -1, (255), thickness=cv2.FILLED)

    bmp = potrace.Bitmap(simplified_contours)
    path = bmp.trace(2, potrace.TURNPOLICY_MINORITY, 1.0, 1, .5)
    return path


def get_latex(filename, low_threshold=50, high_threshold=150, epsilon_factor=0.01):
    start_time = time.time()

    latex = []
    contours = get_contours(filename, low_threshold, high_threshold)
    path = get_trace(contours, epsilon_factor)

    for curve in path:
        try:
            segments = curve.segments
            start = curve.start_point
            x0, y0 = start.x, start.y
            for segment in segments:
                if segment.is_corner:
                    x1, y1 = segment.c.x, segment.c.y
                    x2, y2 = segment.end_point.x, segment.end_point.y
                    latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                    latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
                else:
                    x1, y1 = segment.c1.x, segment.c1.y
                    x2, y2 = segment.c2.x, segment.c2.y
                    x3, y3 = segment.end_point.x, segment.end_point.y
                    latex.append('((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),\
                    (1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))' % \
                                 (x0, x1, x1, x2, x1, x2, x2, x3, y0, y1, y1, y2, y1, y2, y2, y3))
                start = segment.end_point
                x0, y0 = start.x, start.y
        except Exception as e:
            print(f"Error processing curve: {e}")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Time taken to create curves: {processing_time:.2f} seconds")

    return latex


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    expressions = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                low_threshold = int(request.form.get('low_threshold', 50))
                high_threshold = int(request.form.get('high_threshold', 150))
                epsilon_factor = float(request.form.get('epsilon_factor', 0.01))

                if high_threshold <= low_threshold:
                    return "High threshold must be greater than low threshold", 400

                expressions = get_expressions(filepath, low_threshold, high_threshold, epsilon_factor)
            except Exception as e:
                return f"An error occurred while processing the file: {e}"
    return render_template('index.html', expressions=expressions)


@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello World!"


if __name__ == '__main__':
    app.run()


def get_expressions(filename, low_threshold=50, high_threshold=150, epsilon_factor=0.01):
    exprid = 0
    exprs = []
    for expr in get_latex(filename, low_threshold, high_threshold, epsilon_factor):
        exprid += 1
        exprs.append({'id': 'expr-' + str(exprid), 'latex': expr, 'color': COLOUR, 'secret': True})
    return exprs
