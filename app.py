import multiprocessing
import time
import cv2
import numpy as np
import potrace
from flask import Flask, request, render_template, redirect
import base64
from functools import lru_cache

app = Flask(__name__)

COLOUR = '#2464b4'
USE_L2_GRADIENT = False

frame = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock=False)
width = multiprocessing.Value('i', 0, lock=False)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_contours(image, low_threshold=50, high_threshold=150):
    # Convert to grayscale with explicit dtype
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray = cv2.flip(gray, 0)

    if request.form.get('recommended') == 'true':
        # Simplified adaptive thresholding - faster than computing median
        mean = float(np.mean(gray))
        low_threshold = int(max(0, 0.67 * mean))
        high_threshold = int(min(255, 1.33 * mean))

        if mean < 200:  # Only blur if image isn't mostly white
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use FAST_GRADIENT instead of L2_GRADIENT for speed
    edges = cv2.Canny(gray, low_threshold, high_threshold,
                      apertureSize=3,  # Smaller aperture size
                      L2gradient=False)

    with frame.get_lock():
        frame.value += 1
        height.value = max(height.value, image.shape[0])
        width.value = max(width.value, image.shape[1])

    return edges

def get_trace(data, simplification_factor=0.02):
    # Use CHAIN_APPROX_TC89_KCOS for faster contour approximation
    contours, _ = cv2.findContours(data, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_TC89_KCOS)

    # Filter out very small contours - they're likely noise
    min_contour_length = 20 # Can be adjusted
    contours = [cnt for cnt in contours if len(cnt) >= min_contour_length]

    # Create zeros array with same shape and type
    simplified_contours = np.zeros(data.shape, dtype=data.dtype)

    # Batch process contours for better performance
    all_approx = []
    for contour in contours:
        epsilon = simplification_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        all_approx.append(approx)

    # Draw all contours at once
    cv2.drawContours(simplified_contours, all_approx, -1, (255), thickness=cv2.FILLED)

    # Optimize potrace parameters for speed
    # Use a more generic approach to handle different potrace library versions
    bmp = potrace.Bitmap(simplified_contours)
    
    # Try to handle different potrace library versions
    try:
        # Try the specific constant first
        turn_policy = getattr(potrace, 'TURNPOLICY_MINORITY', 1)
    except (AttributeError, NameError):
        # Fallback to a default turn policy
        turn_policy = 1  # Typically represents a default conservative turn policy

    try:
        path = bmp.trace(
            turdsize=2,  # remove speckles
            turnpolicy=turn_policy,
            alphamax=0.5,  # More aggressive corner detection
            opticurve=1,
            opttolerance=0.2  # Increased tolerance for optimization
        )
    except TypeError:
        # Fallback if some parameters are not supported
        path = bmp.trace(
            2,  # turdsize: remove speckles
            alphamax=0.5,  # More aggressive corner detection
            opticurve=1,
            opttolerance=0.2  # Increased tolerance for optimization
        )

    return path

def hex_to_bgr(hex_color):
    # Convert hex color string to a BGR tuple
    hex_color = hex_color.lstrip('#')  # Remove the '#' if it's there
    return int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)  # BGR format

def create_png_image(original_image, path, curve_color_hex, background_color_hex):
    # Create a blank image with the same dimensions as the original
    height, width = original_image.shape[:2]

    # Convert hex colors to BGR tuples
    curve_color = hex_to_bgr(curve_color_hex)
    background_color = hex_to_bgr(background_color_hex)

    # Create the output image with the specified background color
    output_image = np.full((height, width, 3), background_color, dtype=np.uint8)

    # Draw the curves on the output image
    for curve in path:
        for segment in curve.segments:
            if isinstance(segment, potrace.CornerSegment):
                # For CornerSegment, use the control point and the end point
                start = (int(segment.c.x), int(segment.c.y))
                end = (int(segment.end_point.x), int(segment.end_point.y))

                # Draw the corner segment
                cv2.line(output_image, start, end, curve_color, thickness=2)

            elif hasattr(segment, 'c1') and hasattr(segment, 'c2'):
                # This is likely a Bezier segment (assumed from having c1 and c2 attributes)
                start = (int(segment.c1.x), int(segment.c1.y))
                control1 = (int(segment.c2.x), int(segment.c2.y))
                end = (int(segment.end_point.x), int(segment.end_point.y))

                # Draw the curve using the Bezier control points
                cv2.polylines(output_image, [np.array([start, control1, end], dtype=np.int32)], isClosed=False, color=curve_color, thickness=2)

    return output_image[::-1]

def image_to_hashable(image):
    """Convert image to a hashable representation"""
    # Convert to bytes and include shape information
    return (image.tobytes(), image.shape, image.dtype)

def get_latex(image, low_threshold=50, high_threshold=150, simplification_factor=0.02):
    start_time = time.time()
    latex = []

    # Ensure image is uint8
    image = image.astype(np.uint8)

    contours = get_contours(image, low_threshold, high_threshold)
    path = get_trace(contours, simplification_factor)

    for curve in path:
        try:
            segments = curve.segments
            start = curve.start_point
            x0, y0 = start.x, start.y

            # Pre-calculate common expressions
            for segment in segments:
                if segment.is_corner:
                    x1, y1 = segment.c.x, segment.c.y
                    x2, y2 = segment.end_point.x, segment.end_point.y
                    latex.append(f'((1-t){x0}+t{x1},(1-t){y0}+t{y1})')
                    latex.append(f'((1-t){x1}+t{x2},(1-t){y1}+t{y2})')
                else:
                    x1, y1 = segment.c1.x, segment.c1.y
                    x2, y2 = segment.c2.x, segment.c2.y
                    x3, y3 = segment.end_point.x, segment.end_point.y
                    # Use f-strings for faster string formatting
                    latex.append(
                        f'((1-t)((1-t)((1-t){x0}+t{x1})+t((1-t){x1}+t{x2}))+t((1-t){x1}+t{x2})+t((1-t){x2}+t{x3})),'
                        f'(1-t)((1-t)((1-t){y0}+t{y1})+t((1-t){y1}+t{y2}))+t((1-t){y1}+t{y2})+t((1-t){y2}+t{y3})))'
                    )
                start = segment.end_point
                x0, y0 = start.x, start.y
        except Exception as e:
            print(f"Error processing curve: {e}")
            continue  # Skip problematic curves instead of breaking

    end_time = time.time()
    print(f"Time taken to create curves: {end_time - start_time:.2f} seconds")

    return latex

@lru_cache(maxsize=32)
def cached_get_latex(image_bytes, low_threshold=50, high_threshold=150, simplification_factor=0.02):
    """Cached version of get_latex that uses a hashable image representation"""
    # Reconstruct numpy array from bytes
    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((low_threshold, high_threshold, 3))
    return get_latex(image_array, low_threshold, high_threshold, simplification_factor)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    expressions = None
    png_image_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                low_threshold = int(request.form.get('low_threshold', 50))
                high_threshold = int(request.form.get('high_threshold', 150))
                simplification_factor = float(request.form.get('simplification_factor', 0.00))
                curve_color = request.form.get('curve_color', COLOUR)
                background_color = request.form.get('background_color', '#ffffff')
                output_format = request.form.get('output_format', 'expressions')

                if high_threshold <= low_threshold:
                    return "High threshold must be greater than low threshold."

                npimg = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

                if output_format == 'expressions':
                    expressions = get_expressions(img, low_threshold, high_threshold, simplification_factor)
                elif output_format == 'png':
                    contours = get_contours(img, low_threshold, high_threshold)
                    path = get_trace(contours, simplification_factor)
                    png_image = create_png_image(img, path, curve_color, background_color)

                    # Encode the image to PNG format and convert it to a base64 string
                    _, buffer = cv2.imencode('.png', png_image)
                    png_image_url = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

            except Exception as e:
                return f"An error occurred while processing the file: {e}"
    return render_template('index.html', expressions=expressions, png_image_url=png_image_url)

def get_expressions(image, low_threshold=50, high_threshold=150, simplification_factor=0.00):
    exprid = 0
    exprs = []
    for expr in get_latex(image, low_threshold, high_threshold, simplification_factor):
        exprid += 1
        exprs.append({'id': 'expr-' + str(exprid), 'latex': expr, 'color': COLOUR, 'secret': True})
    return exprs

if __name__ == '__main__':
    app.run()