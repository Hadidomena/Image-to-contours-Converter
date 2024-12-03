import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, jsonify
import base64
from utilities import get_contours, get_expressions, allowed_file, get_trace, create_png_image
app = Flask(__name__)

COLOUR = '#2464b4'

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
                    _, buffer = cv2.imencode('.png', png_image)
                    png_image_url = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
            except Exception as e:
                return f"An error occurred while processing the file: {e}"
    return render_template('index.html', expressions=expressions, png_image_url=png_image_url)

@app.route('/get_recommended_values', methods=['POST'])
def get_recommended_values():
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate image statistics
    mean = np.mean(gray)
    median = np.median(gray)
    std = np.std(gray)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized_gray = clahe.apply(gray)

    # Determine image type based on characteristics
    def classify_image_brightness():
        if mean < 50:
            return 'very_dark'
        elif mean > 200:
            return 'very_light'
        else:
            return 'normal'

    image_type = classify_image_brightness()

    # Adaptive thresholding strategy
    if image_type == 'very_dark':
        # For very dark images, use percentile-based approach on equalized image
        recommended_low_threshold = int(np.percentile(equalized_gray, 5))
        recommended_high_threshold = int(np.percentile(equalized_gray, 95))
    elif image_type == 'very_light':
        # For very light images, use percentile-based approach with inverted image
        inverted_gray = 255 - gray
        recommended_low_threshold = int(np.percentile(inverted_gray, 5))
        recommended_high_threshold = int(np.percentile(inverted_gray, 95))
    else:
        # For normal images, use median-based approach
        low_multiplier = 0.5
        high_multiplier = 1.5
        recommended_low_threshold = int(max(0, median * low_multiplier))
        recommended_high_threshold = int(min(255, median * high_multiplier))

    # Ensure meaningful threshold separation
    if recommended_high_threshold <= recommended_low_threshold:
        # Fallback strategy if thresholds are too close
        recommended_low_threshold = max(0, int(median - std))
        recommended_high_threshold = min(255, int(median + std))

    # Additional fallback to prevent edge cases
    if recommended_high_threshold <= recommended_low_threshold:
        # Last resort: use fixed offset from median
        recommended_low_threshold = max(0, int(median - 50))
        recommended_high_threshold = min(255, int(median + 50))

    # Validate thresholds
    recommended_low_threshold = max(0, recommended_low_threshold)
    recommended_high_threshold = min(255, max(recommended_low_threshold + 1, recommended_high_threshold))

    return jsonify({
        'recommended_low_threshold': int(recommended_low_threshold),
        'recommended_high_threshold': int(recommended_high_threshold),
        'recommended_simplification_factor': 0.0
    })

if __name__ == '__main__':
    app.run()