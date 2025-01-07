import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, jsonify
import base64
from utilities import get_contours, get_expressions, allowed_file, get_trace, create_png_image, VideoProcessor
import os
from pathlib import Path

app = Flask(__name__)
processor = VideoProcessor()
COLOUR = '#2464b4'

# Create output directory if it doesn't exist
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    expressions = None
    png_image_url = None
    video_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                low_threshold = request.form.get('low_threshold', '').strip()
                high_threshold = request.form.get('high_threshold', '').strip()
                simplification_factor = request.form.get('simplification_factor', 0,00)
                # Use defaults if fields are empty
                low_threshold = int(low_threshold) if low_threshold else 50
                high_threshold = int(high_threshold) if high_threshold else 150
        
                simplification_factor = float(simplification_factor) if simplification_factor else 0.00

                # Validate thresholds
                if high_threshold <= low_threshold:
                    return "High threshold must be greater than low threshold."
                curve_color = request.form.get('curve_color', COLOUR)
                background_color = request.form.get('background_color', '#ffffff')
                output_format = request.form.get('output_format', 'expressions')

                if high_threshold <= low_threshold:
                    return "High threshold must be greater than low threshold."

                file_ext = file.filename.rsplit('.', 1)[1].lower()
                
                if file_ext in {'mp4', 'gif'}:
                    input_path = output_dir / f"input.{file_ext}"
                    output_path = output_dir / f"output.mp4"
                    
                    # Save uploaded file
                    file.save(input_path)
                    
                    # Process video
                    _, fps = processor.process_video(
                        str(input_path),
                        str(output_path),
                        simplification_factor=simplification_factor,
                        curve_color=curve_color,
                        background_color=background_color
                    )
                    
                    # Read and encode processed video
                    with open(output_path, 'rb') as f:
                        video_data = f.read()
                        video_base64 = base64.b64encode(video_data).decode()
                        video_url = f"data:video/mp4;base64,{video_base64}"
                        
                else:
                    # Process single image
                    file_bytes = file.read()
                    npimg = np.frombuffer(file_bytes, np.uint8)
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
                
    return render_template('index.html', 
                         expressions=expressions, 
                         png_image_url=png_image_url,
                         video_url=video_url)

@app.route('/get_recommended_values', methods=['POST'])
def get_recommended_values():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
        
    file = request.files['file']
    # Read file content first
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({'error': 'Empty file'})
        
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    median = np.median(gray)
    std = np.std(gray)

    # Rest of the function remains the same
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized_gray = clahe.apply(gray)

    if mean < 50:
        recommended_low_threshold = int(np.percentile(equalized_gray, 5))
        recommended_high_threshold = int(np.percentile(equalized_gray, 95))
    elif mean > 200:
        inverted_gray = 255 - gray
        recommended_low_threshold = int(np.percentile(inverted_gray, 5))
        recommended_high_threshold = int(np.percentile(inverted_gray, 95))
    else:
        recommended_low_threshold = int(max(0, median * 0.5))
        recommended_high_threshold = int(min(255, median * 1.5))

    if recommended_high_threshold <= recommended_low_threshold:
        recommended_low_threshold = max(0, int(median - std))
        recommended_high_threshold = min(255, int(median + std))

    if recommended_high_threshold <= recommended_low_threshold:
        recommended_low_threshold = max(0, int(median - 50))
        recommended_high_threshold = min(255, int(median + 50))

    recommended_low_threshold = max(0, recommended_low_threshold)
    recommended_high_threshold = min(255, max(recommended_low_threshold + 1, recommended_high_threshold))

    return jsonify({
        'recommended_low_threshold': int(recommended_low_threshold),
        'recommended_high_threshold': int(recommended_high_threshold),
        'recommended_simplification_factor': 0.0
    })

@app.route('/progress')
def get_progress():
    progress = list(processor.get_progress())
    return jsonify({'progress': progress[-1] if progress else 0})

if __name__ == '__main__':
    app.run()