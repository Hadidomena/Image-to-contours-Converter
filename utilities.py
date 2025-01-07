import multiprocessing
import time
import cv2
import numpy as np
import potrace
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp
import shutil
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Union, Tuple, Iterator

COLOUR = '#2464b4'

frame = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock=False)
width = multiprocessing.Value('i', 0, lock=False)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'mp4', 'gif'}

class VideoProcessor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.progress_queue = queue.Queue()
        
    def get_progress(self) -> Iterator[float]:
        while not self.progress_queue.empty():
            yield self.progress_queue.get()

    def process_frame(self, frame_idx, frame, params):
        try:
            low_threshold = params['low_threshold']
            high_threshold = params['high_threshold']
            simplification_factor = params['simplification_factor']
            curve_color = params['curve_color']
            background_color = params['background_color']

            # Get contours using utility function
            contours = get_contours(frame, low_threshold, high_threshold)

            # Get trace from contours
            path = get_trace(contours, simplification_factor)

            # Create an output image using the utility function
            processed_frame = create_png_image(
                original_image=frame,
                path=path,
                curve_color_hex=curve_color,
                background_color_hex=background_color
            )

            return frame_idx, processed_frame
        except Exception as e:
            raise RuntimeError(f"Error processing frame {frame_idx}: {e}")

    def process_video(self, input_path, output_path, simplification_factor, curve_color, background_color):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_dir = Path(mkdtemp())
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Dynamically calculate recommended thresholds for each frame
                    low_threshold, high_threshold = calculate_recommended_thresholds(frame)
                    frame_params = {
                        'low_threshold': low_threshold,
                        'high_threshold': high_threshold,
                        'simplification_factor': simplification_factor,
                        'curve_color': curve_color,
                        'background_color': background_color
                    }

                    future = executor.submit(self.process_frame, frame_idx, frame, frame_params)
                    futures.append(future)
                    frame_idx += 1
                    self.progress_queue.put(frame_idx / frame_count * 100)

                frames = [future.result()[1] for future in futures]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)
        
            out.release()
            cap.release()
            return str(output_path), fps

        finally:
            shutil.rmtree(temp_dir)



def calculate_recommended_thresholds(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    median = np.median(gray)
    std = np.std(gray)

    if mean < 50:
        low_threshold = int(np.percentile(gray, 5))
        high_threshold = int(np.percentile(gray, 95))
    elif mean > 200:
        inverted_gray = 255 - gray
        low_threshold = int(np.percentile(inverted_gray, 5))
        high_threshold = int(np.percentile(inverted_gray, 95))
    else:
        low_threshold = max(0, int(median * 0.5))
        high_threshold = min(255, int(median * 1.5))

    if high_threshold <= low_threshold:
        low_threshold = max(0, int(median - std))
        high_threshold = min(255, int(median + std))

    low_threshold = max(0, low_threshold)
    high_threshold = min(255, max(low_threshold + 1, high_threshold))

    return low_threshold, high_threshold
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_contours(image, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray = cv2.flip(gray, 0)

    # Adaptive Gaussian thresholding
    block_size = 11
    C = 2
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

    # Combine Sobel and Canny edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx ** 2 + sobely ** 2)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edges = cv2.Canny(edges, low_threshold, high_threshold)

    with frame.get_lock():
        frame.value += 1
        height.value = max(height.value, image.shape[0])
        width.value = max(width.value, image.shape[1])

    return edges

def get_trace(data, simplification_factor=0.02):
    contours, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Filter out very small contours
    min_contour_length = 20
    contours = [cnt for cnt in contours if len(cnt) >= min_contour_length]

    simplified_contours = np.zeros(data.shape, dtype=data.dtype)
    all_approx = []
    for contour in contours:
        epsilon = simplification_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        all_approx.append(approx)

    cv2.drawContours(simplified_contours, all_approx, -1, (255), thickness=cv2.FILLED)

    bmp = potrace.Bitmap(simplified_contours)
    try:
        turn_policy = getattr(potrace, 'TURNPOLICY_MINORITY', 1)
    except (AttributeError, NameError):
        turn_policy = 1

    try:
        path = bmp.trace(
            turdsize=2,
            turnpolicy=turn_policy,
            alphamax=0.5,
            opticurve=1,
            opttolerance=0.2
        )
    except TypeError:
        path = bmp.trace(2, alphamax=0.5, opticurve=1, opttolerance=0.2)

    return path

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)

def create_png_image(original_image, path, curve_color_hex, background_color_hex):
    height, width = original_image.shape[:2]
    curve_color = hex_to_bgr(curve_color_hex)
    background_color = hex_to_bgr(background_color_hex)
    output_image = np.full((height, width, 3), background_color, dtype=np.uint8)

    for curve in path:
        for segment in curve.segments:
            if isinstance(segment, potrace.CornerSegment):
                start = (int(segment.c.x), int(segment.c.y))
                end = (int(segment.end_point.x), int(segment.end_point.y))
                cv2.line(output_image, start, end, curve_color, thickness=2)
            elif hasattr(segment, 'c1') and hasattr(segment, 'c2'):
                start = (int(segment.c1.x), int(segment.c1.y))
                control1 = (int(segment.c2.x), int(segment.c2.y))
                end = (int(segment.end_point.x), int(segment.end_point.y))
                cv2.polylines(output_image, [np.array([start, control1, end], dtype=np.int32)], isClosed=False, color=curve_color, thickness=2)

    return output_image[::-1]

@lru_cache(maxsize=32)
def cached_get_latex(image_bytes, low_threshold=50, high_threshold=150, simplification_factor=0.02):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((low_threshold, high_threshold, 3))
    return get_latex(image_array, low_threshold, high_threshold, simplification_factor)

def get_expressions(image, low_threshold=50, high_threshold=150, simplification_factor=0.00):
    exprid = 0
    exprs = []
    for expr in get_latex(image, low_threshold, high_threshold, simplification_factor):
        exprid += 1
        exprs.append({'id': 'expr-' + str(exprid), 'latex': expr, 'color': COLOUR, 'secret': True})
    return exprs

def get_latex(image, low_threshold=50, high_threshold=150, simplification_factor=0.02):
    latex = []
    image = image.astype(np.uint8)
    contours = get_contours(image, low_threshold, high_threshold)
    path = get_trace(contours, simplification_factor)

    for curve in path:
        try:
            segments = curve.segments
            start = curve.start_point
            x0, y0 = start.x, start.y
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
                    latex.append(
                        f'((1-t)((1-t)((1-t){x0}+t{x1})+t((1-t){x1}+t{x2}))+t((1-t){x1}+t{x2})+t((1-t){x2}+t{x3})),'
                        f'(1-t)((1-t)((1-t){y0}+t{y1})+t((1-t){y1}+t{y2}))+t((1-t){y1}+t{y2})+t((1-t){y2}+t{y3})))'
                    )
                start = segment.end_point
                x0, y0 = start.x, start.y
        except Exception as e:
            print(f"Error processing curve: {e}")
            continue
    return latex