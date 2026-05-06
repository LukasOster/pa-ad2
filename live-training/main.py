import cv2
import threading
import subprocess
import importlib.util
import imagingcontrol4 as ic4
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from src.create_new_dataset import create_timestamped_folder
from src.folder_utils import get_subfolders_with_count, export_data
import shutil
import os
import shutil
import psutil

username = os.getlogin()

model_path = os.path.expanduser("~/Desktop/models/")
dataset_path = os.path.expanduser("~/Desktop/datasets/")

ic4.Library.init()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables
try:
    grabber = ic4.Grabber()
    first_device_info = ic4.DeviceEnum.devices()[0]
    sink = None
    camera_opened = False  # Global flag to track camera state
    lock = threading.Lock()  # Prevent concurrent access
    current_filter = None  # Global variable to track the current filter
    use_filter = False  # Global flag for filter application
    filters = {}  # Dictionary to store loaded filters
    
except:
    pass

def initialize_camera():
    """Initializes the camera only if it is not already open."""
    global grabber, sink, camera_opened
    with lock:
        if camera_opened:
            print("Camera is already initialized.")
            return
        try:
            print("Initializing camera...")
            print(f"Device info: {first_device_info}")
            print("Attempting to open device...")
            grabber.device_open(first_device_info)
            print("Camera opened successfully.")
            grabber.device_property_map.set_value(ic4.PropId.WIDTH, 640)
            grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 480)
            grabber.device_property_map.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, 30)
            print("cam width is: ")
            print(grabber.device_property_map.get_value_int(ic4.PropId.WIDTH))
            print("cam height is: ")
            print(grabber.device_property_map.get_value_int(ic4.PropId.HEIGHT))
            print("Camera properties set successfully.")
            sink = ic4.SnapSink()
            print("Sink created successfully.")
            grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)
            print("Stream setup successfully.")
            camera_opened = True
        except Exception as e:
            print(f"Error initializing camera: {e}")

def close_camera():
    """Safely closes the camera stream and releases the device."""
    global grabber, sink, camera_opened
    with lock:
        if not camera_opened:
            print("Camera is already closed.")
            return

        print("Closing camera...")
        grabber.stream_stop()
        grabber.device_close()
        camera_opened = False

def load_filters(directory):
    """Loads filter modules from the specified directory."""
    global filters
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            filter_module_path = os.path.join(folder_path, 'filter.py')
            if os.path.exists(filter_module_path):
                spec = importlib.util.spec_from_file_location(folder_name, filter_module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                filters[folder_name] = module.apply_filter

def apply_filter(frame, filter_name):
    """Applies the specified filter to the frame."""
    if filter_name in filters:
        return filters[filter_name](frame)
    else:
        return frame
    
@app.route('/move_folder', methods=['POST'])
def move_folder():
    data = request.get_json()
    source_folder_path = data.get("folder_path")
    target_directory = model_path  # Replace with your target directory

    if not source_folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    try:
        # Ensure the target directory exists
        #os.makedirs(target_directory, exist_ok=True)

        # Move the folder to the target directory
        shutil.move(source_folder_path, target_directory)

        return jsonify({"message": f"Folder moved successfully to {target_directory}!"}), 200

    except Exception as e:
        print(f"Error moving folder: {e}")
        return jsonify({"error": str(e)}), 500

def assign_filter():
    """Yields frames for streaming, applying the selected filter if use_filter is true."""
    global sink, grabber, camera_opened, current_filter, use_filter

    # Stop existing stream before starting a new one
    if camera_opened:
        close_camera()

    initialize_camera()  # Ensure camera is initialized before streaming

    try:
        while True:
            try:
                if sink is None:
                    raise Exception("Sink is not initialized.")

                # Capture frame
                image = sink.snap_single(1000)
                frame = image.numpy_wrap()
                raw_frame = frame.copy()
                success = True
            except ic4.IC4Exception as ex:
                print("Camera error:", ex.message)
                success = False

            if not success:
                break
            else:
                # Apply the selected filter if use_filter is true
                if use_filter:
                    frame = apply_filter(frame, current_filter)

                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print("Error in assign_filter:", str(e))

    finally:
        close_camera()  # Ensure the camera is safely closed

@app.route('/set_filter', methods=['POST'])
def set_filter():
    """Sets the current filter type."""
    global current_filter, use_filter
    data = request.json
    filter_type = data.get('filter_type')
    if filter_type:
        current_filter = filter_type
        use_filter = True  # Enable filter application
        return jsonify({"status": "success", "filter_type": current_filter})
    else:
        return jsonify({"status": "error", "message": "Filter type not provided"}), 400

@app.route('/disable_filter', methods=['POST'])
def disable_filter():
    """Disables the current filter."""
    global use_filter
    use_filter = False
    return jsonify({"status": "success", "use_filter": use_filter})

def check_usb_devices():
    """Check for plugged-in USB storage devices and copy folders containing filter.py to ./Desktop/models/."""
    usb_devices = []

    # Manually specify the USB device mount point
    usb_mount_point = "/media/pa-mv1/ESD-USB"
    if os.path.ismount(usb_mount_point):
        usb_devices.append(usb_mount_point)
        print(f"Found USB device: {usb_mount_point}")
    else:
        print(f"No USB device found at {usb_mount_point}")

    # Search for filter.py in USB devices
    for device in usb_devices:
        print(f"Searching in device: {device}")
        for root, dirs, files in os.walk(device):
            if 'filter.py' in files:
                # Copy the folder containing filter.py to the target directory
                target_directory = os.path.expanduser("~/Desktop/models/")
                os.makedirs(target_directory, exist_ok=True)
                shutil.copytree(root, os.path.join(target_directory, os.path.basename(root)))
                print(f"Copied {root} to {target_directory}")

    if not usb_devices:
        print("No USB devices found.")

@app.route('/import_model', methods=['POST'])
def import_model():
    """Endpoint to check USB devices and import models."""
    try:
        check_usb_devices()
        return jsonify({"message": "Models imported successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/load_filters', methods=['GET'])
def load_filters_route():
    """Loads filters from the models directory."""
    models_directory = model_path
    load_filters(models_directory)
    return jsonify({"status": "success", "filters": list(filters.keys())})

@app.route('/restart', methods=['POST'])
def restart_system():
    try:
        # Restart the system (requires sudo permission)
        subprocess.run(["sudo", "reboot"], check=True)
        return jsonify({"message": "Restart command sent"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_models', methods=['GET'])
def get_models():
    models_directory = model_path
    subfolders = [f.name for f in os.scandir(models_directory) if f.is_dir()]
    print(subfolders)
    return jsonify(subfolders)

@app.route('/subfolders', methods=['GET'])
def list_subfolders():
    directory_path = dataset_path
    print("Absolute path:", os.path.abspath(directory_path))

    try:
        folder_data = get_subfolders_with_count(directory_path)
        return jsonify({"folders": folder_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.json
    model_name = data.get('model_name')
    if model_name:
        # Implement your model running logic here
        print(f"Running model: {model_name}")
        return jsonify({"status": "success", "model_name": model_name})
    else:
        return jsonify({"status": "error", "message": "Model name not provided"}), 400

@app.route('/export', methods=['POST'])
def export_folder():
    data = request.get_json()
    folder_path = data.get("folder_path")

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    try:
        full_dest_path = "/media/" + username + folder_path  # Convert to absolute path
        print(full_dest_path)
        full_source_path = dataset_path
        print(full_source_path)

        # Copy the directory
        export_data(full_source_path, full_dest_path)

        return jsonify({"message": f"Folder '{folder_path}' exported successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Handles video streaming, respecting the global `use_filter` flag."""
    return Response(assign_filter(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-daq', methods=['POST'])
def toggle_filter():
    """Toggles motion detection ON/OFF automatically."""
    global use_filter
    use_filter = not use_filter  # Toggle state
    return jsonify({"success": True, "use_filter": use_filter})

@app.route('/filter_status', methods=['GET'])
def filter_status():
    """Returns the current status of the filter."""
    return jsonify({"use_filter": use_filter})

@app.route('/status')
def status():
    """Endpoint to check server status."""
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    print("Starting Flask video stream server...")
    initialize_camera()  # Pre-initialize the camera
    try: 
        load_filters(model_path)  # Load filters at startup
    except: print("Model directory not available")
    app.run(debug=False, host='0.0.0.0', port=5000)
