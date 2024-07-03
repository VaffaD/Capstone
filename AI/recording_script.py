import threading
import cv2
import datetime
import os
import time

# Function to check if a directory exists and create it if not
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to initialize video writer
def initialize_video_writer(video_directory, fourcc, fps, frame_size):
    output_file = os.path.join(video_directory, get_current_datetime() + '.mp4')
    return cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Function to get the current datetime as a string
def get_current_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Function to determine the actual FPS of the camera
def get_actual_fps(cap, test_frames=120):
    start = time.time()
    for _ in range(test_frames):
        ret, _ = cap.read()
        if not ret:
            print("Failed to capture frames for FPS calculation.")
            return None
    end = time.time()
    calculated_fps = test_frames / (end - start)
    print(f"Calculated camera FPS: {calculated_fps}")
    return calculated_fps

# Function to add a timestamp to frames
def add_timestamp(frame, timestamp_format="%Y-%m-%d %H:%M:%S"):
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# The recording function that takes an event as an argument
def record_video(stop_event, video_directory='recordings'):
    cap = cv2.VideoCapture(2)  # change index if needed
    if not cap.isOpened():
        print("Error: could not open camera 2")
        return

    actual_fps = get_actual_fps(cap)
    if actual_fps is None:
        cap.release()
        return
    print(f"Detected camera FPS: {actual_fps}")

    ensure_directory_exists(video_directory)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frames_per_clip = int(actual_fps * 30)  # Calculate frames for 30 seconds

    out = initialize_video_writer(video_directory, fourcc, actual_fps, frame_size)
    frame_count = 0
    start_time = datetime.datetime.now()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            add_timestamp(frame)  # Add timestamp to each frame
            out.write(frame)
            frame_count += 1

            if frame_count >= frames_per_clip:
                out.release()
                out = initialize_video_writer(video_directory, fourcc, actual_fps, frame_size)
                frame_count = 0
                start_time = datetime.datetime.now()
        else:
            print("Error: could not read frame")
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

