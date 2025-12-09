import json
import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from multiprocessing import Pool
import re  # <-- Thêm để extract YouTube ID

# --- CONFIGURATION ---
JSON_FILE_PATH = 'WLASL_v0.3.json'
TARGET_GLOSSES = {'book', 'friend', 'go', 'yes', 'no'}
INPUT_FOLDER = 'raw_videos'
OUTPUT_FOLDER = 'keypoints_data'

# Optimization settings
NUM_PROCESSES = 6
FRAME_SKIP = 1
MIN_CONFIDENCE = 0.3

# --- SETUP OUTPUT FOLDER ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- 1. LOAD AND PREPARE METADATA (JSON FIXES) ---
try:
    with open(JSON_FILE_PATH, 'r') as f:
        full_data = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_FILE_PATH}")
    sys.exit(1)

# --- Helper function to extract YouTube ID ---
YOUTUBE_REGEX = re.compile(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?]|$)")
def extract_youtube_id(url):
    match = YOUTUBE_REGEX.search(url)
    if match:
        return match.group(1)
    return None

# 1. Main Map: WLASL_ID -> Metadata
metadata_map = {}
# 2. Secondary Map: YOUTUBE_HASH -> WLASL_ID
youtube_hash_map = {}

for entry in full_data:
    if entry['gloss'] in TARGET_GLOSSES:
        for inst in entry['instances']:
            wlasl_video_id = str(inst['video_id'])
            
            # Store primary metadata
            metadata_map[wlasl_video_id] = {
                'start': inst['frame_start'],
                'end': inst['frame_end']
            }
            
            # YouTube mapping
            yt_id = extract_youtube_id(inst['url'])
            if yt_id:
                youtube_hash_map[yt_id] = wlasl_video_id

print(f"Metadata loaded. Total WLASL instances mapped: {len(metadata_map)}")

# --- 2. CORE PROCESSING FUNCTION ---
def process_single_video(filename, metadata_map, youtube_hash_map, INPUT_FOLDER, OUTPUT_FOLDER, FRAME_SKIP, MIN_CONFIDENCE):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_CONFIDENCE
    )

    file_id = filename.split('.')[0]
    wlasl_video_id = None

    if file_id in metadata_map:
        wlasl_video_id = file_id
    elif file_id in youtube_hash_map:
        wlasl_video_id = youtube_hash_map[file_id]

    if not wlasl_video_id:
        return f"SKIP: Metadata not found for file {filename}."

    meta = metadata_map[wlasl_video_id]
    start_frame = meta['start']
    end_frame = meta['end']

    video_path = os.path.join(INPUT_FOLDER, filename)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"ERROR: Could not open video file: {filename}"
    except Exception as e:
        return f"ERROR: Exception reading file {filename}: {e}"

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_data = []
    current_frame_index = start_frame

    while cap.isOpened():
        if end_frame != -1 and current_frame_index >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame_index % FRAME_SKIP != 0:
            current_frame_index += 1
            continue

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        full_keypoints = np.zeros(126, dtype=np.float32)

        if results.multi_hand_landmarks:
            keypoints = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
            if len(keypoints) > 126:
                keypoints = keypoints[:126]
            full_keypoints[:len(keypoints)] = keypoints
            frame_data.append(full_keypoints)

        current_frame_index += 1

    cap.release()

    if frame_data:
        output_file = os.path.join(OUTPUT_FOLDER, filename.replace(os.path.splitext(filename)[1], '.npy'))
        np.save(output_file, np.array(frame_data))
        return f"SUCCESS: Saved {filename}. Frames: {len(frame_data)}."
    else:
        return f"WARNING: No keypoints found for {filename}."

# --- 3. MULTIPROCESSING EXECUTION ---
if __name__ == '__main__':
    video_files = os.listdir(INPUT_FOLDER)
    tasks = [
        (filename, metadata_map, youtube_hash_map, INPUT_FOLDER, OUTPUT_FOLDER, FRAME_SKIP, MIN_CONFIDENCE)
        for filename in video_files
        if filename.endswith(('.mp4', '.avi', '.swf'))
    ]

    print(f"Starting parallel keypoint extraction of {len(tasks)} videos using {NUM_PROCESSES} cores...")
    print(f"Sampling rate: 1 frame processed every {FRAME_SKIP} frames.")

    with Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(process_single_video, tasks)

    for result in results:
        print(result)

    print("\nParallel keypoint extraction finished. Proceed to Task 5: Data Preparation.")
