# move_missing_files.py

import os
import shutil
import csv
from urllib.parse import urlparse

# --- CONFIGURATION (MUST CHANGE THESE!) ---
# 1. Path to the folder containing the 21,000+ videos you downloaded from Kaggle.
KAGGLE_SOURCE_DIR = "/Users/khakhakha/wlasl-complete/videos" 

# 2. Path to your project's destination folder.
PROJECT_DESTINATION_DIR = "raw_videos" 

# 3. File containing the list of missing IDs
MISSING_LIST_FILE = "missing_video_list.txt"
# ------------------------------------------

def copy_files_from_list():
    """Reads the missing list and copies the files from the source to the destination."""
    
    # 1. Ensure directories exist
    if not os.path.exists(KAGGLE_SOURCE_DIR):
        print(f"🛑 ERROR: Kaggle Source Directory not found: {KAGGLE_SOURCE_DIR}")
        print("Please check and update the KAGGLE_SOURCE_DIR variable.")
        return

    os.makedirs(PROJECT_DESTINATION_DIR, exist_ok=True)
    
    # 2. Read the missing list
    missing_ids = []
    try:
        with open(MISSING_LIST_FILE, 'r') as f:
            # Skip the header line (ID,GLOSS,URL)
            next(f) 
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    missing_ids.append(row[0].strip()) # Only need the ID (first column)
    except FileNotFoundError:
        print(f"🛑 ERROR: Missing list file '{MISSING_LIST_FILE}' not found. Did you run find_missing_videos.py?")
        return

    print(f"Found {len(missing_ids)} IDs to search and copy.")
    copied_count = 0
    
    # 3. Iterate and copy files
    for video_id in missing_ids:
        # Check for MP4 files (most common format)
        source_mp4 = os.path.join(KAGGLE_SOURCE_DIR, f"{video_id}.mp4")
        
        if os.path.exists(source_mp4):
            destination_path = os.path.join(PROJECT_DESTINATION_DIR, f"{video_id}.mp4")
            shutil.copy2(source_mp4, destination_path)
            copied_count += 1
            print(f"Copied: {video_id}.mp4")
        
        # NOTE: We skip .swf files and other non-standard formats (like the Yale URLs)
        # because the Kaggle download often only contains the standardized .mp4 files.
        # These SWF files usually don't contain signer motion data needed for keypoints.
        else:
            print(f"Skipped/Not found: ID {video_id} (MP4 not found in source directory).")

    print(f"\n=======================================================")
    print(f"✅ FINISHED: Successfully copied {copied_count} files.")
    print(f"Total videos in '{PROJECT_DESTINATION_DIR}' is now: {len(os.listdir(PROJECT_DESTINATION_DIR))}")
    print("=======================================================")


if __name__ == '__main__':
    copy_files_from_list()