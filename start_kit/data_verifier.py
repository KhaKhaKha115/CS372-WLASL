import os

# --- CONFIGURATION ---
RAW_VIDEOS_FOLDER = 'raw_videos'
KEYPOINTS_FOLDER = 'keypoints_data'
# ---------------------

def count_files(directory, extension):
    """Counts files with a specific extension in a directory."""
    if not os.path.exists(directory):
        return 0
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            count += 1
    return count

def verify_data_count():
    """Compares the count of raw videos to keypoint data."""
    
    # 1. Count Raw Videos (.mp4)
    video_count = count_files(RAW_VIDEOS_FOLDER, '.mp4')
    
    # 2. Count Keypoint Data (.npy)
    keypoint_count = count_files(KEYPOINTS_FOLDER, '.npy')
    
    # 3. Report Results
    print("\n==============================================")
    print("           DATA INTEGRITY REPORT")
    print("==============================================")
    print(f"📁 Raw Videos Folder: '{RAW_VIDEOS_FOLDER}'")
    print(f"🔢 Total MP4 Videos Found: \t{video_count}")
    print("----------------------------------------------")
    print(f"🧠 Keypoints Folder: '{KEYPOINTS_FOLDER}'")
    print(f"🔢 Total NPY Keypoint Files: \t{keypoint_count}")
    print("==============================================")
    
    # 4. Check for Mismatch
    if video_count == 0 and keypoint_count == 0:
        print("🛑 WARNING: No video or keypoint files found. Check your folder structure.")
    elif video_count > keypoint_count:
        missing_keypoints = video_count - keypoint_count
        print(f"⚠️ ACTION REQUIRED: {missing_keypoints} videos are missing keypoint data.")
        print("Please run 'keypoint_extractor.py' to process the remaining videos.")
    elif video_count == keypoint_count:
        print("✅ SUCCESS: All found videos have corresponding keypoint files!")
    else:
        # Should not happen unless there are extra NPY files from an old run
        print("❓ UNUSUAL: More keypoint files than raw videos. Data is likely consistent.")


if __name__ == '__main__':
    verify_data_count()