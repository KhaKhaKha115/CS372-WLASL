import os
import json
import re

# --- CONFIGURATION ---
JSON_FILE_PATH = 'WLASL_v0.3.json'
RAW_VIDEOS_FOLDER = 'raw_videos'
TARGET_GLOSSES = {'book', 'friend', 'go', 'yes', 'no'}
# ---------------------

# 1. Get list of IDs/Hashes already downloaded from your folder
downloaded_ids = set()
if os.path.exists(RAW_VIDEOS_FOLDER):
    for filename in os.listdir(RAW_VIDEOS_FOLDER):
        if filename.endswith('.mp4'):
            file_id = filename.split('.')[0]
            # Add WLASL ID (number) or YouTube Hash (alphanumeric)
            if 5 <= len(file_id) <= 11: 
                downloaded_ids.add(file_id)
print(f"✅ Total videos currently found in '{RAW_VIDEOS_FOLDER}': {len(downloaded_ids)}")

# 2. Compare against the WLASL metadata
missing_videos = []

with open(JSON_FILE_PATH, 'r') as f:
    full_data = json.load(f)

for entry in full_data:
    if entry['gloss'] in TARGET_GLOSSES:
        for inst in entry['instances']:
            wlasl_video_id = str(inst['video_id'])
            url = inst['url']
            is_missing = False

            # Check for WLASL ID (the number)
            if wlasl_video_id not in downloaded_ids:
                
                # Check for YouTube Hash (if the file was saved using the hash)
                if 'youtube' in url or 'youtu.be' in url:
                    # Use reliable regex for hash extraction
                    match = re.search(r'(?:v=|youtu\.be/|embed/)([^"&?\/\s]{11})', url)
                    youtube_hash = match.group(1) if match else None
                    
                    if youtube_hash and youtube_hash not in downloaded_ids:
                        # If neither the WLASL ID nor the YouTube Hash is found, it's missing.
                        is_missing = True
                
                # If it's a simple numeric ID and not found, it's missing.
                elif wlasl_video_id not in downloaded_ids:
                    is_missing = True
            
            if is_missing:
                missing_videos.append({
                    'id': wlasl_video_id, 
                    'url': url, 
                    'gloss': entry['gloss'],
                    'hash_if_youtube': youtube_hash if 'youtube' in url else 'N/A'
                })

# --- FINAL REPORT ---
print("\n=======================================================")
print(f"🛑 FOUND {len(missing_videos)} MISSING VIDEOS FOR MANUAL DOWNLOAD:")
print("=======================================================")

if not missing_videos:
    print("Good news! All expected videos are present in the folder.")
else:
    for video in missing_videos:
        print(f"| {video['gloss'].upper():<8} | ID: {video['id']:<5} | URL: {video['url']}")
    
    # Save the list to a text file for easy reference
    with open('missing_video_list.txt', 'w') as f:
        f.write("ID,GLOSS,URL\n")
        for video in missing_videos:
            f.write(f"{video['id']},{video['gloss']},{video['url']}\n")
    print("\n✅ List also saved to 'missing_video_list.txt' for easy reference.")