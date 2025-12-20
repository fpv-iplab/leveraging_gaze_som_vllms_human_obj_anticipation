# ------------------------------------------------- #
# This script extracts all the individual clips of the 
# HD-EPIC Interaction Anticipation benchmark from the 
# videos. The start and end timestamps of the video clips
# have already been extracted and can be found in the
# src/data_processing/clip_timestamps.csv file.

# If you're replicating the experiments in the paper,
# carefully follow the instructions in the repo's README.md
# ------------------------------------------------- #

# Any recent version (2025) of pandas and cv2 should be fine here.
# The specific versions used in our environment are: pandas==2.3.2, opencv-python==4.12.0.88

import argparse
import pandas as pd
import os
import cv2

def hmsms_to_seconds(hmsms: str) -> float:
    """
    Convert a time string in HH:MM:SS.ms format to total seconds as a float.
    """
    hours, minutes, seconds = hmsms.split(":")
    sec, milliseconds = seconds.split(".")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(sec) + int(milliseconds) / 1000
    return total_seconds

def extract_video_segment(input_video_path, output_video_path, end_time_s, clip_length, fps=30):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps > 0:
        fps = video_fps

    num_frames = int(round(clip_length * fps))
    end_frame = int(round(end_time_s * fps))
    if end_frame > total_frames - 1:
        end_frame = total_frames - 1

    start_frame = max(0, end_frame - num_frames + 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_written = 0
    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i} from {input_video_path}")
            break
        out.write(frame)
        frames_written += 1
        if frames_written >= num_frames:
            break

    cap.release()
    out.release()
    print(f"Extracted {frames_written} frames to {output_video_path}")

def extract_all_video_segments(input_videos_folder_path: str, videos_folder_path: str, timestamps: pd.DataFrame):
    output_paths_list = []
    for index, row in timestamps.iterrows():
        print("Extracting: ", row)
        end_time_s = hmsms_to_seconds(row['end_time'])
        start_time_s = hmsms_to_seconds(row['start_time'])
        clip_length = end_time_s - start_time_s

        video_id = row['video_id']
        participant = video_id.split("-")[0]
        input_video_path = os.path.join(input_videos_folder_path, f"{participant}/{video_id}.mp4")
        output_video_path = os.path.join(videos_folder_path, f"{participant}/{video_id}_{index}.mp4")
        if os.path.exists(output_video_path):
            print(f"{output_video_path} already exists... skipping")
            continue
        output_paths_list.append(output_video_path)
        extract_video_segment(input_video_path, output_video_path, end_time_s, clip_length)
    return output_paths_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_timestamps_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    # make sure clips_timestamps_path is a .csv file
    if not args.clips_timestamps_path.endswith(".csv"):
        raise ValueError("clips_timestamps_path must be a .csv file")
    # make sure dataset_path is a valid path
    if not os.path.exists(args.dataset_path):
        raise ValueError("dataset_path must be a valid path")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # insert your local path here
    clips_timestamps = pd.read_csv(args.clips_timestamps_path)    
    extract_all_video_segments(args.dataset_path, args.output_path, clips_timestamps)
    
if __name__ == "__main__":
    main()