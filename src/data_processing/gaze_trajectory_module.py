# ------------------------------------------------- #
# This script extracts the gaze data from the VRS
# files in the HD-EPIC dataset and applies a gaze
# trajectory to each frame of the clips in the 
# Interaction Anticipation benchmark.

# If you're replicating the experiments in the paper,
# just replace all the paths with your local ones and
# carefully follow all the instructions in the README.md file.
# ------------------------------------------------- #

import argparse
import cv2
import pandas as pd
import os
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection, get_nearest_eye_gaze
from datetime import datetime

CLIP_LENGTH = 10.0

def timestamp_to_nanoseconds(timestamp_str) -> int:
    """Convert a timestamp string (H:M:S.ffffff) to microseconds (int)."""

    time_obj = datetime.strptime(timestamp_str, "%H:%M:%S.%f") - datetime(1900, 1, 1)
    return int(time_obj.total_seconds() * 1e9)

def seconds_to_nanoseconds(time: float) -> int:
    """Convert seconds to nanoseconds."""
    return int(time * 1e9)

def apply_gaze_data_to_segments(video_file_path, gaze_cpf, output_folder, vrs_file_path, video_segments_base_folder, participant, interaction_anticipation_segments):
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file_path)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    video_name = os.path.basename(video_file_path).split(".")[0]
    for index, row in interaction_anticipation_segments.iterrows():
        if row["video_id"] == video_name:
            start_time = row["start_time"] # string in HH:MM:SS.ms format
            start_timestamp = timestamp_to_nanoseconds(start_time) # convert to nanoseconds
            print(f"Start timestamp for segment {index}: {start_timestamp} nanoseconds")
            print(f"First gaze data entry of video {video_name}: {int(gaze_cpf[0].tracking_timestamp.total_seconds() * 1e9)}")
            first_gaze_timestamp = int(gaze_cpf[0].tracking_timestamp.total_seconds() * 1e9)

            video_segment_path = os.path.join(video_segments_base_folder, f"P0{participant}/{video_name}_{index}.mp4")
            if not os.path.exists(video_segment_path):
                print(f"Warning: Video segment file does not exist: {video_segment_path}")
                continue

            # Apply gaze data to a copy of the video segment
            output_file = os.path.join(output_folder, f"P0{participant}/{video_name}_{index}.mp4")
            if os.path.exists(output_file):
                print(f"Output file already exists: {output_file}")
                continue
            cap = cv2.VideoCapture(video_segment_path)
            if not cap.isOpened():
                print(f"Error: Could not open video segment file {video_segment_path}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width, frame_height = 1408, 1408
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print(f"Error: Could not create output video file {output_file}")
                cap.release()
                continue
            frame_idx = 0
            gaze_points_coords = list()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Compute current frame timestamp and nearest gaze
                frame_time_us = int((frame_idx / fps) * 1e9) + start_timestamp + first_gaze_timestamp
                eye_gaze_info = get_nearest_eye_gaze(gaze_cpf, frame_time_us)
                if eye_gaze_info:
                    # Re-project gaze for current frame
                    gaze_projection = get_gaze_vector_reprojection(
                        eye_gaze_info,
                        rgb_stream_label,
                        device_calibration,
                        rgb_camera_calibration,
                        eye_gaze_info.depth
                    )
                    if gaze_projection is not None:
                        x, y = int(gaze_projection[0]), int((1-(gaze_projection[1]/frame_height))*frame_height)
                        gaze_points_coords.append((x,y))
                        if len(gaze_points_coords) > 15:
                            gaze_points_coords.pop(0)
                        
                # Draw trajectory as dots connected by lines, with fading thickness
                if len(gaze_points_coords) > 1:
                    for i in range(len(gaze_points_coords) - 1):
                        alpha = (i + 1) / len(gaze_points_coords)
                        color = (
                            int(255 * (1 - alpha)),  # Blue → Red
                            0,
                            int(255 * alpha)
                        )
                        thickness = max(2, int(8 * alpha))  # slightly bigger lines
                        pt1 = gaze_points_coords[i]
                        pt2 = gaze_points_coords[i + 1]
                        cv2.line(frame, pt1, pt2, color, thickness)

                # Draw slightly bigger dots at each gaze point
                for i, (x, y) in enumerate(gaze_points_coords):
                    alpha = (i + 1) / len(gaze_points_coords)
                    color = (
                        int(255 * (1 - alpha)),  # Blue → Red
                        0,
                        int(255 * alpha)
                    )
                    cv2.circle(frame, (x, y), 10, color, -1)  # radius 10, filled

                out.write(frame)
                frame_idx += 1
            cap.release()
            out.release()
            print(f"Gaze overlay video saved as {output_file} (Size: {os.path.getsize(output_file)/1024/1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", type=str, required=True, help="Path to the videos folder")
    parser.add_argument("--gaze_path", type=str, required=True, help="Path to the gaze folder")
    parser.add_argument("--vrs_path", type=str, required=True, help="Path to the vrs folder")
    parser.add_argument("--timestamp_path", type=str, required=True, help="Path to the timestamp file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--som", type=bool, required=True, help="Whether the frames are SoM annotated")
    args = parser.parse_args()
    # make sure videos_path is a valid path
    if not os.path.exists(args.videos_path):
        raise ValueError("videos_path must be a valid path")
    # make sure gaze_path is a valid path
    if not os.path.exists(args.gaze_path):
        raise ValueError("gaze_path must be a valid path")
    # make sure vrs_path is a valid path
    if not os.path.exists(args.vrs_path):
        raise ValueError("vrs_path must be a valid path")
    # make sure timestamp_path is a valid path
    if not os.path.exists(args.timestamp_path):
        raise ValueError("timestamp_path must be a valid path")
    # make sure output_path is a valid path
    if not os.path.exists(args.output_path):
        raise ValueError("output_path must be a valid path")
    # make sure som is a boolean
    if not isinstance(args.som, bool):
        raise ValueError("som must be a boolean")
    args = parser.parse_args()
    # insert your local paths where you downloaded the videos, gaze and vrs files from the HD-EPIC dataset
    videos_base_folder = f"{args.videos_path}/"
    gaze_base_folder = f"{args.gaze_path}/"
    vrs_base_folder = f"{args.vrs_path}/"
    interaction_anticipation_timestamps_path = f"{args.timestamp_path}"
    interaction_anticipation_timestamps = pd.read_csv(interaction_anticipation_timestamps_path)
    videos_in_question = interaction_anticipation_timestamps["video_id"].unique()

    video_segments_base_folder = f"{args.videos_path}/{'SoM_last_' if args.som else ''}video_segments/"
    output_folder = f"{args.output_path}/{'SoM_' if args.som else ''}Gaze_video_segments/"
    print(f"Video segments path: {video_segments_base_folder}")
    print(f"Output folder: {output_folder}")
    print("Applying gaze trajectories to video segments...")

    for participant in range(1, 10):
        # create the participant directory if it doesn't exist
        participant_folder_out = os.path.join(output_folder, f"P0{participant}")
        if not os.path.exists(participant_folder_out):
            os.makedirs(participant_folder_out, exist_ok=True)
            print(f"Created participant directory in output: {participant_folder_out}")
        
        gaze_participant_folder = os.path.join(gaze_base_folder, f"P0{participant}/GAZE_HAND/")
        if not os.path.exists(gaze_participant_folder):
            print(f"Warning: Gaze participant folder does not exist: {gaze_participant_folder}")
            continue
        
        for folder in os.listdir(gaze_participant_folder):
            gaze_folder = os.path.join(gaze_participant_folder, f"{folder}/{folder}/eye_gaze/")
            if not os.path.exists(gaze_folder):
                print(f"Warning: Gaze folder does not exist: {gaze_folder}")
                continue
            
            video_name = folder.split("_")[1]
            if video_name not in videos_in_question:
                print(f"Video {video_name} is not in the interaction anticipation questions")
                continue
            
            video_file = os.path.join(videos_base_folder, f"P0{participant}/{video_name}.mp4")
            if not os.path.exists(video_file):
                print(f"Warning: Video file does not exist: {video_file}")
                continue
            
            gaze_file = os.path.join(gaze_folder, "general_eye_gaze.csv")
            if not os.path.exists(gaze_file):
                print(f"Warning: Gaze file does not exist: {gaze_file}")
                continue
                
            vrs_file_path = vrs_base_folder + f"P0{participant}/{video_name}_anonymized.vrs"
            if not os.path.exists(vrs_file_path):
                print(f"{vrs_file_path} does not exist, skipping...")
                continue

            gaze_cpf = mps.read_eyegaze(gaze_file)
            print(f"Applying gaze data to segments for {video_file} using {gaze_file}")
            apply_gaze_data_to_segments(video_file, gaze_cpf, output_folder, vrs_file_path, video_segments_base_folder, participant, interaction_anticipation_segments)

if __name__ == "__main__":
    main()