# ------------------------------------------------- #
# This script extracts all the individual clips of the 
# applies Set-of-Mark prompting (without the alphanumerical 
# marks) to the final frame of the clips of 

# If you're replicating the experiments in the paper,
# just replace all the paths with your local ones and
# carefully follow all the instructions in the README.md file.
# ------------------------------------------------- #

import os
import argparse

# Set CUDA_VISIBLE_DEVICES before importing torch or any model code
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
args, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
from PIL import Image
import torch
from pathlib import Path
import cv2

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label

model_semsam = None 
model_sam = None 
model_seem = None

def build_models(device):
    global model_semsam, model_sam, model_seem
    '''
    build args
    '''
    semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
    seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

    semsam_ckpt = "./swinl_only_sam_many2many.pth"
    sam_ckpt = "./sam_vit_h_4b8939.pth"
    seem_ckpt = "./seem_focall_v1.pt"

    opt_semsam = load_opt_from_config_file(semsam_cfg)
    opt_seem = load_opt_from_config_file(seem_cfg)
    opt_seem = init_distributed_seem(opt_seem)

    '''
    build model
    '''
    model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().to(device)
    model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().to(device)
    model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().to(device)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    _image = image['background'].convert('RGB')
    _mask = image['layers'][0].convert('L') if image['layers'] else None

    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(_mask))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(model, _image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, _image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        return output
    
def handle_source_folder(parameters):
    source_path = parameters["source_path"]
    if source_path[len(source_path) - 1] != '/':
        source_path = source_path + "/"
    parameters["source_path"] = source_path
    source_folder = Path(source_path)
    for item in source_folder.iterdir():
        # check if the item already exists in dest_path before processing
        dest_path = parameters["dest_path"]
        if os.path.exists(dest_path + item.stem + ".jpg"):
            print(f"File {item.stem}.jpg already exists in {dest_path}, skipping...")
            continue
        
        # check if the item is a file and has a valid image extension
        if item.is_file() and item.suffix.lower() in [".jpeg", ".jpg", ".png", ".webp"]:
            mark_image({'background': item, 'layers': []}, parameters, item.stem)

def mark_image(image, parameters, filename):
    # Ensure 'background' is a PIL Image
    if isinstance(image['background'], (str, Path)):
        image['background'] = Image.open(image['background'])
    image_out = inference(image, parameters["granularity"], "Automatic", parameters["alpha"], parameters["label_mode"], parameters["ann_mode"])
    os.makedirs(os.path.dirname(parameters["dest_path"]), exist_ok=True)

    # Save the output as a jpg
    output_filename = os.path.join(parameters["dest_path"], f"{filename}.jpg")
    # If image_out is RGB, convert to BGR for OpenCV
    if image_out.shape[-1] == 3:
        image_out_bgr = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)
    else:
        image_out_bgr = image_out
    cv2.imwrite(output_filename, image_out_bgr)
    print(f"[SOM] File {filename}.jpg created at: {parameters['dest_path']}")

def handle_source_file(parameters):
    source_file = Path(parameters["source_path"])
    parameters["source_file"] = str(source_file.parent) + "/"
    mark_image({'background': source_file, 'layers': []}, parameters, source_file.stem)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", type=str, required=True, help="Path to the videos folder")
    parser.add_argument("--gaze", type=bool, required=True, help="Whether the frames are gaze annotated")
    args = parser.parse_args()
    # No need to parse args again, already done at the top
    device = torch.device("cuda:0")  # cuda:0 after setting CUDA_VISIBLE_DEVICES
    print(f"Using GPU: {args.gpu} (CUDA_VISIBLE_DEVICES remaps this to cuda:0)")

    participants = [1,2,3,4,5,6,7,8,9] # manually change this if you want to apply SoM to only certain participants
    print(f"Participants: {participants}")

    # Initialize the models
    build_models(device)
    print("Models initialized successfully.")

    # make sure gaze is a boolean
    if not isinstance(args.gaze, bool):
        raise ValueError("gaze must be a boolean")

    gaze = args.gaze
    # make sure videos_path is a valid path
    if not os.path.exists(args.videos_path):
        raise ValueError("videos_path must be a valid path")
    source_path = f"{args.videos_path}/{'Gaze_' if gaze == 'y' else ''}video_segments/"
    dest_path = f"{args.videos_path}/SoM_last_{'Gaze_' if gaze == 'y' else ''}video_segments/"

    os.makedirs(dest_path, exist_ok=True)

    granularity = 1.91
    alpha = 0.05
    label_mode = "Number" # this is set but not used in our implementation of SoM, in fact ann_mode is only "Mask"
    ann_mode = ["Mask"]
    parameters = dict(
        source_path=source_path,
        dest_path=dest_path,
        granularity=granularity,
        alpha=alpha,
        label_mode=label_mode,
        ann_mode=ann_mode,
    )

    for participant in participants:
        participant_folder_out = os.path.join(dest_path, f"P0{participant}/")
        os.makedirs(participant_folder_out, exist_ok=True)

        participant_folder_in = os.path.join(source_path, f"P0{participant}")
        video_folder_out = ""

        # for each video file in the input folder
        for video in os.listdir(participant_folder_in):
            if not video.endswith(".mp4"):
                continue
            video_name = video.split(".")[0]
            video_folder_out = os.path.join(participant_folder_out + "tmp/", video_name)
            video_out = os.path.join(participant_folder_out, f"{video_name}.mp4")
            if os.path.exists(video_out):
                print(f"Video {video_out} already exists, skipping...")
                continue
            
            if not os.path.exists(video_folder_out):
                os.makedirs(video_folder_out, exist_ok=True)
                print(f"Created video directory in output: {video_folder_out}")
            
            # extract frames from the video
            video_path = os.path.join(participant_folder_in, video)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_i in range(frame_count):
                ret, frame = cap.read()
                frame_path = os.path.join(video_folder_out, f"{frame_i}.jpg")
                cv2.imwrite(frame_path, frame)
            cap.release()
            print(f"Extracted {frame_count} frames from video {video_name}")

            parameters["source_path"] = video_folder_out + f"/{frame_count-1}.jpg"
            parameters["dest_path"] = video_folder_out
            handle_source_file(parameters)

            # create a video from the SoM frames using OpenCV
            frame_files = sorted(
                [f for f in os.listdir(video_folder_out) if f.endswith(".jpg")],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            if not frame_files:
                print(f"No frames found in {video_folder_out}, skipping video creation.")
            else:
                first_frame = cv2.imread(os.path.join(video_folder_out, frame_files[0]))
                height, width = first_frame.shape[:2]
                out_video = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                for frame_file in frame_files:
                    frame = cv2.imread(os.path.join(video_folder_out, frame_file))
                    if frame is not None:
                        out_video.write(frame)
                out_video.release()
                print(f"Created SoM video: {video_out}")

            # remove the temporary folder
            os.system(f"rm -r {video_folder_out}")
            print(f"Removed temporary folder: {video_folder_out}")

if __name__ == "__main__":
    main()