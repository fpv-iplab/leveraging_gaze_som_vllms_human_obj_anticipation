# experiments using HD-EPIC's interaction anticipation benchmark.
# modalities: standard, som_last, som, gaze, som_last_gaze, som_gaze
# inverse exponential sampling with lambda parameter
# this script only uses LLaVA-OV

import argparse
import os
import time
import torch
import pandas as pd
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import av
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

device = "cuda"
GPU = 0
NP_SEED = 23211387

# initiate the model
def init_llava_ov():
    print("LLaVA-OV")
    global processor_llava_ov
    global model_llava_ov
    processor_llava_ov = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", use_fast=True)
    model_llava_ov = LlavaOnevisionForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-7b-ov-hf", 
        dtype=torch.float16,
        device_map={'': GPU}
    )

# sampling strategy
def sample_from_inv_exp_distrib(size, max_frame, lam):
    np.random.seed(NP_SEED)
    # exponential sampling in [0, max_frame-2]
    exp_range = np.arange(0, max_frame - 1)
    # Compute exponential weights that increase as we approach max_frame-2
    distances = (max_frame - 2) - exp_range  # distance from the end
    weights = np.exp(-lam * distances)
    weights /= weights.sum()  # normalize

    exp_samples = np.random.choice(exp_range, size=size-1, replace=False, p=weights)

    # always include max_frame - 1
    final_samples = np.concatenate([exp_samples, [max_frame - 1]])
    assert len(final_samples) == size
    return np.sort(final_samples)

# sampling module
def load_video(video_path):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = sample_from_inv_exp_distrib(sampled_frames, total_frames, lambda_param)
    indices.sort()
    print(indices)

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    video = np.stack([x.to_ndarray(format="rgb24") for x in frames])

    return video

# single query to the model
def video_query_llava_ov(prompt: str, video_path: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    video = load_video(video_path)
    prompt_text = processor_llava_ov.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor_llava_ov(videos=list(video), text=prompt_text, return_tensors="pt").to(model_llava_ov.device, torch.float16) # pass the prompt and video to LLaVA-OV

    out = model_llava_ov.generate(**inputs, max_new_tokens=150) # take the model's output and decode it
    response = processor_llava_ov.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return response[0].split("assistant\n")[1]

def video_queries_to_model(questions: pd.DataFrame, base_folder: str, mode: str):
    responses = []
    for index, row in questions.iterrows():
        participant = row['inputs']['video 1']['id'].split("-")[0]
        video_path = f"{base_folder}{participant}/{row['inputs']['video 1']['id']}_{index.split('_')[-1]}.mp4"

        if not os.path.exists(video_path):  # skip if file is missing
            print(f"{video_path} does not exist... Skipping...")
            continue

        prompt = ""
        if "last" in mode:
            prompt += "Focus on the last frame to make your prediction and use the rest of the video to infer the context.\n\n"
        
        prompt += row['question'] + ''.join(f"\n{choice}" for choice in row['choices'])
        
        if "gaze" in mode:
            prompt += "\n\nFollow the user's gaze trajectory closely: the red circles indicate where the user has most recently looked, and the connected path shows the sequence of gaze points across the most recent frames. The objects that have just been fixated are very likely to include the one the user will interact with next. Use this visual cue to make your prediction."

        correct_choice = row['correct_idx']
        response = video_query_llava_ov(prompt, video_path)
        print("---------------------------------------------------------------")
        print("Prompt: ", prompt)
        print("Question number: ", index)
        print("Video path: ", video_path)
        print(f"Model's answer: {response} - correct answer: {row['choices'][correct_choice]}")
        print("---------------------------------------------------------------")

        is_correct = str(response == row['choices'][correct_choice])
        video_name = f"{row['inputs']['video 1']['id']}_{index}"
        responses.append((response, row['choices'][correct_choice], row['question'], is_correct, video_name))
    return responses

def main():
    parser = argparse.ArgumentParser(description="OBJ IA VQA new script")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (0 to 3)")
    parser.add_argument("--mode", type=str, default="standard",
                        help="Mode, e.g. 'standard', 'gaze', 'som', or 'som_gaze'")
    parser.add_argument("--lamb", type=float, default=0.0, help="Inv exp lambda")
    parser.add_argument("--sample", type=int, default=10, help="Number of sampled frames")
    parser.add_argument("--video_clips_path", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to the annotations folder")
    args = parser.parse_args()
    # make sure video_clips_path is a valid path
    if not os.path.exists(args.video_clips_path):
        raise ValueError("video_clips_path must be a valid path")
    # make sure annotations_path is a valid path
    if not os.path.exists(args.annotations_path):
        raise ValueError("annotations_path must be a valid path")
    
    global sampled_frames
    sampled_frames = args.sample

    global lambda_param 
    lambda_param = args.lamb

    global GPU
    GPU = args.gpu
    if GPU < 0 or GPU > 3:
        raise Exception("GPU not valid")
    torch.cuda.set_device(GPU)

    init_llava_ov()

    mode = args.mode
    base_folder = f"{args.video_clips_path}/"

    interaction_anticipation_questions_path = f"{args.annotations_path}/gaze_interaction_anticipation.json"
    interaction_anticipations_questions = pd.read_json(interaction_anticipation_questions_path).T

    responses = video_queries_to_model(interaction_anticipations_questions, base_folder, mode)

    print(responses)
    correct_responses = [resp[0] == resp[1] for resp in responses]
    print(f"Correct responses: {sum(correct_responses)} / {len(responses)}")

    responses_df = pd.DataFrame(responses, columns=["model_response", "correct_response", "question", "is_correct", "video_name"])
    responses_df.to_csv(
        f"./results_LLaVA-OV_{mode}_{lambda_param}_{sampled_frames}_{int(time.time())}.csv",
        index=False
    )

    # example output file name
    # results_LLaVA-OV_gaze_original_0.01_20_1760169710.csv

if __name__ == "__main__":
    main()