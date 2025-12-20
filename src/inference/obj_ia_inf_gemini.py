# ----------------
# experiments using HD-EPIC's interaction anticipation benchmark.
# modalities: standard, som_last, som, gaze, som_last_gaze, som_gaze
# this script only uses Gemini

from google import genai
from google.genai import types
import argparse
import os
import time
import pandas as pd

genai_client = genai.Client(api_key="your_api_key")
model_name = "gemini-2.0-flash"

# single query to the model
def video_query_gemini(prompt: str, video_path: str, max_retries=5):
    video_bytes = open(video_path, 'rb').read()
    
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=f'models/{model_name}',
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                            video_metadata=types.VideoMetadata(fps=custom_fps)
                        ),
                        types.Part(text=prompt)
                    ]
                )
            )
            text = response.text.replace('\r', '').replace('\n', '')
            return text
        except:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # backoff esponenziale: 5s, 10s, 20s, 40s, 80s
                print(f"Rate limit hit (429). Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                raise

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

        prompt += "\n\nYour reply must contain one of the provided answers."

        correct_choice = row['correct_idx']
        response = video_query_gemini(prompt, video_path)
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
    parser.add_argument("--mode", type=str, default="standard",
                        help="Mode, e.g. 'standard', 'gaze', 'som', or 'som_gaze'")
    parser.add_argument("--fps", type=int, default=1, help="Fps for sampling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to the annotations folder")
    args = parser.parse_args()
    # make sure data_path is a valid path
    if not os.path.exists(args.data_path):
        raise ValueError("data_path must be a valid path")
    # make sure annotations_path is a valid path
    if not os.path.exists(args.annotations_path):
        raise ValueError("annotations_path must be a valid path")
    
    global custom_fps
    custom_fps = args.fps

    mode = args.mode
    # insert your local path here if needed
    base_folder = f"{args.data_path}/"
    # Build the appropriate folder name
    if mode == "som":
        base_folder += "SoM_"
    elif mode == "som_last":
        base_folder += "SoM_last_"
    elif mode == "gaze":
        base_folder += "Gaze_"
    elif mode == "som_last_gaze":
        base_folder += "SoM_last_Gaze_"
    elif mode == "som_gaze":
        base_folder += "SoM_Gaze_"

    base_folder += f"video_segments/"

    # insert your local path where you downloaded the HD-EPIC annotations
    interaction_anticipation_questions_path = f"{args.annotations_path}/gaze_interaction_anticipation.json"
    interaction_anticipations_questions = pd.read_json(interaction_anticipation_questions_path).T

    responses = video_queries_to_model(interaction_anticipations_questions, base_folder, mode)

    print(responses)
    correct_responses = [resp[0] == resp[1] for resp in responses]
    print(f"Correct responses: {sum(correct_responses)} / {len(responses)}")

    responses_df = pd.DataFrame(responses, columns=["model_response", "correct_response", "question", "is_correct", "video_name"])
    responses_df.to_csv(
        f"./results_{model_name}_{mode}_{custom_fps}_{int(time.time())}.csv",
        index=False
    )

    # example output file name
    # results_gemini-2.0-flash_gaze_original_20_1760169710.csv

if __name__ == "__main__":
    main()