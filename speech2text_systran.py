import logging
from pydantic import BaseModel, ConfigDict
from faster_whisper import WhisperModel, BatchedInferencePipeline

import subprocess


from transformers import pipeline
from unidecode import unidecode
import re
import shutil
import os
import json

output_dir = "./src/temp"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
import os
import torch

def find_closest_index(array, value):
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    # Calculate the absolute difference between each element and the target value
    diff = torch.abs(array - value)
    # Find the index of the minimum difference
    closest_index = torch.argmin(diff).item()  
    return closest_index

def uppercase_first_letter(text):
    """Converts the first letter of a text to uppercase."""
    return text.capitalize()

def check_if_first_letter_is_uppercase(text):
    """Checks if the first letter of a text is uppercase."""
    for letter in text:
        if letter.isupper():
            return True
    return False

def check_if_dot_or_comma_follows(text):
    """Checks if a dot or comma follows a text."""
    last_char = text[-1]
    return text[-1] in [".", ","], last_char

def caculate_levenstein_distance(text1, text2):
    """Calculates the Levenshtein distance between two texts."""
    # Initialize the matrix with zeros
    matrix = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]
    
    # Fill the first row and column with the index
    for i in range(len(text1) + 1):
        matrix[i][0] = i
    for j in range(len(text2) + 1):
        matrix[0][j] = j
    
    # Fill the matrix with the Levenshtein distance
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            cost = 0 if text1[i - 1] == text2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # Deletion
                matrix[i][j - 1] + 1,  # Insertion
                matrix[i - 1][j - 1] + cost  # Substitution
            )
    
    return matrix[-1][-1]

def sanitize_filename(filename):
    # Convert Unicode characters to ASCII
    ascii_name = unidecode(filename)
    # Replace special characters with underscores
    safe_name = re.sub(r'[^\w\.-]', '_', ascii_name)  # Keeps letters, numbers, underscores, dots, and dashes
    return safe_name

def rename_file(file_path):
    """Renames a specific file by removing special characters and converting Unicode to ASCII."""
    directory, filename = os.path.split(file_path)  # Extract directory and filename

    new_filename = unidecode(filename)  # Convert Unicode to ASCII
    new_filename = re.sub(r'[^\w\.-]', '_', new_filename)  # Replace special characters

    new_path = os.path.join(directory, new_filename)

    if new_filename != filename:  # Rename only if the name has changed
        # copy the file to new path, not rename
        shutil.copy2(file_path, new_path)
        print(f'Renamed: "{filename}" → "{new_filename}"')
        return new_path
    else:
        print("No changes needed.")
        return file_path

# Configure logging to print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Dummy function to convert video to audio
def convert_video_to_audio(video_path, audio_path):
    logger.info(f"Converting {video_path} to {audio_path}")
    print('video_path', type(video_path))
    # audio path exsits
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    # Replace this with actual video-to-audio conversion logic
  
    # Build the ffmpeg command
    command = [
        "ffmpeg",
        "-i", f"{video_path}",  # Input file
        "-vn",                   # Skip video stream
        "-acodec", "pcm_s16le",  # Audio codec: WAV format
        f"{audio_path}"        # Output file

    ]
    
    try:
            # Run the command and wait for it to complete
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Check if the output file was created
                if os.path.exists(audio_path):
                    logger.info("Conversion successful.")
                    return True  # Indicate success
                else:
                    logger.error("Output file not created.")
                    return False  # Indicate failure
            else:
                # Log the error details
                logger.error(f"FFmpeg failed: {result.stderr}")
                return False  # Indicate failure
    except Exception as e:
            logger.exception(f"An error occurred: {e}")
            return False  # Indicate failure


class Speech2Text:
    def __init__(self,):
        # self.model = WhisperModel("large-v3-turbo", device="cuda")
        # self.model = WhisperModel("models/models--Systran--faster-whisper-large-v2/snapshots/f0fe81560cb8b68660e564f55dd99207059c092e", device="cuda")
        stt_model = "tiny.en"  # Default model, can be changed based on config
        self.model_ver = stt_model  
        self.batched_model = BatchedInferencePipeline(
            model=WhisperModel(
                stt_model,  # Model size can be 'base', 'small', 'medium', 'large'
                device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
                compute_type="float16" if torch.cuda.is_available() else "int8_float16"
            )
        )
        language = "en"  # Default language, can be changed based on config
      
        
  

    def process_audio(self, file=None):
       
        try:

            logger.info(f"Received file: {file}")
            file = rename_file(file)
            if file is None:
                logger.warning("No audio file found")
                return "No audio file found"
            extension = file.split(".")[-1]
            filename = os.path.basename(file)
            audio_path = "none.wav"
            if extension in ["mp4", "avi", "mov", "mxf"]:   
                audio_path = filename[:-3] + "wav"
                audio_path = os.path.join(output_dir, audio_path)
                convert_video_to_audio(file, audio_path)
                file = audio_path
                logger.info(f"Converted video to audio: {file}")

            logger.info("Starting transcription")

            segments, info = self.batched_model.transcribe(file, batch_size=8,  word_timestamps=True,   vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1500), condition_on_previous_text=False, language_detection_threshold=0.5,  multilingual=True, no_speech_threshold=0.5, hallucination_silence_threshold=0.5, language_detection_segments=2)

            final_msg = []
            normal_transcription = ""
            word_timestamps_list = {}
            word_timestamps_list['start'] = []
            word_timestamps_list['end'] = []
            word_timestamps_list['word'] = []
            long_word_list = []
            ori_word_list = []
            # start time list 
            pre_is_dot = True
            single_sentence = {}

            single_sentence["word"] = []
            sentent_list = []
            for segment in segments:
                    word = segment
                    start, end, text = word.start, word.end, word.text
                    for _word in segment.words:
                        starting_time = _word.start
                        if _word.end - _word.start > 0.65:
                            long_word_list.append(_word)
                        word_timestamps_list['start'].append(_word.start)
                        word_timestamps_list['end'].append(_word.end)
                        word_timestamps_list['word'].append(_word.word)
                        single_sentence["word"].append(_word.word)
                        if pre_is_dot:
                            ori_word_list.append("\n"+str(round(word_timestamps_list['start'][-1],2)))
                            ori_word_list.append(":")
                            pre_is_dot = False
                            single_sentence["start"] = round(word_timestamps_list['start'][-1],2)
                            
                            
                            
                        ori_word_list.append(_word.word)
                        if _word.word[-1] in ["."]:
                            pre_is_dot = True
                            single_sentence["end"] = round(_word.end,2)
                            sentent_list.append(single_sentence)
                            single_sentence = {}
                            single_sentence["word"] = []
                            

         
            
            _final_text = []
            for s in sentent_list:
                _final_text.append(f"{s['start']}-{s['end']}: {''.join(s['word'])}")
            joined_text = '\n'.join(_final_text)
            
            response = {"transcription": joined_text}
            
            return response
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            return "no audio found"


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


if __name__ == "__main__":
    speech2text = Speech2Text()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='input audio file')
    args = parser.parse_args()
    print(args.file)
    
    # Create the Gradio interface
    # interface = gr.Interface(
    #     fn=speech2text.process_audio,  # Function to process the audio input
    #     inputs=[
            
    #         gr.Radio(
    #             choices=["base", "medium", "large", "turbo"],
    #             label="Choose Model Size",
    #             value="large"
    #         ),  # Radio buttons for model selection
         
    #         gr.Audio(type="filepath", label="upload audio or video file here")  # Audio input from microphone
    #     ],
    #     outputs=gr.HTML(label="Transcription"),  # Output transcription only
    #     live=True,  # Enable live feedback
    #     css="footer{display:none !important}"
    #     )

    # interface.launch(share=False)

    # print(speech2text.process_audio(file='Input_State_2.wav'))
    speech2text.process_audio(file=args.file)
    
    # print(speech2text.process_audio(file=args.file))