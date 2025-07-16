import asyncio
import websockets
import time 
import subprocess
import os
import threading
import time 
from queue import Queue
import torchaudio

from config import SAMPLE_RATE, CHANNELS, BUFFER_SIZE, input_source
import numpy as np
import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')


cmd = [
    'ffmpeg',

    '-i', input_source,
    "-f", "s16le",            # raw PCM 16-bit little endian
    "-acodec", "pcm_s16le",
    "-ar", f"{SAMPLE_RATE}",
    "-ac", f"{CHANNELS}",               # mono (change if stereo)
    "-"
]
resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=16000)

def bytes2seconds(byte_length, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    return byte_length / (sample_rate * channels * 2)  # 2 bytes per sample for PCM 16-bit
def seconds2bytes(seconds, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    return int(seconds * sample_rate * channels * 2)  # 2 bytes per sample for PCM 16-bit

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,  stderr=subprocess.STDOUT)
cache_list = []

(get_speech_timestamps, _, read_audio, _, _) = utils
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Use GPU if available
def read_data_from_process(proc, chunk_size=BUFFER_SIZE, queue=None):

    counter = 0
    redundant_data = b''  # To store redundant data if needed
    while True:
        try:
            data = proc.stdout.read(chunk_size)
            # print(len(data))
            # print('counter:', counter)
            counter += 1
            
            if queue is not None:
                queue.put(data)
        except Exception as e:
            print(f"Error reading from process: {e}")
            break
        if not data:
            queue.put(None)  # Indicate end of stream
            print("No more data from process.")
            break

async def send_data_to_websocket(websocket, queue=None):
    redundant_data = b''  # To store redundant data if needed   
    while True:
        try:
            data =  queue.get() if queue and queue else None
            if data:
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_np)
                try:
                    audio_tensor = resampler(audio_tensor)
                except Exception as e:
                    print(f"Error during resampling: {e}")

                audio_tensor = audio_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Move to GPU if available
                speech_timestamps = get_speech_timestamps(
                    audio_tensor,
                    model,
                    sampling_rate=16000,  # Ensure this matches your audio sample rate
                    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
                    )
                last_speech_end = speech_timestamps[-1]['end'] if speech_timestamps else 0
                # print(f"Last speech end: {last_speech_end} seconds, Data length: {len(data)} bytes")
                cut_data = redundant_data + data[:seconds2bytes(last_speech_end)] if last_speech_end > 0 else data
                redundant_data = b''  # Reset redundant data after cutting
                # print(f"Cut data length: {len(cut_data)} bytes, Last speech end: {last_speech_end} seconds")
                if last_speech_end > 0:
                    # print("last_speech_end",last_speech_end)
                    redundant_data = data[seconds2bytes(last_speech_end):]
                    # print(f"Redundant data length: {len(redundant_data)} bytes")
                # print(f"Sending data of length: {len(cut_data)} bytes")
                await websocket.send(cut_data)
                # wait for a response if needed
                response = await websocket.recv()
                current_time = time.strftime("%Y%m%d_%H%M%S")
                response = f"{current_time} --- {response.split(":")[-1].strip().replace('}"', '')}"  # Extract the text after the first colon and remove quotes
                print(response)
            else:
                await asyncio.sleep(0.1)
            
            if data is None:
                break
        except websockets.ConnectionClosed:
            print("WebSocket connection closed.")
            break
        

async def websocket_client():
    uri = "ws://localhost:8000/ws"  
    async with websockets.connect(uri) as websocket:

        cache_list = Queue()

        t1 = threading.Thread(target=read_data_from_process, args=(proc, BUFFER_SIZE, cache_list))
        t1.start()

        await send_data_to_websocket(websocket, cache_list)

        t1.join()



        

        # # Nhận phản hồi
        # response = await websocket.recv()
        # print(f"Received: {response}")

asyncio.run(websocket_client())
