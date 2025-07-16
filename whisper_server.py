from fastapi import FastAPI, WebSocket
from speech2text_systran import Speech2Text
import asyncio
import time
import wave 
import os 
from config import SAMPLE_RATE, CHANNELS, AUDIO_FORMAT, BUFFER_SIZE, save_dir
app = FastAPI()
stt = Speech2Text()
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        data = await websocket.receive_bytes()
        # asyncio.sleep(3)
        # write data to file or process it .wav
        # WAV parameters
        num_channels = 1          # Mono
        sample_width = 2          # 2 bytes per sample (16-bit audio)
        frame_rate = SAMPLE_RATE
        # Save to WAV file
        current_time = time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        file_path = f"{save_dir}/{current_time}.{AUDIO_FORMAT}"
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(frame_rate)
            wf.writeframes(data)
        output = stt.process_audio(file_path)
        print(f"Received data of length: {len(data)} bytes")
        await websocket.send_text(f"{current_time} --- {output}")

# To run the server, use the command: uvicorn whisper_server:app --reload