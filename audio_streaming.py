import subprocess

# Simulate RTSP-like audio stream using RTP (FFmpeg doesn't natively output RTSP without a full server)
command = [
    'ffmpeg',
    '-re',  # Read input in real-time
    '-i', r'D:\source_AI\scene\scene_grouping\videoplayback.wav',  # Input audio
    # "-c:a", "aac",
        "-f", "rtsp",
        "rtsp://localhost:8554/audio"
]

subprocess.run(command)
