import subprocess
import time
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

class GameRecorder:
    def __init__(self, output_prefix="episode_recording", resolution="1920x1080", fps=30):
        self.output_prefix = output_prefix
        self.resolution = resolution
        self.fps = fps
        self.recording_process = None
        self.record_using_ffmpeg = False  # Toggle for FFmpeg or OpenCV

    def start_recording_ffmpeg(self, episode):
        """Start recording with FFmpeg."""
        output_file = f"{self.output_prefix}_ep{episode}.mp4"
        command = [
            "ffmpeg",
            "-video_size", self.resolution,
            "-framerate", str(self.fps),
            "-f", "gdigrab",  # Replace with "gdigrab" for Windows
            "-i", ":0.0",  # Adjust for Windows or Mac
            output_file
        ]
        self.recording_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_recording_ffmpeg(self):
        """Stop the FFmpeg recording process."""
        if self.recording_process:
            self.recording_process.terminate()
            self.recording_process = None

    def record_game_opencv(self, episode, duration):
        """Record the game using OpenCV."""
        game_window = gw.getWindowsWithTitle('Flappy Bird')[0]  # Replace with your game window title
        x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height
        output_file = f"{self.output_prefix}_ep{episode}.avi"

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, (width, height))

        # Record for the duration
        start_time = time.time()
        while time.time() - start_time < duration:
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)

        out.release()