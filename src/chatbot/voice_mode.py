from pathlib import Path
from openai import OpenAI
import pyaudio
import wave
import audioop
from .utils.utils import create_folder
from .utils.paths import VOICES_DIR
from .config import get_api_key
from playsound import playsound
import tempfile
import os

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAVE_OUTPUT_FILENAME = create_folder(VOICES_DIR) / "output.wav"
SILENCE_THRESHOLD = 500  # Adjust this value based on your environment (RMS amplitude)
SILENCE_DURATION = 2.0
TTS_OUTPUT_FILE = create_folder(VOICES_DIR) / "speech.mp3"


class VoiceMode:
    def __init__(self):
        self.client = OpenAI(api_key=get_api_key("OPENAI"))
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        self.frames = []
        self.silent_chunks = 0
        self.chunks_per_second = RATE / CHUNK
        self.silent_chunks_threshold = int(self.chunks_per_second * SILENCE_DURATION)

    def start_recording(self):
        print(f"* Recording... Will stop after {SILENCE_DURATION} seconds of silence.")

        while True:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            self.frames.append(data)

            rms = audioop.rms(data, 2)

            if rms < SILENCE_THRESHOLD:
                self.silent_chunks += 1
            else:
                self.silent_chunks = 0

            if self.silent_chunks >= self.silent_chunks_threshold:
                print(
                    f"* Silence detected for {SILENCE_DURATION} second. Stopping recording."
                )
                break

        self._stop_recording()
        self._save_recording()

        return WAVE_OUTPUT_FILENAME

    def _stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def _save_recording(self):
        wf = wave.open(str(WAVE_OUTPUT_FILENAME), "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(self.frames))
        wf.close()

    def transcribe(self, audio_path):
        audio_file = open(audio_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )

        return transcription.text

    def text_to_speech(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(response.content)

        try:
            playsound(temp_filename)
        finally:
            os.unlink(temp_filename)
