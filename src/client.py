import os
import sys
from datetime import datetime

import pyaudio
import requests
import speech_recognition as sr
from dotenv import load_dotenv

from speech_to_text import Recorder

load_dotenv()

if __name__ == "__main__":
    # settings
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

    output_filename = f"./data/{datetime.now().strftime('%d%m%Y_%H%M%S')}.wav"

    print("Recording...")

    # --- record a spoken sentence ---
    rec = Recorder(
        config={
            "FORMAT": pyaudio.paInt16,
            "CHANNELS": 2,
            "RATE": 44100,
            "CHUNK": 1024,
            "RECORD_SECONDS": 5,
            "OUTPUT_FILENAME": output_filename,
        }
    )
    rec.record()

    # --- perform speech recognition on the recorded sentence ---
    r = sr.Recognizer()

    # file_dir = "./data/harvard.wav"
    file_dir = output_filename
    sr_r = requests.post(f"{api_url}/sr", files={"file": open(file_dir, "rb")})

    # quit if request was not succesful
    if not sr_r.ok:
        raise Exception(f"Request status: {sr_r.status_code}")

    sr_text = sr_r.json()["text"]

    print(f"Frase riconosciuta: {sr_text}")

    # --- send the speech-recognized text to RAG ---
    r = requests.get(f"{api_url}/query", {"query": sr_r.text})

    # quit if request was not succesful
    if not r.ok:
        raise Exception(f"Request status: {sr_r.status_code}")

    print(f"\nRisposta: {r.text}")

    # --- send query output to Pepper for TTS ---
    r = requests.get(f"{api_url}/tts", {"text": r.text})

    # quit if request was not succesful
    if not r.ok:
        raise Exception(f"Request status: {sr_r.status_code}")
