import wave

import numpy as np
import pyaudio
import speech_recognition as sr


class Recorder:
    def __init__(self, config):
        self.audio = pyaudio.PyAudio()
        self.config = config
        self.stream = self.audio.open(
            format=config["FORMAT"],
            channels=config["CHANNELS"],
            rate=config["RATE"],
            frames_per_buffer=config["CHUNK"],
            input=True,
        )

    def record(self):
        # list to store the recorded audio frames
        frames = []

        # record audio data in chunks
        for _ in range(
            0,
            int(
                self.config["RATE"]
                / self.config["CHUNK"]
                * self.config["RECORD_SECONDS"]
            ),
        ):
            data = self.stream.read(self.config["CHUNK"])
            frames.append(data)

        self._save_to_wav(frames, self.config["OUTPUT_FILENAME"])

        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()

        # Terminate the PyAudio object
        self.audio.terminate()

    def record_until_silence(
        self,
        silence_duration: float = 3,
        silence_threshold: float = 500,
    ):
        print("Recording until silence...")

        frames = []

        # used to track silence length
        silence_count = 0
        while True:
            audio_data = self.stream.read(self.config["CHUNK"])
            frames.append(audio_data)

            # Calculate the volume (RMS)
            volume = self._rms(audio_data)

            # Check if the volume is below the silence threshold
            if volume < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0  # Reset counter if sound is detected

            # If silence lasts for more than the defined duration, stop recording
            if silence_count > int(
                self.config["RATE"] / self.config["CHUNK"] * silence_duration
            ):
                break

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def _save_to_wav(
        self,
        frames: list,
        filename: str,
    ):
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.config["CHANNELS"])
        wf.setsampwidth(self.audio.get_sample_size(self.config["FORMAT"]))
        wf.setframerate(self.config["RATE"])
        wf.writeframes(b"".join(frames))
        wf.close()

    def _rms(self, audio_data):
        """Calculates the RMS (Root Mean Square) of the audio data, useful in determining the sound volume."""

        shorts = np.frombuffer(audio_data, dtype=np.int16)
        rms_value = np.sqrt(np.mean(shorts**2))

        return rms_value


if __name__ == "__main__":
    # PyAudio parameters for recording
    config = dict(
        FORMAT=pyaudio.paInt16,
        CHANNELS=2,
        RATE=44100,
        CHUNK=1024,
        RECORD_SECONDS=5,  # todo: automatically stop recording when silent
        OUTPUT_FILENAME="output.wav",
    )

    rec = Recorder(config=config)
    # rec.record()

    r = sr.Recognizer()

    rec_output = sr.AudioFile("output.wav")
    with rec_output as source:
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)

    text = r.recognize_google(audio, language="it-IT")

    print(text)
