import pyaudio, wave
import numpy as np
from openwakeword.model import Model

from vosk import Model as vModel, KaldiRecognizer
import json

import agent

from piper import PiperVoice

from scipy.io import wavfile
from scipy.signal import resample
import soundfile as sf

def robotify(input_path="output.wav", final_path="robot_output.wav",
             speed_factor=1, carrier_freq=35, bit_depth=8, sample_rate_reduction=2):

    rate, y = wavfile.read(input_path)
    y = y.astype(np.float32)

    # Cheap pitch-down via resampling (also slightly slows speech, which reads as mechanical)
    new_len = int(len(y) / speed_factor)
    y = resample(y, new_len)

    # Ring modulate
    t = np.arange(len(y)) / rate
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    y = y * carrier

    # Bitcrush
    max_val = 2 ** (bit_depth - 1)
    y = np.round(y / 32768 * max_val) / max_val * 32768
    y = np.repeat(y[::sample_rate_reduction], sample_rate_reduction)[:len(y)]

    y = y / (np.max(np.abs(y)) + 1e-9)
    wavfile.write(final_path, rate, (y * 32767).astype(np.int16))
    return final_path

RATE = 16000
CHUNK_SIZE = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 4
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1
MAX_RECORD_SECONDS = 10

chatbot = agent.Agent("""You are a servitor skull from the game Warhammer 40k.
speak in only 1-2 sentences. do not respond with sound effects."
""")

vmodel = vModel("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(vmodel, 16000)

audio = pyaudio.PyAudio()  # create ONCE, globally

input_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

def play_wav_file(filepath):
    wf = wave.open(filepath, 'rb')
    out_stream = audio.open(   # reuse the SAME audio object
        format=audio.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        out_stream.write(data)
        data = wf.readframes(chunk_size)

    out_stream.stop_stream()
    out_stream.close()
    # NOTE: do NOT call audio.terminate() here
    wf.close()


def is_silent(audio_chunk, threshold=SILENCE_THRESHOLD):
    audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
    volume = np.abs(audio_np).mean()
    return volume < threshold

def record_command(stream):
    print("Listening for command...")
    frames = []
    silent_chunks = 0
    silence_chunk_limit = int(SILENCE_DURATION * RATE / CHUNK_SIZE)
    max_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK_SIZE)

    for _ in range(max_chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
        rec.AcceptWaveform(data)  # feed Vosk live, as we go

        if is_silent(data):
            silent_chunks += 1
            if silent_chunks > silence_chunk_limit:
                break
        else:
            silent_chunks = 0

    result = json.loads(rec.FinalResult())  # just flush final result, no re-processing needed
    return result["text"]

def play_audio_bytes(audio_bytes, rate=16000, channels=1):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        output=True
    )
    stream.write(audio_bytes)
    stream.stop_stream()
    stream.close()
    audio.terminate()

piper_voice = PiperVoice.load("en_GB-alan-low.onnx")
def speak_and_play(text, output_path="output.wav"):
    if not text:
        text = "ERROR. This machine spirit did not comprehend."
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(piper_voice.config.sample_rate)
        for audio_chunk in piper_voice.synthesize(text):
            wf.writeframes(audio_chunk.audio_int16_bytes)

    processed_path = robotify(output_path, "robot_output.wav")
    play_wav_file(processed_path)

model = Model(wakeword_models=["./ServoSkull_spirit_lite.onnx"], inference_framework="onnx")

print("Listening for wake word...")

while True:
    audio_data = input_stream.read(CHUNK_SIZE, exception_on_overflow=False)

    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    prediction = model.predict(audio_np)

    for wakeword, score in prediction.items():
        if score > 0.8:
            print(f"Detected wake word: {wakeword} (score: {score.item():.2f})")
            command = record_command(input_stream)
            model.reset()
            print(command)
            print(f"user statement: {command}")
            if command and command.strip():
                toSpeak = chatbot.call(command)
                print(toSpeak)
                speak_and_play(text=toSpeak)
            print("Awaiting next command")