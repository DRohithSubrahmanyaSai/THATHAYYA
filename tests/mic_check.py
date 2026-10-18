import numpy as np
import sounddevice as sd
import librosa

from agent import config
from agent.stt import STT
from agent.tts import TTS

def test_mic_stt():
    print("Testing mic + STT (Deepgram nova-3)...")
    print("Say something in Telugu for 3 seconds...")
    # Record 3 seconds at 16 kHz, mono
    audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    # Transcribe
    stt = STT(config.STT_ENGINE)  # deepgram nova-3
    try:
        text = stt.transcribe(audio, 16000)
        print(f"Transcribed: '{text}'")
    except Exception as e:
        print(f"STT error: {e}")

def test_tts_playback():
    print("\nTesting TTS playback (Cartesia)...")
    tts = TTS(config.TTS_LIVE_ENGINE)  # cartesia
    try:
        wav, sr = tts.synthesize("నమస్కారం తాతయ్య, బాగున్నావా?")
        print(f"Playing: '{'నమస్కారం తాతయ్య, బాగున్నావా?'} ({sr} Hz, {len(wav)/sr:.2f}s)")
        sd.play(wav, sr)
        sd.wait()
        print("Done.")
    except Exception as e:
        print(f"TTS/playback error: {e}")

if __name__ == "__main__":
    test_mic_stt()
    test_tts_playback()