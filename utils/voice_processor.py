# ============================================================================
# FILE: utils/voice_processor.py
# ============================================================================
"""Voice transcription using Whisper"""

import os
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from config import Config


class VoiceTranscriber:
    """Real-time voice transcription using Whisper"""
    
    def __init__(self, model_size: str = None):
        model_size = model_size or Config.WHISPER_MODEL_SIZE
        print(f"Loading Whisper model ({model_size})...")
        
        self.model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8"
        )
        self.sample_rate = Config.SAMPLE_RATE
        
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        return transcription.strip()
    
    def record_and_transcribe(self, duration: int = None) -> str:
        """Record audio and transcribe"""
        duration = duration or Config.RECORDING_DURATION
        print(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        
        # Save temporarily
        temp_file = "temp_recording.wav"
        write(temp_file, self.sample_rate, recording)
        
        # Transcribe
        transcription = self.transcribe_audio(temp_file)
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return transcription
