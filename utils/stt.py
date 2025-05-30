import numpy as np
import av

class AudioProcessor:
    def __init__(self, model):
        self.model = model
        self.audio_data = []
        self.transcription = ""

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        audio = audio.flatten().astype(np.float32) / 32768.0  # Convert int16 to float32
        self.audio_data.extend(audio.tolist())

        if len(self.audio_data) > 16000 * 5:  # 5 seconds
            audio_array = np.array(self.audio_data)
            self.transcription = self.model.transcribe(audio_array, language="en")["text"]
            self.audio_data = []
        return frame
