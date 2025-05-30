import streamlit as st
from streamlit_webrtc import webrtc_streamer
import whisper
import av
import numpy as np
import traceback

st.set_page_config(page_title="Whisper + WebRTC Debug Demo")

st.title("🎤 Whisper Speech-to-Text with Debugging")

@st.cache_resource
def load_whisper_model():
    st.write("Loading Whisper model...")
    model = whisper.load_model("base")
    st.write("Whisper model loaded.")
    return model

model = load_whisper_model()

class WhisperAudioProcessor:
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # Convert audio frame to numpy array (float32 mono)
            audio = frame.to_ndarray(format="flt32", layout="mono").flatten()
            st.session_state["debug_audio_shape"] = audio.shape
            st.session_state["debug_audio_dtype"] = str(audio.dtype)
            st.session_state["debug_audio_min"] = np.min(audio)
            st.session_state["debug_audio_max"] = np.max(audio)

            # Resample from 48kHz to 16kHz (required by Whisper)
            audio_16k = whisper.audio.resample_audio(audio, orig_sr=48000, target_sr=16000)
            st.session_state["debug_audio16k_shape"] = audio_16k.shape

            # Run Whisper inference
            result = model.transcribe(audio_16k, fp16=False)
            st.session_state["transcription"] = result["text"]
            st.session_state["debug_last_inference"] = "Success"

        except Exception as e:
            st.session_state["debug_last_inference"] = f"Error: {e}"
            st.session_state["debug_traceback"] = traceback.format_exc()

        return frame

# Initialize debug keys
for key in ["transcription", "debug_audio_shape", "debug_audio_dtype", "debug_audio_min", "debug_audio_max", "debug_audio16k_shape", "debug_last_inference", "debug_traceback"]:
    if key not in st.session_state:
        st.session_state[key] = "N/A"

webrtc_ctx = webrtc_streamer(
    key="whisper_debug",
    mode="recvonly",
    audio_processor_factory=WhisperAudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

st.markdown("### Transcription")
st.write(st.session_state["transcription"])

st.markdown("### Debug info")
st.write(f"Audio shape: {st.session_state['debug_audio_shape']}")
st.write(f"Audio dtype: {st.session_state['debug_audio_dtype']}")
st.write(f"Audio min value: {st.session_state['debug_audio_min']}")
st.write(f"Audio max value: {st.session_state['debug_audio_max']}")
st.write(f"Resampled audio shape (16kHz): {st.session_state['debug_audio16k_shape']}")
st.write(f"Last inference status: {st.session_state['debug_last_inference']}")

if st.session_state["debug_last_inference"].startswith("Error"):
    st.markdown("#### Traceback")
    st.code(st.session_state["debug_traceback"])
