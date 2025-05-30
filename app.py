import streamlit as st
from utils.stt import AudioProcessor
from utils.llm import generate_response
from utils.tts import speak
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper

st.set_page_config(page_title="AI Voice Agent - Health Tourism", layout="centered")
st.title("🎤 AI Voice Agent for Health Tourism")
st.markdown("Ask about medical travel, clinics, procedures, visas, and more.")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

audio_processor = AudioProcessor(model=model)

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    in_audio_enabled=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_receiver_size=1024,
    on_audio_frame=audio_processor.recv
)

if audio_processor.transcription:
    user_input = audio_processor.transcription
    st.write("🗣️ You said:", user_input)

    with st.spinner("🧠 Thinking..."):
        ai_response = generate_response(user_input)
        st.write("🤖 Assistant:", ai_response)

        with st.spinner("🔊 Speaking..."):
            speak(ai_response)
            st.audio("output.wav", format="audio/wav")
