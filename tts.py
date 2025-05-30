import pyttsx3
import wave

def speak(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, 'output.wav')
    engine.runAndWait()
