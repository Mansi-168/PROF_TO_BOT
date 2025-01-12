import streamlit as st
import sounddevice as sd
import wave
import threading
import os
import datetime
import requests
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioRecorder:
    """Handles audio recording functionality using sounddevice."""
    
    def __init__(self):
        self.SAMPLE_RATE = 44100  # Sample rate for audio recording
        self.CHANNELS = 1  # Mono audio
        self.DURATION = 10  # Default duration of recording in seconds
        self.recording = False
        self.frames = []
        
    def start_recording(self, duration=10):
        """Start a new recording session."""
        try:
            self.DURATION = duration
            self.recording = True
            self.frames = []
            threading.Thread(target=self._record, daemon=True).start()
            logger.info("Recording started successfully")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            raise

    def stop_recording(self) -> None:
        """Stop the current recording session."""
        self.recording = False
        logger.info("Recording stopped")

    def _record(self) -> None:
        """Internal method to handle the recording process using sounddevice."""
        try:
            logger.info(f"Recording for {self.DURATION} seconds...")
            recording = sd.rec(int(self.SAMPLE_RATE * self.DURATION), 
                               samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype='int16')
            sd.wait()  # Wait until recording is done
            self.frames = recording
            logger.info("Recording completed")
        except Exception as e:
            logger.error(f"Error during recording: {e}")

    def save_recording(self, filename: str) -> str:
        """Save the recorded audio to a WAV file."""
        try:
            os.makedirs("recordings", exist_ok=True)
            filepath = os.path.join("recordings", filename)

            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(self.frames.tobytes())

            logger.info(f"Recording saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            raise

class GroqAPI:
    """Handles interactions with the Groq API."""
    
    def __init__(self, api_key: str):
        self.api_key = "your_api_key_here"
        self.base_url = "https://api.groq.com/openai"
        
    def summarize_text(self, text: str) -> str:
        """Summarize text using Groq API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mixtral-8x7b-32768",  # Using Mixtral model
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": f"""Please provide a concise summary of the following text: -MAIN TOPICS AND KEY POINTS
                     IMPORTANT CONCEPTS AND DEFINITIONS
                     -ANY SIGNIFICANT EXAMPLES MENTIONED
                     -KEY TAKEAWAYS
                     HERE IS THE TEXT{text}"""}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise

def transcribe_audio(audio_file: str) -> Optional[str]:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return None

def save_summary(summary: str, filename: str = "class_notes.txt") -> None:
    """Save summary to a text file with timestamp."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, "a") as file:
            file.write(f"\n--- Summary from {timestamp} ---\n")
            file.write(summary + "\n")
        logger.info(f"Summary saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
        raise

def main():
    st.set_page_config(page_title="Audio Recorder & Summarizer", layout="wide")
    st.title("📝 PROF TO BOT")
    
    # Initialize session state
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = False
    if 'recorded_files' not in st.session_state:
        st.session_state.recorded_files = []
        
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        groq_api_key = "your_api_key_here"
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""1. Enter your Groq API key
        2. Click 'Start Recording' to begin
        3. Speak clearly into your microphone
        4. Click 'Stop Recording' when finished
        5. Wait for transcription and summary""")
        
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Start Recording", disabled=st.session_state.recording_status):
            try:
                st.session_state.recorder.start_recording()
                st.session_state.recording_status = True
            except Exception as e:
                st.error(f"Failed to start recording: {e}")
                
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording_status):
            try:
                st.session_state.recorder.stop_recording()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
                filepath = st.session_state.recorder.save_recording(filename)
                st.session_state.recorded_files.append(filepath)
                st.session_state.recording_status = False
                st.success("Recording saved successfully!")
            except Exception as e:
                st.error(f"Failed to stop recording: {e}")
                
    # Display recording status
    if st.session_state.recording_status:
        st.warning("🔴 Recording in progress...")
        
    # Process latest recording
    if st.session_state.recorded_files:
        st.markdown("### Latest Recording")
        last_file = st.session_state.recorded_files[-1]
        
        # Audio player
        st.audio(last_file)
        
        if st.button("🤖 Transcribe and Summarize"):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar.")
                return
                
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(last_file)
                
            if transcription:
                st.markdown("### Transcription")
                st.write(transcription)
                
                with st.spinner("Generating summary..."):
                    try:
                        groq = GroqAPI(groq_api_key)
                        summary = groq.summarize_text(transcription)
                        
                        st.markdown("### Summary")
                        st.write(summary)
                        
                        save_summary(summary)
                        st.success("Summary saved to class_notes.txt")
                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}")
            else:
                st.error("Transcription failed. Please try recording again.")

if __name__ == "__main__":
    main()
