#  Aman --added prompt--
import asyncio
import logging
import json
import wave
import os
import time
import base64
from dotenv import load_dotenv
import websockets
from websockets.exceptions import ConnectionClosed
import openai
import numpy as np
import pydub
from scipy import signal

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio
PORT = 8000

# Audio processing settings
class AudioProcessor:
    def __init__(self):
        self.buffer = bytearray()
        self.is_receiving = False
        self.temp_file = "temp_audio.wav"
        self.response_file = "response_audio.wav"
        
    def start_recording(self):
        """Initialize a new recording session"""
        logger.info("Starting new recording")
        self.buffer = bytearray()
        self.is_receiving = True
        
    def stop_recording(self):
        """Finalize recording and save to WAV file"""
        logger.info("Stopping recording, received %d bytes", len(self.buffer))
        self.is_receiving = False
        
        if len(self.buffer) == 0:
            logger.warning("Empty recording, skipping processing")
            return False
            
        # Save buffer to WAV file
        try:
            with wave.open(self.temp_file, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(self.buffer)
            return True
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}")
            return False
            
    def add_audio_data(self, data):
        """Add audio data to the buffer"""
        if self.is_receiving:
            self.buffer.extend(data)
            
    def process_audio(self):
        """Process the audio for better quality (noise reduction, normalization)"""
        try:
            # Convert buffer to numpy array for processing
            audio_data = np.frombuffer(self.buffer, dtype=np.int16)
            
            # Normalize audio (adjust volume)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
            
            # Simple noise reduction (high-pass filter to remove low-frequency noise)
            b, a = signal.butter(4, 100/(SAMPLE_RATE/2), 'highpass')
            audio_data = signal.filtfilt(b, a, audio_data).astype(np.int16)
            
            # Update buffer with processed data
            self.buffer = bytearray(audio_data.tobytes())
            
            logger.info("Audio processed successfully")
            return True
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False

class ChatGPTHandler:
    def __init__(self):
        self.system_prompt = os.getenv("CHATGPT_SYSTEM_PROMPT", 
            "You are a helpful voice assistant. Keep your responses concise and natural for speech.")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        self.max_context_messages = 10  # 5 user-assistant pairs
    
    async def transcribe_audio(self, filename):
        """Transcribe audio file using OpenAI's Whisper API"""
        try:
            with open(filename, "rb") as file:
                logger.info("Transcribing audio with Whisper API")
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                logger.info(f"Audio file size: {file_size} bytes")
                
                if file_size == 0:
                    logger.error("Empty audio file")
                    return None
                    
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file
                )
                transcription = response.text
                logger.info(f"Transcription: {transcription}")
                return transcription
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    async def generate_response(self, transcription, custom_prompt=None):
        """Generate response using OpenAI's ChatGPT API with memory"""
        try:
            logger.info("Generating ChatGPT response")

            # Replace system prompt if provided
            if custom_prompt:
                self.message_history[0] = {"role": "system", "content": custom_prompt}

            # Append user message
            self.message_history.append({"role": "user", "content": transcription})

            # Trim to keep only the last N messages (excluding system prompt)
            if len(self.message_history) > self.max_context_messages + 1:  # +1 for system prompt
                self.message_history = [self.message_history[0]] + self.message_history[-self.max_context_messages:]

            # Get response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.message_history,
                max_tokens=150,
                temperature=0.7
            )

            reply = response.choices[0].message.content

            # Append assistant response
            self.message_history.append({"role": "assistant", "content": reply})

            # Again trim after appending assistant reply
            if len(self.message_history) > self.max_context_messages + 1:
                self.message_history = [self.message_history[0]] + self.message_history[-self.max_context_messages:]

            logger.info(f"Generated response: {reply}")
            return reply

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process your request."
            
    async def text_to_speech(self, text, output_file):
        """Convert text to speech using OpenAI's TTS API"""
        try:
            logger.info("Converting text to speech")
            temp_mp3 = "temp_speech.mp3"
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Save MP3 audio to temporary file
            with open(temp_mp3, "wb") as file:
                response.stream_to_file(temp_mp3)
                
            # Convert MP3 to WAV using pydub
            from pydub import AudioSegment
            sound = AudioSegment.from_mp3(temp_mp3)
            sound = sound.set_frame_rate(16000)  # Match ESP32's sample rate
            sound = sound.set_channels(1)  # Mono
            sound = sound.set_sample_width(2)  # 16-bit
            sound.export(output_file, format="wav")
            
            logger.info(f"Speech saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

class WebSocketServer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.chatgpt_handler = ChatGPTHandler()
        self.active_connections = set()
        
    async def handler(self, websocket):
        """Handle a WebSocket connection"""
        self.active_connections.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New connection from {client_info}")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    # Handle text commands
                    await self.handle_text_command(websocket, message)
                else:
                    # Handle binary audio data
                    self.audio_processor.add_audio_data(message)
                    
        except ConnectionClosed:
            logger.info(f"Connection closed from {client_info}")
        finally:
            self.active_connections.remove(websocket)
            
    async def handle_text_command(self, websocket, message):
        """Process text commands from clients"""
        logger.info(f"Received command: {message}")
        
        if message == "START_AUDIO":
            self.audio_processor.start_recording()
            await websocket.send("Recording started")
            
        elif message == "STOP_AUDIO":
            if self.audio_processor.stop_recording():
                # Process the recording
                self.audio_processor.process_audio()
                
                # Start processing pipeline
                await websocket.send("Processing audio...")
                
                # 1. Transcribe audio
                transcription = await self.chatgpt_handler.transcribe_audio(
                    self.audio_processor.temp_file)
                
                if not transcription:
                    await websocket.send("Failed to transcribe audio")
                    return
                    
                # 2. Generate response from ChatGPT

                ## custom prompt
                custom_prompt = ''' 
                You are a friend to a 5 year old kid. Keep your responses concise and in happy and joyfull tone.
                If the user asks you about your name, you should say "I am Jack, your friend".
                give the results in a way that a 5 year old kid would understand.

                give the results in not more than 3 lines. ----maximum response should be in 3 lines-----
                '''

                response_text = await self.chatgpt_handler.generate_response(transcription, custom_prompt=custom_prompt)
                
                # 3. Convert response to speech
                if await self.chatgpt_handler.text_to_speech(
                    response_text, self.audio_processor.response_file):
                    
                    # 4. Stream the audio back to client
                    await websocket.send("START_PLAYBACK")
                    await self.stream_audio_to_client(websocket, self.audio_processor.response_file)
                    await websocket.send("STOP_PLAYBACK")
                else:
                    await websocket.send("Failed to generate speech")
            else:
                await websocket.send("No audio recorded")
                
        elif message.startswith("DEVICE:"):
            # Handle device identification
            device_id = message.split(":", 1)[1]
            logger.info(f"Device identified as: {device_id}")
            await websocket.send("Device registered")
    async def stream_audio_to_client(self, websocket, filename):
        """Stream audio file to the client in chunks"""
        try:
            # Read file as binary
            with open(filename, 'rb') as f:
                file_content = f.read()
                
            logger.info(f"Streaming {len(file_content)} bytes to client")
            
            # Send in 1KB chunks
            chunk_size = 1024
            for i in range(0, len(file_content), chunk_size):
                chunk = file_content[i:i+chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(0.05)
                
            logger.info("Audio streaming complete")
            return True
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False        
    
            
    async def start_server(self):
        """Start the WebSocket server"""
        server = await websockets.serve(
            self.handler,
            "0.0.0.0",
            PORT
        )
        
        logger.info(f"WebSocket server started on port {PORT}")
        
        # Keep the server running
        await server.wait_closed()

async def main():
    """Main entry point for the server"""
    logger.info("Starting ESP32 voice assistant server")
    
    # Check for OpenAI API key
    if not openai.api_key:
        logger.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return
        
    server = WebSocketServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())