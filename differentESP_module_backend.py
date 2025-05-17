# voice_assistant_server.py

import asyncio
import logging
import os
import wave
import numpy as np
from dotenv import load_dotenv
import websockets
import openai
from scipy import signal
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
PORT = 8000

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


class AudioProcessor:
    def __init__(self):
        self.buffer = bytearray()
        self.temp_file = "temp_input.wav"
        self.response_file = "temp_output.wav"

    def start(self):
        self.buffer = bytearray()

    def add(self, data):
        self.buffer.extend(data)

    def stop_and_save(self):
        if not self.buffer:
            return False
        with wave.open(self.temp_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(self.buffer)
        return True

    def process(self):
        try:
            audio = np.frombuffer(self.buffer, dtype=np.int16)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 32767 * 0.9
            b, a = signal.butter(4, 100 / (SAMPLE_RATE / 2), 'highpass')
            audio = signal.filtfilt(b, a, audio).astype(np.int16)
            self.buffer = bytearray(audio.tobytes())
        except Exception as e:
            logger.error(f"Audio processing error: {e}")


class ChatGPTHandler:
    def __init__(self):
        self.system_prompt = os.getenv("CHATGPT_SYSTEM_PROMPT", "You are a helpful voice assistant.")
        self.client = openai.OpenAI()
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.max_context = 10

    async def transcribe(self, filename):
        with open(filename, "rb") as f:
            response = self.client.audio.transcriptions.create(model="whisper-1", file=f)
            return response.text

    async def chat(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        self.history = [self.history[0]] + self.history[-self.max_context:]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.history,
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        self.history = [self.history[0]] + self.history[-self.max_context:]
        return reply

    async def text_to_speech(self, text, out_path):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        temp_mp3 = "output.mp3"
        response.stream_to_file(temp_mp3)
        sound = AudioSegment.from_mp3(temp_mp3)
        sound.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(SAMPLE_WIDTH).export(out_path, format="wav")
        return out_path


class VoiceAssistantServer:
    def __init__(self):
        self.sender_conn = None
        self.receiver_conn = None
        self.processor = AudioProcessor()
        self.chatbot = ChatGPTHandler()

    async def handler(self, websocket):
        client = websocket.remote_address
        logger.info(f"New connection: {client}")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self.handle_command(websocket, message)
                else:
                    if websocket == self.sender_conn:
                        self.processor.add(message)
        except websockets.ConnectionClosed:
            logger.info(f"Connection closed: {client}")
        finally:
            if websocket == self.sender_conn:
                self.sender_conn = None
            elif websocket == self.receiver_conn:
                self.receiver_conn = None

    async def handle_command(self, websocket, message):
        if message.startswith("DEVICE:"):
            role = message.split(":")[1].strip()
            if role == "A":
                self.sender_conn = websocket
                await websocket.send("Registered as ESP32-A")
            elif role == "B":
                self.receiver_conn = websocket
                await websocket.send("Registered as ESP32-B")
        elif message == "START_AUDIO":
            self.processor.start()
            await websocket.send("Recording started")
        elif message == "STOP_AUDIO":
            if self.processor.stop_and_save():
                self.processor.process()
                await websocket.send("Processing audio...")
                text = await self.chatbot.transcribe(self.processor.temp_file)
                reply = await self.chatbot.chat(text)
                await self.chatbot.text_to_speech(reply, self.processor.response_file)
                await self.stream_to_receiver()
            else:
                await websocket.send("No audio received")
        elif message == "RESET_MEMORY":
            self.chatbot.history = [{"role": "system", "content": self.chatbot.system_prompt}]
            await websocket.send("Memory reset.")

    async def stream_to_receiver(self):
        if not self.receiver_conn:
            logger.warning("ESP32-B not connected")
            return
        await self.receiver_conn.send("START_PLAYBACK")
        with open(self.processor.response_file, "rb") as f:
            data = f.read()
            for i in range(0, len(data), 1024):
                await self.receiver_conn.send(data[i:i+1024])
                await asyncio.sleep(0.05)
        await self.receiver_conn.send("STOP_PLAYBACK")
        logger.info("Audio sent to ESP32-B")

    async def start(self):
        async with websockets.serve(self.handler, "0.0.0.0", PORT):
            logger.info(f"Server started on port {PORT}")
            await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(VoiceAssistantServer().start())
