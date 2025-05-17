# esp32_b_speaker_test_server.py

import asyncio
import websockets
import logging
import os

# Configuration
AUDIO_FILE = "test.wav"
PORT = 9000
CHUNK_SIZE = 1024

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ESP32B Speaker Test Server")

async def handle_connection(websocket):
    client = websocket.remote_address
    logger.info(f"Connected: {client}")

    try:
        # Wait for ESP32-B to say hello
        hello_msg = await websocket.recv()
        logger.info(f"Received from ESP32: {hello_msg}")

        if hello_msg != "DEVICE:B":
            await websocket.send("ERROR: Not ESP32-B")
            return

        # Send commands and audio
        await websocket.send("START_PLAYBACK")

        with open(AUDIO_FILE, "rb") as f:
            data = f.read()
            logger.info(f"Streaming {len(data)} bytes of audio")
            for i in range(0, len(data), CHUNK_SIZE):
                await websocket.send(data[i:i+CHUNK_SIZE])
                await asyncio.sleep(0.05)

        await websocket.send("STOP_PLAYBACK")
        logger.info("Finished streaming audio to ESP32-B")

    except websockets.ConnectionClosed:
        logger.warning(f"Connection with {client} closed")
    except Exception as e:
        logger.error(f"Error: {e}")

async def start_test_server():
    logger.info(f"Starting speaker test server on port {PORT}")
    async with websockets.serve(handle_connection, "0.0.0.0", PORT):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        logger.error(f"Test audio file '{AUDIO_FILE}' not found!")
    else:
        asyncio.run(start_test_server())
