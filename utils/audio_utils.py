import logging
import os
import subprocess
import tempfile

from openai import OpenAI

from .video_utils import get_audio_duration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_tts_audio(text, output_path, voice="alloy"):
    """
    Generate TTS audio for a given text using OpenAI's API

    Args:
        text (str): The text to convert to speech
        output_path (str): Path where the audio file will be saved
        voice (str): The voice to use for TTS

    Returns:
        tuple: (bool, float) - Success status and audio duration in seconds
    """
    try:
        client = OpenAI()

        # Use the recommended streaming approach
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=text,
        ) as response:
            # Write the streaming response to a file
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

        # Check if the file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Get the duration of the audio file
            duration = get_audio_duration(output_path)
            return True, duration
        else:
            logger.warning("Generated audio file is empty or missing")
            return False, 0

    except Exception as e:
        logger.error(f"Error generating TTS audio: {e}")
        return False, 0


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using OpenAI's Whisper API

    Args:
        audio_path (str): Path to the audio file

    Returns:
        str: Transcription text
    """
    try:
        client = OpenAI()
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcription.text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""


def split_audio_file(input_path, start_time, duration, output_path):
    """
    Split an audio file based on start time and duration

    Args:
        input_path (str): Path to the input audio file
        start_time (float): Start time in seconds
        duration (float): Duration in seconds
        output_path (str): Path where the split audio will be saved

    Returns:
        bool: True if successful
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-c:a",
            "copy",  # Copy audio without re-encoding
            "-y",
            output_path,
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            logger.error(f"Failed to create split audio: {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error splitting audio file: {e}")
        return False
