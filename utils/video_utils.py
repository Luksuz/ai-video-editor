import logging
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_pexels_videos(query, per_page=10, api_key=None, used_urls=None, target_duration=None):
    """
    Search for videos on Pexels based on a query.

    Args:
        query (str): The search term
        per_page (int): Number of results to return
        api_key (str): Pexels API key, defaults to environment variable
        used_urls (set): Set of already used video URLs to avoid duplicates
        target_duration (float): Target duration in seconds to find matching videos

    Returns:
        dict: JSON response from Pexels API
    """
    if used_urls is None:
        used_urls = set()

    if api_key is None:
        api_key = os.environ.get("PEXELS_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided and PEXELS_API_KEY not found in environment variables"
            )

    headers = {"Authorization": api_key}

    # Increase per_page to get more options for duration matching
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Filter out videos that have already been used
        if "videos" in result:
            filtered_videos = []
            for video in result["videos"]:
                # Check if any of the video files have been used
                video_files = video.get("video_files", [])
                all_urls = [file.get("link") for file in video_files if file.get("link")]

                # Only include the video if none of its URLs have been used
                if not any(url in used_urls for url in all_urls):
                    filtered_videos.append(video)

            result["videos"] = filtered_videos

            # If we have a target duration, sort videos by how close they are to the target
            if target_duration is not None and filtered_videos:
                for video in filtered_videos:
                    # Get the duration of the video
                    video_duration = video.get("duration", 0)
                    # Calculate how close this video is to our target duration
                    video["duration_match"] = abs(video_duration - target_duration)

                # Sort videos by duration match (closest first)
                filtered_videos.sort(key=lambda x: x.get("duration_match", float("inf")))
                result["videos"] = filtered_videos

        return result
    except Exception as e:
        logger.error(f"Error searching Pexels: {e}")
        return {"videos": []}


def download_video(video_url, output_path, max_retries=3):
    """
    Download a video from a URL to a specified path

    Args:
        video_url (str): URL of the video to download
        output_path (str): Path where the video will be saved
        max_retries: Maximum number of retry attempts

    Returns:
        bool: True if download was successful
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(video_url, stream=True, timeout=10)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                logger.warning(f"Warning: Content length is 0 for {video_url}")
                return False

            block_size = 1024  # 1 Kibibyte

            with open(output_path, "wb") as file, tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            else:
                logger.warning(f"Downloaded file is empty or missing: {output_path}")
                return False

        except Exception as e:
            logger.error(f"Download attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return False

    return False


def check_video_validity(video_path):
    """
    Check if a video file is valid and can be processed

    Args:
        video_path (str): Path to the video file

    Returns:
        bool: True if the video is valid
    """
    try:
        # Check if the file exists and has content
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return False

        # Try to get video information
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )

        if result.returncode != 0:
            logger.error(f"Error checking video validity: {result.stderr}")
            return False

        # Parse the JSON output
        import json

        info = json.loads(result.stdout)

        # Check if we have video streams
        if "streams" not in info or len(info["streams"]) == 0:
            logger.warning(f"No video streams found in {video_path}")
            return False

        # Get the first video stream
        stream = info["streams"][0]

        # Check width and height
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        if width == 0 or height == 0:
            logger.warning(f"Invalid dimensions in {video_path}: {width}x{height}")
            return False

        # Check aspect ratio
        aspect_ratio = width / height if height > 0 else 0

        # Print video information
        logger.info(f"Video dimensions: {width}x{height}, Aspect ratio: {aspect_ratio:.2f}")

        # Video is valid
        return True

    except Exception as e:
        logger.error(f"Error checking video validity: {e}")
        return False


def create_simple_video(output_path, duration, width=1920, height=1080):
    """
    Create a simple black video with the specified duration

    Args:
        output_path (str): Path where the video will be saved
        duration (float): Duration of the video in seconds
        width (int): Width of the video
        height (int): Height of the video

    Returns:
        bool: True if successful
    """
    try:
        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:d={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",  # Optimize for streaming
            "-g",
            "30",  # Set keyframe interval
            "-keyint_min",
            "30",  # Minimum keyframe interval
            "-force_key_frames",
            f"expr:gte(t,n_forced*1)",  # Force keyframe every second
            "-metadata",
            "title=",  # Clear metadata
            "-metadata",
            "comment=",
            "-y",
            output_path,
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.error(f"Error creating simple video: {e}")
        return False


def get_video_duration(video_path):
    """
    Get the duration of a video file

    Args:
        video_path (str): Path to the video file

    Returns:
        float: Duration in seconds, or 0 if an error occurs
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )

        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            logger.error(f"Error getting video duration: {result.stderr}")
            return 0
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        return 0


def get_audio_duration(audio_path):
    """
    Get the duration of an audio file

    Args:
        audio_path (str): Path to the audio file

    Returns:
        float: Duration in seconds, or 0 if an error occurs
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )

        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            logger.error(f"Error getting audio duration: {result.stderr}")
            return 0
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 0


def get_pexels_video_duration(video_data):
    """
    Extract the duration from Pexels video data

    Args:
        video_data (dict): Video data from Pexels API

    Returns:
        float: Duration in seconds
    """
    return float(video_data.get("duration", 0))
