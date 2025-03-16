import asyncio
import logging
import os
import shutil
import tempfile
import time

import requests
from agents import Agent, Runner
from pydantic import BaseModel
from supabase import create_client

from models.video_models import AudioBreakpoint, Query
from utils.audio_utils import get_audio_duration, transcribe_audio
from utils.supabase_utils import (
    create_video_preview,
    increment_breakpoints_completed,
    save_video_breakpoints,
    save_video_metadata,
    set_video_status,
    upload_video_to_supabase,
)
from utils.video_processing import (
    combine_audio_video,
    concatenate_videos_with_crossfade,
    process_video_for_audio,
)
from utils.video_utils import (
    check_video_validity,
    create_simple_video,
    download_video,
    get_pexels_video_duration,
    get_video_duration,
    search_pexels_videos,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_audio_from_breakpoints_api(
    request_data,
    output_dir="output_videos",
    combine_videos=False,
    save_to_supabase=True,
    background_tasks=None,
):
    """
    Process audio from a Supabase URL and split it based on breakpoints

    Args:
        request_data (list): List containing supabase URL and breakpoints
        output_dir (str): Directory to save output videos
        combine_videos (bool): Whether to combine all videos into one
        save_to_supabase (bool): Whether to save the videos to Supabase storage
        background_tasks: FastAPI BackgroundTasks object

    Returns:
        tuple: (list, dict) - Paths to created videos and Supabase metadata
    """
    # Create initial response data
    supabase_data = {"video_id": None, "original_url": None, "preview_url": None}

    if not request_data or len(request_data) == 0:
        logger.warning("No request data provided")
        return [], supabase_data

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Get the first item in the request data
        item = request_data[0]
        supabase_url = item.supabase_url
        breakpoints = item.breakpoints

        if not supabase_url or not breakpoints:
            logger.warning("Missing supabase_url or breakpoints in request data")
            return [], supabase_data

        # Extract the storage key from the URL
        storage_key = supabase_url.split("public/")[1] if "public/" in supabase_url else None

        if not storage_key:
            logger.error(f"Could not extract storage key from URL: {supabase_url}")
            return [], supabase_data

        # Download the full audio file
        logger.info(f"Downloading audio file from: {supabase_url}")
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

        try:
            audio_file = supabase.storage.from_("audio-files").download(storage_key)
        except Exception as e:
            logger.error(f"Error downloading from Supabase: {e}")
            # Try direct download as fallback
            response = requests.get(supabase_url)
            response.raise_for_status()
            audio_file = response.content

        # Save the full audio file
        full_audio_path = os.path.join(temp_dir, "full_audio.mp3")
        with open(full_audio_path, "wb") as f:
            f.write(audio_file)

        # Get the original filename from the storage key
        original_filename = os.path.basename(storage_key)

        full_audio_duration = get_audio_duration(full_audio_path)

        # Add 0 as the first breakpoint if not present
        if 0 not in breakpoints:
            breakpoints = [0] + breakpoints + [full_audio_duration]

        # Save breakpoints to Supabase and get the video_id
        video_id = save_video_breakpoints(breakpoints)

        if not video_id:
            logger.error("Failed to save breakpoints and get video_id")
            return [], supabase_data

        logger.info(f"Created video record with ID: {video_id}")

        # Update status to "generating"
        set_video_status(video_id, "generating")

        # Update supabase_data with the video_id
        supabase_data["video_id"] = video_id

        # If we have a background_tasks object, add the processing task to it
        if background_tasks:
            # Create a copy of the request data for the background task
            background_tasks.add_task(
                process_audio_in_background,
                request_data=request_data,
                output_dir=output_dir,
                combine_videos=combine_videos,
                save_to_supabase=save_to_supabase,
                temp_dir=temp_dir,
                full_audio_path=full_audio_path,
                original_filename=original_filename,
                breakpoints=breakpoints,
                video_id=video_id,
            )

            # Return immediately with the video_id
            return [], supabase_data
        else:
            # If no background_tasks object is provided, continue processing synchronously
            # This is mainly for testing or direct API calls
            created_videos, updated_supabase_data = await process_audio_complete(
                temp_dir=temp_dir,
                full_audio_path=full_audio_path,
                original_filename=original_filename,
                breakpoints=breakpoints,
                video_id=video_id,
                output_dir=output_dir,
                combine_videos=combine_videos,
                save_to_supabase=save_to_supabase,
            )

            return created_videos, updated_supabase_data

    except Exception as e:
        logger.error(f"Error in initial processing: {e}")
        # Clean up temporary directory if we're not using background tasks
        if not background_tasks:
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.error(f"Error removing temporary directory: {cleanup_error}")
        return [], supabase_data


async def process_audio_in_background(
    request_data,
    output_dir,
    combine_videos,
    save_to_supabase,
    temp_dir,
    full_audio_path,
    original_filename,
    breakpoints,
    video_id,
):
    """
    Process audio in the background after the initial response has been sent
    """
    try:
        await process_audio_complete(
            temp_dir=temp_dir,
            full_audio_path=full_audio_path,
            original_filename=original_filename,
            breakpoints=breakpoints,
            video_id=video_id,
            output_dir=output_dir,
            combine_videos=combine_videos,
            save_to_supabase=save_to_supabase,
        )
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        # Set status to failed if there was an error
        set_video_status(video_id, "failed")
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error removing temporary directory: {e}")


async def process_audio_complete(
    temp_dir,
    full_audio_path,
    original_filename,
    breakpoints,
    video_id,
    output_dir,
    combine_videos,
    save_to_supabase,
):
    """
    Complete the audio processing after the initial response has been sent
    """
    overall_start_time = time.time()
    timing_report = {}
    audio_chunks = []
    supabase_data = {"video_id": video_id, "original_url": None, "preview_url": None}

    try:
        # Split the audio file based on breakpoints
        split_start = time.time()
        for i in range(len(breakpoints) - 1):
            start_time = breakpoints[i]
            end_time = breakpoints[i + 1]
            duration = end_time - start_time

            # Create a chunk file
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")

            # Use ffmpeg to extract the chunk
            cmd = [
                "ffmpeg",
                "-i",
                full_audio_path,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c:a",
                "copy",  # Copy audio without re-encoding
                "-y",
                chunk_path,
            ]

            import subprocess

            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
            )

            # Verify the chunk was created
            if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                # Create a storage key for this chunk
                chunk_storage_key = f"audio/chunk_{i:03d}_{original_filename}"

                # Create an audio chunk object
                chunk = {
                    "originalFileName": original_filename,
                    "chunkStartTime": start_time,
                    "chunkEndTime": end_time,
                    "chunkDuration": duration,
                    "storageKey": chunk_storage_key,
                    "localPath": chunk_path,  # Add local path for direct access
                    "video_id": video_id,  # Add video_id for tracking
                    "chunk_index": i,  # Add chunk index for tracking
                }

                audio_chunks.append(chunk)
                logger.info(
                    f"Created chunk {i+1}/{len(breakpoints)-1}: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)"
                )
            else:
                logger.warning(f"Failed to create chunk {i+1}")
        timing_report["split_audio"] = time.time() - split_start

        # Initialize the sentence classifier agent
        agent_start = time.time()
        sentence_classifier = Agent(
            name="Sentence classifier",
            instructions="You are a video search specialist. Based on the transcription of an audio clip, output only the search query that will be used for finding the most appropriate video on Pexels website. Make the query specific and visual.",
            model="gpt-4o-mini",
            output_type=Query,
        )
        timing_report["initialize_agent"] = time.time() - agent_start

        # Process the audio chunks
        chunks_start = time.time()
        created_videos = await process_audio_chunks_local(
            audio_chunks, sentence_classifier, output_dir
        )
        timing_report["process_chunks"] = time.time() - chunks_start

        final_video_path = None

        # Optionally concatenate all videos into a single file
        if combine_videos and len(created_videos) > 1:
            concat_start = time.time()
            output_path = os.path.join(output_dir, "combined_video.mp4")
            if concatenate_videos_with_crossfade(
                created_videos, output_path, transition_duration=0.5
            ):
                logger.info(f"Combined video saved to {output_path}")
                timing_report["concatenate_videos"] = time.time() - concat_start
                final_video_path = output_path
                created_videos = [output_path]
        elif len(created_videos) > 0:
            final_video_path = created_videos[0]

        # Save to Supabase if requested
        if save_to_supabase and final_video_path:
            supabase_start = time.time()

            # Create a preview of the video
            preview_path = create_video_preview(final_video_path)

            # Upload the original video
            original_url = upload_video_to_supabase(final_video_path)

            # Upload the preview video
            preview_url = None
            if preview_path:
                preview_url = upload_video_to_supabase(preview_path)

            # Save metadata to database
            if original_url and preview_url:
                # Update Supabase data
                supabase_data = {
                    "video_id": video_id,
                    "original_url": original_url,
                    "preview_url": preview_url,
                }

                # Update the video record with URLs
                supabase = create_client(
                    os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
                )
                supabase.table("videos").update(
                    {"original_url": original_url, "preview_url": preview_url, "status": "complete"}
                ).eq("id", video_id).execute()

            timing_report["supabase_upload"] = time.time() - supabase_start
        else:
            # If we're not saving to Supabase but processing was successful
            if len(created_videos) > 0:
                set_video_status(video_id, "complete")
            else:
                set_video_status(video_id, "failed")

        # Calculate overall processing time
        overall_time = time.time() - overall_start_time
        timing_report["total_processing"] = overall_time

        # Print overall timing report
        logger.info("\n" + "=" * 50)
        logger.info("OVERALL TIMING REPORT")
        logger.info("=" * 50)
        logger.info(
            f"Total processing time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)"
        )
        logger.info(f"Number of chunks processed: {len(audio_chunks)}")
        logger.info(
            f"Average time per chunk: {timing_report.get('process_chunks', 0)/max(1, len(audio_chunks)):.2f} seconds"
        )
        logger.info("-" * 50)

        # Print individual function timings
        for func_name, duration in timing_report.items():
            if func_name != "total_processing":
                percentage = (duration / overall_time) * 100
                logger.info(f"{func_name}: {duration:.2f}s ({percentage:.1f}% of total)")

        logger.info("=" * 50 + "\n")

        return created_videos, supabase_data

    except Exception as e:
        logger.error(f"Error processing audio from breakpoints: {e}")
        # Set status to failed if there was an error
        set_video_status(video_id, "failed")
        return [], supabase_data


async def process_audio_chunks_local(audio_chunks, sentence_classifier, output_dir="output_videos"):
    """
    Process audio chunks from local files, transcribe them, find matching videos,
    and create final videos with synchronized audio and video.

    Args:
        audio_chunks (list): List of audio chunk objects with file info and local paths
        sentence_classifier (Agent): The agent to classify sentences
        output_dir (str): Directory to save output videos

    Returns:
        list: Paths to created videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    created_videos = []

    # Track used video URLs to avoid duplicates
    used_video_urls = set()

    # Create subdirectories
    video_dir = os.path.join(temp_dir, "video")
    final_dir = os.path.join(temp_dir, "final")

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Process each audio chunk
    for i, chunk in enumerate(audio_chunks):
        chunk_start_time = time.time()
        timing_report = {}

        try:
            logger.info(
                f"\nProcessing chunk {i+1}/{len(audio_chunks)}: {chunk['originalFileName']} ({chunk['chunkStartTime']:.2f}s to {chunk['chunkEndTime']:.2f}s)"
            )

            # Use the local path of the audio file
            audio_path = chunk["localPath"]

            # Get the audio duration
            start_time = time.time()
            audio_duration = get_audio_duration(audio_path)
            if audio_duration <= 0:
                audio_duration = chunk[
                    "chunkDuration"
                ]  # Use the provided duration if ffprobe fails
            timing_report["get_audio_duration"] = time.time() - start_time

            logger.info(f"Audio duration: {audio_duration:.2f} seconds")

            # Transcribe the audio
            start_time = time.time()
            transcription = transcribe_audio(audio_path)
            if not transcription:
                logger.warning(
                    f"Failed to transcribe audio chunk {i+1}, using filename as fallback"
                )
                transcription = chunk["originalFileName"].replace("-", " ").replace("_", " ")
            timing_report["transcribe_audio"] = time.time() - start_time

            logger.info(f"Transcription: {transcription}")

            # Generate search query based on transcription
            start_time = time.time()

            class Query(BaseModel):
                query: str

            result = await Runner.run(
                sentence_classifier, f"Find a relevant video for this audio: {transcription}"
            )
            query = result.final_output.query
            timing_report["generate_query"] = time.time() - start_time

            logger.info(f"Search query: {query}")

            # Search for videos
            start_time = time.time()
            video_data = search_pexels_videos(
                query, per_page=15, used_urls=used_video_urls, target_duration=audio_duration
            )
            timing_report["search_videos"] = time.time() - start_time

            video_success = False

            if video_data.get("videos") and len(video_data["videos"]) > 0:
                # Try each video until we find one that works
                for video_index, video in enumerate(video_data["videos"]):
                    if video_success:
                        break

                    video_attempt_start = time.time()

                    # Get the video duration from Pexels data
                    pexels_duration = get_pexels_video_duration(video)
                    duration_match = abs(pexels_duration - audio_duration)

                    # Print duration match information
                    logger.info(
                        f"Video {video_index+1}: Duration {pexels_duration:.2f}s (diff: {duration_match:.2f}s)"
                    )

                    video_files = video.get("video_files", [])

                    # Find a suitable video file
                    video_files = sorted(
                        video_files, key=lambda x: x.get("height", 0), reverse=True
                    )
                    download_url = None

                    for file in video_files:
                        if file.get("file_type") == "video/mp4" and file.get("height", 0) <= 720:
                            download_url = file.get("link")
                            break

                    if not download_url and video_files:
                        download_url = video_files[0].get("link")

                    if download_url and download_url not in used_video_urls:
                        # Download the video
                        raw_path = os.path.join(temp_dir, f"raw_{i:03d}_{video_index}.mp4")

                        download_start = time.time()
                        download_success = download_video(download_url, raw_path)
                        timing_report[f"download_video_{video_index}"] = (
                            time.time() - download_start
                        )

                        if download_success:
                            # Add to used URLs
                            used_video_urls.add(download_url)

                            # Check if the video is valid
                            validity_start = time.time()
                            is_valid = check_video_validity(raw_path)
                            timing_report[f"check_validity_{video_index}"] = (
                                time.time() - validity_start
                            )

                            if is_valid:
                                # Process the video to match audio duration
                                video_path = os.path.join(video_dir, f"video_{i:03d}.mp4")

                                # Use different processing approach based on duration match
                                processing_success = False

                                processing_start = time.time()
                                # If duration is very close, use simple trimming
                                if duration_match < 1.0:
                                    logger.info(f"Good duration match, using simple trimming")
                                    processing_success = process_video_for_audio(
                                        raw_path, video_path, audio_duration, max_speed_change=1.1
                                    )
                                else:
                                    # Otherwise use standard processing
                                    processing_success = process_video_for_audio(
                                        raw_path, video_path, audio_duration
                                    )
                                timing_report[f"process_video_{video_index}"] = (
                                    time.time() - processing_start
                                )

                                if processing_success:
                                    # Combine audio and video
                                    combine_start = time.time()
                                    final_path = os.path.join(final_dir, f"final_{i:03d}.mp4")
                                    combine_success = combine_audio_video(
                                        video_path, audio_path, final_path
                                    )
                                    timing_report[f"combine_audio_video_{video_index}"] = (
                                        time.time() - combine_start
                                    )

                                    if combine_success:
                                        # Copy to output directory with meaningful name
                                        output_name = f"{chunk['originalFileName']}_{chunk['chunkStartTime']:.2f}_{chunk['chunkEndTime']:.2f}.mp4"
                                        output_path = os.path.join(output_dir, output_name)
                                        shutil.copy(final_path, output_path)
                                        created_videos.append(output_path)
                                        video_success = True
                                        logger.info(f"Successfully created video for chunk {i+1}")
                                        timing_report[f"total_video_attempt_{video_index}"] = (
                                            time.time() - video_attempt_start
                                        )

                                        # Increment the breakpoints completed counter
                                        if "video_id" in chunk:
                                            increment_breakpoints_completed(chunk["video_id"])

                                        break

            # If we couldn't create a video, create a simple black video with audio
            if not video_success:
                fallback_start = time.time()
                logger.info(f"Creating fallback video for chunk {i+1}")
                video_path = os.path.join(video_dir, f"video_{i:03d}.mp4")
                if create_simple_video(video_path, audio_duration):
                    final_path = os.path.join(final_dir, f"final_{i:03d}.mp4")
                    if combine_audio_video(video_path, audio_path, final_path):
                        # Copy to output directory with meaningful name
                        output_name = f"{chunk['originalFileName']}_{chunk['chunkStartTime']:.2f}_{chunk['chunkEndTime']:.2f}.mp4"
                        output_path = os.path.join(output_dir, output_name)
                        shutil.copy(final_path, output_path)
                        created_videos.append(output_path)
                        logger.info(f"Created fallback video for chunk {i+1}")

                        # Increment the breakpoints completed counter
                        if "video_id" in chunk:
                            increment_breakpoints_completed(chunk["video_id"])
                timing_report["fallback_video"] = time.time() - fallback_start

        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")

        # Calculate total processing time for this chunk
        chunk_total_time = time.time() - chunk_start_time
        timing_report["total_chunk_processing"] = chunk_total_time

        # Print timing report for this chunk
        logger.info("\n" + "=" * 50)
        logger.info(f"TIMING REPORT FOR CHUNK {i+1}/{len(audio_chunks)}")
        logger.info("=" * 50)
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        logger.info(
            f"Total processing time: {chunk_total_time:.2f} seconds ({chunk_total_time/60:.2f} minutes)"
        )
        logger.info("-" * 50)

        # Print individual function timings
        for func_name, duration in timing_report.items():
            if func_name != "total_chunk_processing":
                percentage = (duration / chunk_total_time) * 100
                logger.info(f"{func_name}: {duration:.2f}s ({percentage:.1f}% of total)")

        logger.info("=" * 50 + "\n")

    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Error removing temporary directory: {e}")

    return created_videos
