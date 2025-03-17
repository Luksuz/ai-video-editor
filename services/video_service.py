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
from utils.video_utils import search_pexels_videos, download_video, check_video_validity, create_simple_video, get_video_duration, get_pexels_video_duration
from utils.audio_utils import transcribe_audio
from utils.video_processing import process_video_for_audio, combine_audio_video
from utils.supabase_utils import upload_video_to_supabase, update_chunks_total , save_video_breakpoints, increment_breakpoints_completed, set_video_status, update_video_urls
from utils.audio_utils import get_audio_duration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_audio_from_breakpoints_api(request_data, output_dir="output_videos", save_to_supabase=True, background_tasks=None):
    """
    Process audio from a Supabase URL and split it based on breakpoints

    Args:
        request_data (list): List containing supabase URL and breakpoints
        output_dir (str): Directory to save output videos
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
        # Calculate total chunks across all audio files
        total_chunks = 0
        all_breakpoints = []
        
        # First, process all audio items to calculate total chunks
        for item in request_data:
            supabase_url = item.supabase_url
            breakpoints = item.breakpoints
            
            if not supabase_url or not breakpoints:
                logger.warning(f"Missing supabase_url or breakpoints in item: {item}")
                continue
                
            # Calculate chunks for this item: number of segments is one more than breakpoints
            # Each breakpoint represents the start of a new chunk, plus the last chunk
            item_chunks = len(breakpoints)
            total_chunks += item_chunks
            
            # Store breakpoints for each item
            all_breakpoints.append(breakpoints)
            
        logger.info(f"Processing {len(request_data)} audio files with {total_chunks} total chunks")
        
        # Save breakpoints to Supabase and get the video_id
        video_id = save_video_breakpoints(all_breakpoints, total_chunks)
        
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
                save_to_supabase=save_to_supabase,
                temp_dir=temp_dir,
                video_id=video_id
            )

            # Return immediately with the video_id
            return [], supabase_data
        else:
            # If no background_tasks object is provided, continue processing synchronously
            # This is mainly for testing or direct API calls
            created_videos, updated_supabase_data = await process_audio_complete(
                temp_dir=temp_dir,
                request_data=request_data,
                video_id=video_id,
                output_dir=output_dir,
                combine_videos=False,
                save_to_supabase=save_to_supabase
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


async def process_audio_in_background(request_data, output_dir, save_to_supabase, temp_dir, video_id):
    """
    Process audio in the background after the initial response has been sent
    """
    try:
        await process_audio_complete(
            temp_dir=temp_dir,
            request_data=request_data,
            video_id=video_id,
            output_dir=output_dir,
            combine_videos=False,
            save_to_supabase=save_to_supabase
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


async def process_audio_complete(temp_dir, request_data, video_id, output_dir, combine_videos, save_to_supabase):
    """
    Complete the audio processing after the initial response has been sent
    """
    overall_start_time = time.time()
    timing_report = {}
    audio_chunks = []
    supabase_data = {
        "video_id": video_id,
        "original_url": None,
        "preview_url": None
    }
    
    try:
        # Initialize the sentence classifier agent
        agent_start = time.time()
        sentence_classifier = Agent(
            name="Sentence classifier",
            instructions="You are a video search specialist. Based on the transcription of an audio clip, output only the search query that will be used for finding the most appropriate video on Pexels website. Make the query specific and visual.",
            model="gpt-4o-mini",
            output_type=Query,
        )
        timing_report["initialize_agent"] = time.time() - agent_start
        
        # Process each audio file in the request data
        split_start = time.time()
        chunk_index = 0  # Global chunk index across all files
        
        for file_index, item in enumerate(request_data):
            supabase_url = item.supabase_url
            breakpoints = item.breakpoints
            
            if not supabase_url or not breakpoints:
                logger.warning(f"Missing supabase_url or breakpoints in item {file_index}")
                continue
                
            logger.info(f"Processing audio file {file_index+1}/{len(request_data)}: {supabase_url}")
            
            # Extract the storage key from the URL
            storage_key = supabase_url.split("public/")[1] if "public/" in supabase_url else None
            
            if not storage_key:
                logger.error(f"Could not extract storage key from URL: {supabase_url}")
                continue
            
            # Download the full audio file
            logger.info(f"Downloading audio file from: {supabase_url}")
            supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
            
            try:
                audio_file = supabase.storage.from_("audio-files").download(storage_key)
            except Exception as e:
                logger.error(f"Error downloading from Supabase: {e}")
                # Try direct download as fallback
                try:
                    response = requests.get(supabase_url)
                    response.raise_for_status()
                    audio_file = response.content
                except Exception as download_error:
                    logger.error(f"Error with direct download fallback: {download_error}")
                    continue
            
            # Save the full audio file
            full_audio_path = os.path.join(temp_dir, f"full_audio_{file_index}.mp3")
            with open(full_audio_path, "wb") as f:
                f.write(audio_file)
            
            # Get the original filename from the storage key
            original_filename = os.path.basename(storage_key)
            
            # Get audio duration
            full_audio_duration = get_audio_duration(full_audio_path)
            if full_audio_duration <= 0:
                logger.error(f"Could not determine duration of audio file: {full_audio_path}")
                continue
                
            logger.info(f"Audio file duration: {full_audio_duration:.2f}s")
            
            # Make sure breakpoints are sorted
            breakpoints = sorted(breakpoints)
            
            # Make sure we have start/end breakpoints
            if 0 not in breakpoints:
                breakpoints = [0] + breakpoints
            if full_audio_duration not in breakpoints:
                breakpoints = breakpoints + [full_audio_duration]
                
            logger.info(f"Processing {len(breakpoints)-1} chunks from this audio file")
            
            # Split the audio file based on breakpoints
            for i in range(len(breakpoints) - 1):
                start_time = breakpoints[i]
                end_time = breakpoints[i + 1]
                duration = end_time - start_time
                
                if duration <= 0.1:  # Skip extremely short segments
                    logger.warning(f"Skipping extremely short segment: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
                    continue
                
                # Create a chunk file
                chunk_path = os.path.join(temp_dir, f"chunk_{file_index}_{i:03d}.mp3")
                
                # Use ffmpeg to extract the chunk
                cmd = [
                    'ffmpeg',
                    '-i', full_audio_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c:a', 'copy',  # Copy audio without re-encoding
                    '-y',
                    chunk_path
                ]
                
                import subprocess
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                
                # Verify the chunk was created
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                    # Create a storage key for this chunk
                    chunk_storage_key = f"audio/chunk_{chunk_index:03d}_{original_filename}"
                    
                    # Create an audio chunk object
                    chunk = {
                        "originalFileName": original_filename,
                        "chunkStartTime": start_time,
                        "chunkEndTime": end_time,
                        "chunkDuration": duration,
                        "storageKey": chunk_storage_key,
                        "localPath": chunk_path,  # Add local path for direct access
                        "video_id": video_id,     # Add video_id for tracking
                        "chunk_index": chunk_index,  # Use the global chunk index
                        "file_index": file_index    # Keep track of which file this came from
                    }
                    
                    audio_chunks.append(chunk)
                    logger.info(f"Created chunk {chunk_index} (file {file_index+1}, segment {i+1}): {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
                    
                    # Increment global chunk index
                    chunk_index += 1
                else:
                    logger.warning(f"Failed to create chunk for file {file_index+1}, segment {i+1}")
        
        timing_report["split_audio"] = time.time() - split_start
        
        # Update total chunks in the database to match the actual number of chunks created
        total_chunks = len(audio_chunks)
        logger.info(f"Successfully split {len(request_data)} audio files into {total_chunks} chunks")
        
        # Update the total chunks count in the database with the ACTUAL number of chunks
        update_chunks_total(video_id, total_chunks)
        logger.info(f"Updated chunks_total in database to {total_chunks}")
        
        # Upload each chunk to Supabase immediately after splitting
        upload_start = time.time()
        chunk_video_urls = {}
        
        # Initialize the dictionary for this video ID
        if video_id not in chunk_video_urls:
            chunk_video_urls[video_id] = {}
        
        # Process the audio chunks
        chunks_start = time.time()
        created_videos = await process_audio_chunks_local(
            audio_chunks, sentence_classifier, output_dir
        )
        timing_report["process_chunks"] = time.time() - chunks_start
        
        # The video_urls have already been saved to Supabase in process_audio_chunks_local
        # Just mark the video as complete
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
    
    # Track video upload URLs for each chunk
    chunk_video_urls = {}
    
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
            video_id = chunk.get("video_id")
            chunk_index = chunk.get("chunk_index")
            
            if video_data.get('videos') and len(video_data['videos']) > 0:
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
                                        raw_path, video_path, audio_duration
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
                                        timing_report[f"total_video_attempt_{video_index}"] = time.time() - video_attempt_start
                                        
                                        # Upload the video to Supabase
                                        upload_start = time.time()
                                        video_url = upload_video_to_supabase(output_path)
                                        timing_report["upload_video"] = time.time() - upload_start
                                        
                                        if video_url and video_id:
                                            # Store the video URL by video_id and chunk_index
                                            if video_id not in chunk_video_urls:
                                                chunk_video_urls[video_id] = {}
                                            chunk_video_urls[video_id][chunk_index] = video_url
                                            logger.info(f"Uploaded video chunk {i+1} to Supabase: {video_url}")
                                            # Log the current state of chunk_video_urls for this video
                                            stored_indices = sorted([int(idx) for idx in chunk_video_urls[video_id].keys()])
                                            logger.info(f"Current stored chunks for video {video_id}: {stored_indices} (total: {len(stored_indices)})")
                                        
                                        # Increment the breakpoints completed counter
                                        if video_id:
                                            increment_breakpoints_completed(video_id)
                                        
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
                        
                        # Upload the fallback video to Supabase
                        upload_start = time.time()
                        video_url = upload_video_to_supabase(output_path)
                        timing_report["upload_fallback_video"] = time.time() - upload_start
                        
                        if video_url and video_id:
                            # Store the video URL by video_id and chunk_index
                            if video_id not in chunk_video_urls:
                                chunk_video_urls[video_id] = {}
                            chunk_video_urls[video_id][chunk_index] = video_url
                            logger.info(f"Uploaded fallback video chunk {i+1} to Supabase: {video_url}")
                            # Log the current state of chunk_video_urls for this video
                            stored_indices = sorted([int(idx) for idx in chunk_video_urls[video_id].keys()])
                            logger.info(f"Current stored chunks for video {video_id}: {stored_indices} (total: {len(stored_indices)})")
                        else:
                            logger.warning(f"Failed to upload fallback chunk {i+1} (index {chunk_index}) to Supabase or missing video_id")
                        
                        # Increment the breakpoints completed counter
                        if video_id:
                            increment_breakpoints_completed(video_id)
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
        
        logger.info("="*50 + "\n")
    
    # Update video_urls in Supabase for each video
    for video_id, chunk_urls in chunk_video_urls.items():
        # Log the total chunks processed vs stored
        logger.info(f"Video {video_id}: Processed {len(audio_chunks)} chunks, storing {len(chunk_urls)} URLs")
        
        # Convert the dictionary to a sorted list based on chunk index
        # Ensure chunk_index is treated as an integer for proper numerical sorting
        sorted_items = sorted(chunk_urls.items(), key=lambda x: int(x[0]))
        sorted_urls = [url for _, url in sorted_items]
        
        # Log the sorted URLs for debugging
        logger.info(f"Sorted video URLs for video {video_id}:")
        for i, url in enumerate(sorted_urls):
            logger.info(f"  Position {i}: {url}")
            
        # Get the actual chunk indices that we processed
        actual_indices = set([chunk["chunk_index"] for chunk in audio_chunks if chunk["video_id"] == video_id])
        # Get the stored chunk indices
        stored_indices = set([int(idx) for idx in chunk_urls.keys()])
        
        # Calculate missing chunks
        missing_indices = actual_indices - stored_indices
        if missing_indices:
            logger.warning(f"Missing chunk indices for video {video_id}: {sorted(missing_indices)}")
            
        # Update the video_urls column in Supabase
        update_video_urls(video_id, sorted_urls)
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Error removing temporary directory: {e}")

    return created_videos


async def process_custom_video_for_chunk(custom_video_url, chunk_video_url, video_id, chunk_index):
    """
    Process a custom video to match the duration of a chunk video while preserving the original audio
    
    Args:
        custom_video_url (str): URL to the custom video in Supabase storage
        chunk_video_url (str): URL to the chunk video in Supabase storage
        video_id (str): ID of the video record in the database
        chunk_index (int): Index of the chunk being replaced
        
    Returns:
        tuple: (str, bool) - URL to the processed video and success flag
    """
    try:
        logger.info(f"Processing custom video to match chunk {chunk_index} for video {video_id}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clean up URLs by removing query parameters if present
            clean_custom_video_url = custom_video_url.split('?')[0]
            clean_chunk_video_url = chunk_video_url.split('?')[0]
            
            logger.info(f"Downloading custom video from: {clean_custom_video_url}")
            logger.info(f"Downloading chunk video from: {clean_chunk_video_url}")
            
            # Download the custom video
            custom_video_path = os.path.join(temp_dir, f"custom_video_{video_id}.mp4")
            if not download_video(clean_custom_video_url, custom_video_path):
                logger.error(f"Failed to download custom video: {clean_custom_video_url}")
                return None, False
                
            # Download the chunk video
            chunk_video_path = os.path.join(temp_dir, f"chunk_video_{video_id}_{chunk_index}.mp4")
            if not download_video(clean_chunk_video_url, chunk_video_path):
                # If download fails, try to extract direct download URL using Supabase client
                try:
                    logger.info("Attempting to download chunk video directly from Supabase")
                    # Parse the URL to get the path - assuming URL format is like:
                    # https://[project].supabase.co/storage/v1/object/public/[bucket]/[path]
                    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
                    
                    url_parts = clean_chunk_video_url.split("/")
                    bucket_index = url_parts.index("public") + 1 if "public" in url_parts else -1
                    
                    if bucket_index != -1 and bucket_index < len(url_parts):
                        bucket_name = url_parts[bucket_index]
                        storage_path = "/".join(url_parts[bucket_index+1:])
                        
                        # Download directly using Supabase client
                        logger.info(f"Downloading from bucket: {bucket_name}, path: {storage_path}")
                        try:
                            data = supabase.storage.from_(bucket_name).download(storage_path)
                            with open(chunk_video_path, "wb") as f:
                                f.write(data)
                            if os.path.exists(chunk_video_path) and os.path.getsize(chunk_video_path) > 0:
                                logger.info(f"Successfully downloaded chunk video using Supabase client")
                            else:
                                logger.error(f"Downloaded file is empty")
                                return None, False
                        except Exception as e:
                            logger.error(f"Error downloading from Supabase: {e}")
                            return None, False
                    else:
                        logger.error(f"Could not parse Supabase URL: {clean_chunk_video_url}")
                        return None, False
                except Exception as e:
                    logger.error(f"Failed to download chunk video: {e}")
                    return None, False
                
            # Get the duration of the chunk video
            chunk_duration = get_video_duration(chunk_video_path)
            if chunk_duration <= 0:
                logger.error(f"Failed to get duration of chunk video: {chunk_video_path}")
                return None, False
                
            logger.info(f"Chunk video duration: {chunk_duration} seconds")
            
            # Extract audio from the original chunk video
            chunk_audio_path = os.path.join(temp_dir, f"chunk_audio_{video_id}_{chunk_index}.aac")
            logger.info(f"Extracting audio from original chunk video")
            
            extract_audio_cmd = [
                'ffmpeg',
                '-i', chunk_video_path,
                '-vn',  # No video
                '-acodec', 'aac',
                '-b:a', '192k',
                '-y',
                chunk_audio_path
            ]
            
            try:
                import subprocess
                result = subprocess.run(extract_audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                logger.info(f"Successfully extracted audio from chunk video")
                
                # Verify the audio file was created
                if not os.path.exists(chunk_audio_path) or os.path.getsize(chunk_audio_path) == 0:
                    logger.error(f"Failed to extract audio: audio file is empty or not created")
                    return None, False
            except Exception as e:
                logger.error(f"Error extracting audio from chunk video: {e}")
                return None, False
                
            # Process the custom video to match the chunk duration
            processed_video_path = os.path.join(temp_dir, f"processed_custom_{video_id}_{chunk_index}.mp4")
            success = process_video_for_audio(
                custom_video_path,
                processed_video_path,
                chunk_duration,
            )
            
            if not success:
                logger.error(f"Failed to process custom video: {custom_video_path}")
                return None, False
                
            # Combine the processed video with the original audio
            final_video_path = os.path.join(temp_dir, f"final_video_{video_id}_{chunk_index}.mp4")
            logger.info(f"Combining processed video with original audio")
            
            combine_cmd = [
                'ffmpeg',
                '-i', processed_video_path,  # Video input
                '-i', chunk_audio_path,      # Audio input
                '-c:v', 'copy',              # Copy video codec (no re-encoding)
                '-c:a', 'aac',               # Audio codec
                '-map', '0:v:0',             # Use video from first input
                '-map', '1:a:0',             # Use audio from second input
                '-shortest',                 # Finish encoding when the shortest input stream ends
                '-y',
                final_video_path
            ]
            
            try:
                result = subprocess.run(combine_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                logger.info(f"Successfully combined video with original audio")
                
                # Verify the final video was created
                if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
                    logger.error(f"Failed to create final video: file is empty or not created")
                    return None, False
            except Exception as e:
                logger.error(f"Error combining video with original audio: {e}")
                return None, False
                
            # Replace the original video by uploading to the same path
            # Extract the bucket and path from the chunk_video_url
            supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
            
            # Parse the URL to get the path - assuming URL format is like:
            # https://[project].supabase.co/storage/v1/object/public/[bucket]/[path]
            url_parts = clean_chunk_video_url.split("/")
            bucket_index = url_parts.index("public") + 1 if "public" in url_parts else -1
            
            if bucket_index == -1 or bucket_index >= len(url_parts):
                logger.error(f"Could not parse Supabase URL: {clean_chunk_video_url}")
                return None, False
                
            bucket_name = url_parts[bucket_index]
            storage_path = "/".join(url_parts[bucket_index+1:])
            
            logger.info(f"Uploading to bucket: {bucket_name}, path: {storage_path}")
            
            # Upload the final video to replace the original chunk
            try:
                with open(final_video_path, "rb") as f:
                    file_data = f.read()
                    supabase.storage.from_(bucket_name).update(storage_path, file_data)
                logger.info(f"Successfully updated video in Supabase at {storage_path}")
            except Exception as e:
                logger.error(f"Error updating file in Supabase: {e}")
                
                # Try creating the file if update fails (might not exist yet)
                try:
                    logger.info("Attempting to create file instead of updating")
                    with open(final_video_path, "rb") as f:
                        file_data = f.read()
                        supabase.storage.from_(bucket_name).upload(storage_path, file_data)
                    logger.info(f"Successfully created video in Supabase at {storage_path}")
                except Exception as e2:
                    logger.error(f"Error creating file in Supabase: {e2}")
                    return None, False
            
            # Return the original URL that was passed in, since we're replacing the file at the same location
            # This ensures we preserve any query parameters or exact format the client was using
            return chunk_video_url, True
            
    except Exception as e:
        logger.error(f"Error processing custom video: {e}")
        return None, False