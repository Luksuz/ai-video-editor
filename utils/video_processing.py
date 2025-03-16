import logging
import os
import platform
import shutil
import subprocess
import time

from .video_utils import get_audio_duration, get_video_duration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_video_for_audio(
    input_path,
    output_path,
    target_duration,
    max_speed_change=1.5,
    target_width=1920,
    target_height=1080,
):
    """
    Process a video to match the target audio duration with minimal distortion

    Args:
        input_path (str): Path to the input video
        output_path (str): Path to the processed video
        target_duration (float): Target duration in seconds
        max_speed_change (float): Maximum factor to speed up or slow down video
        target_width (int): Target width for the output video
        target_height (int): Target height for the output video

    Returns:
        bool: True if successful
    """
    try:
        # Get the duration of the input video
        input_duration = get_video_duration(input_path)

        if input_duration <= 0:
            logger.error(f"Could not determine duration of {input_path}")
            return False

        logger.info(f"Input video duration: {input_duration:.2f}s, Target: {target_duration:.2f}s")

        # Calculate the ratio between target and input duration
        ratio = target_duration / input_duration

        # Create a temporary file for intermediate processing
        temp_output = output_path + ".temp.mp4"

        # Disable hardware acceleration as it's causing issues
        hw_accel = []
        video_codec = "libx264"

        # Use ultrafast preset for maximum speed
        preset = "ultrafast"

        # Get number of CPU cores and use them all
        cpu_count = os.cpu_count() or 4
        threads = min(16, cpu_count)  # Cap at 16 threads to avoid diminishing returns

        # Prepare scaling filter - simplified for speed
        scale_filter = f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"

        # Determine the best approach based on the ratio
        if 1 / max_speed_change <= ratio <= max_speed_change:
            # We can adjust the speed within acceptable limits
            logger.info(f"Adjusting video speed by factor of {ratio:.2f}")

            # Use setpts filter to adjust speed
            speed_filter = f"setpts={1/ratio}*PTS"

            # Combine with scaling filter
            combined_filter = f"{scale_filter},{speed_filter}"

            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-vf",
                combined_filter,
                "-an",  # Remove audio
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(threads),
                "-movflags",
                "+faststart",  # Optimize for streaming
                "-y",
                temp_output,
            ]

            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600
            )

        elif ratio > 1:  # Target is longer than input
            # For very long videos, use a different approach
            if target_duration > 60 and ratio > 5:
                logger.info(
                    f"Long duration detected ({target_duration:.2f}s), using optimized approach"
                )

                # Create a temporary file for the scaled video
                scaled_video = input_path + ".scaled.mp4"

                # First, scale the video without changing speed (faster)
                scale_cmd = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-vf",
                    scale_filter,
                    "-an",  # Remove audio
                    "-c:v",
                    video_codec,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    str(threads),
                    "-y",
                    scaled_video,
                ]

                subprocess.run(
                    scale_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=600,
                )

                # Then use stream_loop which is more efficient than complex filters
                loop_cmd = [
                    "ffmpeg",
                    "-stream_loop",
                    "-1",  # Loop indefinitely
                    "-i",
                    scaled_video,
                    "-t",
                    str(target_duration),  # Limit to target duration
                    "-an",  # Remove audio
                    "-c:v",
                    video_codec,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    str(threads),
                    "-y",
                    temp_output,
                ]

                subprocess.run(
                    loop_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=600,
                )

                # Clean up
                if os.path.exists(scaled_video):
                    os.remove(scaled_video)

            else:
                # Loop the video to reach target duration
                logger.info(f"Looping video to reach target duration")

                # Calculate how many times we need to loop
                loop_count = int(ratio) + 1

                cmd = [
                    "ffmpeg",
                    "-stream_loop",
                    str(loop_count - 1),  # -1 because we already have 1 copy
                    "-i",
                    input_path,
                    "-vf",
                    scale_filter,
                    "-t",
                    str(target_duration),  # Limit to target duration
                    "-an",  # Remove audio
                    "-c:v",
                    video_codec,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    str(threads),
                    "-y",
                    temp_output,
                ]

                # Increase timeout for longer videos
                timeout_value = max(600, int(target_duration * 1.5))
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_value,
                )

        else:  # ratio < 1, Target is shorter than input
            # Extract the most interesting segment
            logger.info(f"Extracting segment from video")

            # Try to extract from the middle for more interesting content
            start_time = max(0, (input_duration - target_duration) / 2)

            cmd = [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-i",
                input_path,
                "-vf",
                scale_filter,
                "-t",
                str(target_duration),
                "-an",  # Remove audio
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(threads),
                "-y",
                temp_output,
            ]

            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600
            )

        # Verify the output duration
        temp_duration = get_video_duration(temp_output)
        logger.info(f"Processed video duration: {temp_duration:.2f}s")

        # If the duration is still significantly off, do a second pass with exact trimming
        if abs(temp_duration - target_duration) > 0.5:
            logger.info(f"Adjusting duration in second pass to exactly {target_duration:.2f}s")

            cmd = [
                "ffmpeg",
                "-i",
                temp_output,
                "-t",
                str(target_duration),
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(threads),
                "-y",
                output_path,
            ]

            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600
            )

            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        else:
            # Just rename the temp file
            shutil.move(temp_output, output_path)

        # Final verification
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            final_duration = get_video_duration(output_path)
            logger.info(
                f"Final video duration: {final_duration:.2f}s (target: {target_duration:.2f}s)"
            )
            return True
        else:
            logger.error(f"Failed to create processed video: {output_path}")
            return False

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return False


def combine_audio_video(video_path, audio_path, output_path):
    """
    Combine a video file with an audio file
    Optimized for maximum performance on Mac systems

    Args:
        video_path (str): Path to the video file
        audio_path (str): Path to the audio file
        output_path (str): Path where the combined video will be saved

    Returns:
        bool: True if successful
    """
    try:
        # Disable hardware acceleration as it's causing issues
        hw_accel = []
        video_codec = "libx264"
        preset = "ultrafast"

        # Get number of CPU cores and use them all
        cpu_count = os.cpu_count() or 4
        threads = min(16, cpu_count)  # Cap at 16 threads

        # Get the duration of the audio file
        audio_duration = get_audio_duration(audio_path)

        # Get the duration of the video file
        video_duration = get_video_duration(video_path)

        logger.info(f"Audio duration: {audio_duration:.2f}s, Video duration: {video_duration:.2f}s")

        # Ensure the video is exactly the same length as the audio
        if abs(video_duration - audio_duration) > 0.1:
            logger.info(f"Adjusting video duration to match audio exactly")
            temp_video = video_path + ".temp.mp4"

            if video_duration > audio_duration:
                # Trim the video
                cmd = [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-t",
                    str(audio_duration),
                    "-c:v",
                    video_codec,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    str(threads),
                    "-an",
                    "-y",
                    temp_video,
                ]
            else:
                # Loop the video if needed
                cmd = [
                    "ffmpeg",
                    "-stream_loop",
                    "10",  # Loop up to 10 times if needed
                    "-i",
                    video_path,
                    "-t",
                    str(audio_duration),
                    "-c:v",
                    video_codec,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    str(threads),
                    "-an",
                    "-y",
                    temp_video,
                ]

            # Increase timeout for longer videos
            timeout_value = max(600, int(audio_duration * 1.5))
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_value,
            )
            video_path = temp_video

        # Combine the video with the audio using re-encoding to ensure clean output
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",  # Ensure output is only as long as the shortest input
            "-threads",
            str(threads),
            "-y",
            output_path,
        ]

        # Increase timeout for longer videos
        timeout_value = max(600, int(audio_duration * 1.5))
        subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_value
        )

        # Clean up temporary file if it exists
        if video_path.endswith(".temp.mp4") and os.path.exists(video_path):
            os.remove(video_path)

        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Double-check the final duration
            final_duration = get_video_duration(output_path)
            logger.info(f"Combined clip duration: {final_duration:.2f}s")
            return True
        else:
            logger.error(f"Failed to create combined video: {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error combining audio and video: {e}")
        return False


def concatenate_videos_with_crossfade(video_paths, output_path, transition_duration=0.5):
    """
    Concatenate multiple videos with crossfade transitions

    Args:
        video_paths (list): List of paths to the videos to concatenate
        output_path (str): Path where the concatenated video will be saved
        transition_duration (float): Duration of fade transition in seconds

    Returns:
        bool: True if successful
    """
    if not video_paths:
        logger.warning("No videos to concatenate")
        return False

    if len(video_paths) == 1:
        # If only one video, just copy it
        try:
            shutil.copy(video_paths[0], output_path)
            return True
        except Exception as e:
            logger.error(f"Error copying single video: {e}")
            return False

    try:
        # Disable hardware acceleration as it's causing issues
        hw_accel = []
        video_codec = "libx264"
        preset = "ultrafast"

        # Create temporary directory for intermediate files
        import tempfile

        temp_dir = tempfile.mkdtemp()

        # Process videos in batches to avoid complex filter graphs
        batch_size = 4  # Process 4 videos at a time
        intermediate_videos = []

        for batch_start in range(0, len(video_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(video_paths))
            batch_videos = video_paths[batch_start:batch_end]

            if len(batch_videos) == 1:
                # If only one video in batch, just add it directly
                intermediate_videos.append(batch_videos[0])
                continue

            # Create output path for this batch
            batch_output = os.path.join(temp_dir, f"batch_{batch_start}.mp4")

            # Create a complex filter for crossfading all videos in this batch
            filter_complex = ""
            for i in range(len(batch_videos) - 1):
                # Get duration of current video
                duration = get_video_duration(batch_videos[i])
                overlap_start = max(0, duration - transition_duration)

                # Add crossfade between current and next video
                if i == 0:
                    filter_complex += f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={overlap_start}[v01];"
                    filter_complex += f"[0:a][1:a]acrossfade=d={transition_duration}[a01];"
                else:
                    filter_complex += f"[v{i-1}{i}][{i+1}:v]xfade=transition=fade:duration={transition_duration}:offset={overlap_start}[v{i}{i+1}];"
                    filter_complex += (
                        f"[a{i-1}{i}][{i+1}:a]acrossfade=d={transition_duration}[a{i}{i+1}];"
                    )

            # Add final output mapping
            last_idx = len(batch_videos) - 2
            filter_complex += f"[v{last_idx}{last_idx+1}][a{last_idx}{last_idx+1}]"

            # Build the ffmpeg command
            cmd = ["ffmpeg"]

            # Add input files
            for video in batch_videos:
                cmd.extend(["-i", video])

            # Add filter complex and output options
            cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    f"[v{last_idx}{last_idx+1}]",
                    "-map",
                    f"[a{last_idx}{last_idx+1}]",
                    "-c:v",
                    video_codec,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-y",
                    batch_output,
                ]
            )

            # Run the command with a generous timeout
            total_duration = sum(get_video_duration(v) for v in batch_videos)
            timeout_value = max(600, int(total_duration * 1.5))

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_value,
                )
                if os.path.exists(batch_output) and os.path.getsize(batch_output) > 0:
                    intermediate_videos.append(batch_output)
                else:
                    logger.warning(f"Failed to create batch video {batch_start}")
                    # Fall back to adding individual videos
                    intermediate_videos.extend(batch_videos)
            except Exception as e:
                logger.error(f"Error processing batch {batch_start}: {e}")
                # Fall back to adding individual videos
                intermediate_videos.extend(batch_videos)

        # If we have multiple intermediate videos, concatenate them
        if len(intermediate_videos) > 1:
            # Create a concat list for the intermediate videos
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                concat_list_path = f.name
                for video_path in intermediate_videos:
                    f.write(f"file '{os.path.abspath(video_path)}'\n")

            # Use the concat demuxer for the final combination (faster)
            final_cmd = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_list_path,
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-y",
                output_path,
            ]

            # Use a longer timeout for the final concatenation
            final_timeout = 600  # 10 minutes
            subprocess.run(
                final_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=final_timeout,
            )

            # Clean up
            os.unlink(concat_list_path)
        elif len(intermediate_videos) == 1:
            # If we only have one intermediate video, just copy it
            shutil.copy(intermediate_videos[0], output_path)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        # Verify the output
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.error(f"Error concatenating videos with crossfade: {e}")
        return False
