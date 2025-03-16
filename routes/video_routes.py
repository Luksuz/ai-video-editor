import logging
import os
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from models.video_models import AudioProcessingRequest, VideoResponse, VideoResponseWithStorage
from services.video_service import process_audio_from_breakpoints_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["Video Processing"])


@router.post("/process", response_model=VideoResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_audio(request: AudioProcessingRequest, background_tasks: BackgroundTasks):
    """
    Process audio files into videos based on breakpoints.

    This endpoint accepts audio files from Supabase storage and breakpoints,
    then processes them into videos with matching visuals.
    """
    try:
        start_time = time.time()

        # Ensure output directory exists
        os.makedirs(request.output_dir, exist_ok=True)

        # Process the audio - this will return quickly after downloading the audio
        # and setting up the background task
        created_videos, supabase_data = await process_audio_from_breakpoints_api(
            request.data,
            request.output_dir,
            request.combine_videos,
            save_to_supabase=False,  # Don't save to Supabase for this endpoint
            background_tasks=background_tasks,
        )

        processing_time = time.time() - start_time

        return VideoResponse(
            video_paths=created_videos,
            processing_time=processing_time,
            success=True,
            message=f"Processing started with ID: {supabase_data['video_id']}. Check status using this ID.",
        )
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}",
        )


@router.post(
    "/process-and-store",
    response_model=VideoResponseWithStorage,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_audio_and_store(
    request: AudioProcessingRequest, background_tasks: BackgroundTasks
):
    """
    Process audio files into videos and store them in Supabase.

    This endpoint accepts audio files from Supabase storage and breakpoints,
    then processes them into videos with matching visuals, and stores the
    results in Supabase storage and database.
    """
    try:
        start_time = time.time()

        # Ensure output directory exists
        os.makedirs(request.output_dir, exist_ok=True)

        # Process the audio and save to Supabase - this will return quickly after downloading the audio
        # and setting up the background task
        created_videos, supabase_data = await process_audio_from_breakpoints_api(
            request.data,
            request.output_dir,
            request.combine_videos,
            save_to_supabase=True,
            background_tasks=background_tasks,
        )

        processing_time = time.time() - start_time

        if not supabase_data["video_id"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save video to Supabase",
            )

        return VideoResponseWithStorage(
            video_id=supabase_data["video_id"],
            original_url=supabase_data.get(
                "original_url", ""
            ),  # Will be updated later in background task
            preview_url=supabase_data.get(
                "preview_url", ""
            ),  # Will be updated later in background task
            processing_time=processing_time,
            success=True,
            message=f"Processing started with ID: {supabase_data['video_id']}. Check status using this ID.",
        )
    except Exception as e:
        logger.error(f"Error processing and storing audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing and storing audio: {str(e)}",
        )
