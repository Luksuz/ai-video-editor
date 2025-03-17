import logging
import os
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from models.video_models import AudioProcessingRequest, VideoResponse, VideoResponseWithStorage, CustomVideoChunkRequest, CustomVideoResponse
from services.video_service import process_audio_from_breakpoints_api, process_custom_video_for_chunk
import time
import os
import logging

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
            detail=f"Error processing and storing audio: {str(e)}"
        )


@router.post("/replace-chunk", response_model=CustomVideoResponse, status_code=status.HTTP_200_OK)
async def replace_video_chunk(request: CustomVideoChunkRequest):
    """
    Replace a video chunk with a custom video.
    
    This endpoint accepts URLs for a custom video and a chunk video from Supabase storage,
    processes the custom video to match the chunk video's duration, and then replaces
    the chunk video with the processed custom video.
    """
    try:
        start_time = time.time()
        
        # Process the custom video to match the chunk video duration
        processed_url, success = await process_custom_video_for_chunk(
            request.custom_video_url,
            request.chunk_video_url,
            request.video_id,
            request.chunk_index
        )
        
        if not success or not processed_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process custom video"
            )
        
        processing_time = time.time() - start_time
        
        return CustomVideoResponse(
            video_url=processed_url,
            processing_time=processing_time,
            success=True,
            message=f"Successfully replaced chunk {request.chunk_index} for video {request.video_id}"
        )
    except Exception as e:
        logger.error(f"Error replacing video chunk: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error replacing video chunk: {str(e)}"
        ) 