from typing import List, Optional

from pydantic import BaseModel, Field


class AudioBreakpoint(BaseModel):
    """Model for audio breakpoints request"""

    supabase_url: str = Field(..., description="URL to the audio file in Supabase storage")
    breakpoints: List[float] = Field(..., description="List of breakpoints in seconds")


class AudioProcessingRequest(BaseModel):
    """Model for audio processing request"""

    data: List[AudioBreakpoint] = Field(..., description="List of audio files with breakpoints")
    combine_videos: bool = Field(False, description="Whether to combine all videos into one")
    output_dir: Optional[str] = Field(
        "output_videos", description="Directory to save output videos"
    )


class VideoResponse(BaseModel):
    """Model for video processing response"""

    video_paths: List[str] = Field(..., description="Paths to the created videos")
    processing_time: float = Field(..., description="Total processing time in seconds")
    success: bool = Field(..., description="Whether the processing was successful")
    message: str = Field("", description="Additional information or error message")


class Query(BaseModel):
    query: str = Field(..., description="Query to search for videos")


class Video(BaseModel):
    video_url: str = Field(..., description="URL to the video file")
    video_duration: float = Field(..., description="Duration of the video in seconds")
    video_width: int = Field(..., description="Width of the video")
    video_height: int = Field(..., description="Height of the video")


class VideoResponseWithStorage(BaseModel):
    """Model for video processing response with Supabase storage URLs"""

    video_id: str = Field(..., description="ID of the video record in the database")
    original_url: Optional[str] = Field(
        "", description="URL to the original video in Supabase storage"
    )
    preview_url: Optional[str] = Field(
        "", description="URL to the preview video in Supabase storage"
    )
    processing_time: float = Field(..., description="Total processing time in seconds")
    success: bool = Field(..., description="Whether the processing was successful")
    message: str = Field("", description="Additional information or error message")
