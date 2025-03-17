import os
import time
import logging
import tempfile
import subprocess
from supabase import create_client
import uuid
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_video_to_supabase(video_path, bucket_name="audio-files", folder="processed"):
    """
    Upload a video file to Supabase storage
    
    Args:
        video_path (str): Path to the video file
        bucket_name (str): Name of the Supabase storage bucket
        folder (str): Folder within the bucket
        
    Returns:
        str: Public URL of the uploaded video
    """
    try:
        # Create Supabase client
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        
        # Generate a unique filename
        filename = os.path.basename(video_path)
        unique_id = str(uuid.uuid4())[:8]
        storage_path = f"{folder}/{unique_id}_{filename}"
        
        # Upload the file
        with open(video_path, "rb") as f:
            file_data = f.read()
            supabase.storage.from_(bucket_name).upload(storage_path, file_data)
        
        # Get the public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(storage_path)
        
        logger.info(f"Uploaded video to Supabase: {public_url}")
        return public_url
    
    except Exception as e:
        logger.error(f"Error uploading video to Supabase: {e}")
        return None

def create_video_preview(video_path, max_duration=180):
    """
    Create a preview of a video with maximum duration
    
    Args:
        video_path (str): Path to the video file
        max_duration (int): Maximum duration in seconds
        
    Returns:
        str: Path to the preview video
    """
    try:
        # Get the original video duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        
        if result.returncode != 0:
            logger.error(f"Error getting video duration: {result.stderr}")
            return None
            
        duration = float(result.stdout.strip())
        
        # If video is already shorter than max_duration, just copy it
        if duration <= max_duration:
            preview_path = video_path.replace(".mp4", "_preview.mp4")
            import shutil
            shutil.copy(video_path, preview_path)
            return preview_path
        
        # Create a preview by taking the first max_duration seconds
        preview_path = video_path.replace(".mp4", "_preview.mp4")
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', str(max_duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            preview_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        
        # Verify the output file exists and has content
        if os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
            logger.info(f"Created video preview: {preview_path}")
            return preview_path
        else:
            logger.error(f"Failed to create video preview")
            return None
    
    except Exception as e:
        logger.error(f"Error creating video preview: {e}")
        return None

def save_video_metadata(original_url, preview_url, breakpoints):
    """
    Save video metadata to Supabase database
    
    Args:
        original_url (str): URL of the original video
        preview_url (str): URL of the preview video
        breakpoints (list): List of breakpoints in seconds
        
    Returns:
        int: ID of the created record
    """
    try:
        # Create Supabase client
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        
        # Insert the record
        response = supabase.table("videos").insert({
            "original_url": original_url,
            "preview_url": preview_url,
            "breakpoints": breakpoints,
        }).execute()
        
        # Get the ID of the created record
        if response.data and len(response.data) > 0:
            video_id = response.data[0].get("id")
            logger.info(f"Saved video metadata with ID: {video_id}")
            return video_id
        else:
            logger.error(f"Failed to save video metadata: {response}")
            return None
    
    except Exception as e:
        logger.error(f"Error saving video metadata: {e}")
        return None 
    
def save_video_breakpoints(breakpoints_list, total_chunks=None):
    """
    Save the breakpoints for a video and create a new record in the videos table
    
    Args:
        breakpoints_list (list): List of breakpoints arrays, one per audio file
        total_chunks (int, optional): Total number of chunks to process, if pre-calculated
        
    Returns:
        str: ID of the created record
    """
    try:
        if not breakpoints_list:
            logger.error("No breakpoints provided to save_video_breakpoints")
            return None
            
        logger.info(f"Saving breakpoints to Supabase: {breakpoints_list}")
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials in environment variables")
            return None
            
        logger.info(f"Creating Supabase client with URL: {supabase_url}")
        supabase = create_client(supabase_url, supabase_key)
        
        # Calculate a rough estimate of total chunks if not provided
        # The actual count will be updated after audio splitting
        if total_chunks is None:
            total_chunks = 0
            for breakpoints in breakpoints_list:
                # Each audio file will have approximately len(breakpoints) chunks
                # This is only an initial estimate - we'll update it later with the actual count
                total_chunks += len(breakpoints)
        
        logger.info(f"Initial estimate of chunks to process: {total_chunks}")
        
        # Insert a new record with breakpoints and initial status
        data = {
            "breakpoints": breakpoints_list,
            "chunks_total": total_chunks,  # This will be updated later with accurate count
            "chunks_completed": 0,
            "status": "generating",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Inserting record into videos table: {data}")
        response = supabase.table("videos").insert(data).execute()
        
        logger.info(f"Supabase response: {response}")
        
        if response.data and len(response.data) > 0:
            video_id = response.data[0].get("id")
            logger.info(f"Saved video breakpoints with ID: {video_id}")
            return video_id
        else:
            logger.error(f"Failed to save video breakpoints: {response}")
            return None
    
    except Exception as e:
        logger.error(f"Error saving video breakpoints: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
def increment_breakpoints_completed(video_id):
    """
    Increment the number of completed breakpoints for a video
    
    Args:
        video_id (str): ID of the video
        
    Returns:
        list: Updated video data
    """
    try:
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        
        # Get current breakpoints_completed value and total chunks
        get_response = supabase.table("videos").select("chunks_completed,chunks_total").eq("id", video_id).execute()
        
        if not get_response.data or len(get_response.data) == 0:
            logger.error(f"Video with ID {video_id} not found")
            return None
            
        current_breakpoints_completed = get_response.data[0].get("chunks_completed", 0)
        chunks_total = get_response.data[0].get("chunks_total", 0)
        
        # Prevent chunks_completed from exceeding chunks_total
        if current_breakpoints_completed >= chunks_total:
            logger.warning(f"chunks_completed ({current_breakpoints_completed}) has already reached chunks_total ({chunks_total}). Not incrementing.")
            return get_response.data
        
        # Increment the value
        response = supabase.table("videos").update({
            "chunks_completed": current_breakpoints_completed + 1,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }).eq("id", video_id).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Incremented chunks_completed for video {video_id} to {current_breakpoints_completed + 1}/{chunks_total}")
            return response.data
        else:
            logger.error(f"Failed to increment chunks_completed: {response}")
            return None
    
    except Exception as e:
        logger.error(f"Error incrementing chunks_completed: {e}")
        return None
    
def set_video_status(video_id, status):
    """
    Set the status of a video
    
    Args:
        video_id (str): ID of the video
        status (str): Status to set ("generating", "complete", or "failed")
        
    Returns:
        list: Updated video data
    """
    try:
        if status not in ["generating", "complete", "failed"]:
            logger.warning(f"Invalid status: {status}. Using 'failed' instead.")
            status = "failed"
            
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        
        response = supabase.table("videos").update({
            "status": status,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }).eq("id", video_id).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Set status for video {video_id} to '{status}'")
            return response.data
        else:
            logger.error(f"Failed to set video status: {response}")
            return None
    
    except Exception as e:
        logger.error(f"Error setting video status: {e}")
        return None

def update_video_urls(video_id, video_urls):
    """
    Update the video_urls column for a record in the database
    
    Args:
        video_id (str): ID of the video record
        video_urls (list): List of URLs to the videos
        
    Returns:
        bool: True if successful
    """
    try:
        if not video_id:
            logger.error("Cannot update video_urls: No video_id provided")
            return False
            
        if not video_urls:
            logger.warning(f"No video URLs to update for video_id {video_id}")
            return False
            
        logger.info(f"Updating video_urls for {video_id} with {len(video_urls)} URLs")
        
        # Create Supabase client
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        
        # Convert video_urls to JSON array format expected by Supabase
        # Make sure to deduplicate URLs while preserving order
        unique_urls = []
        for url in video_urls:
            if url not in unique_urls and url: # Skip empty URLs
                unique_urls.append(url)
                
        if len(unique_urls) != len(video_urls):
            logger.warning(f"Removed {len(video_urls) - len(unique_urls)} duplicate or empty URLs")
        
        # Update the record
        result = supabase.table("videos").update({
            "video_urls": unique_urls,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", video_id).execute()
        
        if "error" in result:
            logger.error(f"Error updating video_urls: {result['error']}")
            return False
            
        logger.info(f"Successfully updated video_urls for {video_id} with {len(unique_urls)} URLs")
        return True
    
    except Exception as e:
        logger.error(f"Error updating video_urls: {e}")
        return False

def update_chunks_total(video_id, total_chunks):
    """
    Update the total number of chunks for a video
    
    Args:
        video_id (str): ID of the video record
        total_chunks (int): Total number of chunks
        
    Returns:
        bool: True if successful
    """
    try:
        if not video_id:
            logger.error("Cannot update chunks_total: No video_id provided")
            return False
            
        logger.info(f"Updating chunks_total for video {video_id} to {total_chunks}")
        
        # Create Supabase client
        supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
        
        # Update the record
        response = supabase.table("videos").update({
            "chunks_total": total_chunks,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", video_id).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Successfully updated chunks_total for video {video_id} to {total_chunks}")
            return True
        else:
            logger.error(f"Failed to update chunks_total: {response}")
            return False
    
    except Exception as e:
        logger.error(f"Error updating chunks_total: {e}")
        return False
