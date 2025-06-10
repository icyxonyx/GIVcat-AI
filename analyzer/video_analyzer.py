import os
import gc
import time
import tempfile
from collections import Counter

import psutil
import torch
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image

from analyzer.image_analyzer import analyze_image_file, cleanup_models

# ───  optimizations ───────────────────────────────────────────────────

MAX_VIDEO_DURATION = 300  # 5 minutes max processing time
MAX_FRAME_SIZE = (512, 512)  # Shrink large frames for memory efficiency
MIN_FRAMES_FOR_ANALYSIS = 3  # Minimum frames to analyze
MAX_FRAMES_TO_ANALYZE = 8  # Never analyze more than this many frames


def check_system_resources():
    """Check available RAM and GPU memory before processing a video."""
    memory = psutil.virtual_memory()
    gpu_memory = 0

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_memory = props.total_memory / (1024**3)  # in GB

    return {
        "ram_available_gb": memory.available / (1024**3),
        "ram_percent_used": memory.percent,
        "gpu_memory_gb": gpu_memory,
    }


def get_optimal_frame_count(duration, file_size_mb):
    """Return how many keyframes to extract, based on duration and system resources."""
    resources = check_system_resources()

    # Base number of frames on video duration
    if duration <= 10:
        base_frames = min(6, max(3, int(duration / 2)))
    elif duration <= 60:
        base_frames = min(8, max(4, int(duration / 10)))
    else:
        base_frames = min(10, max(5, int(duration / 15)))

    # Adjust down if RAM is limited
    if resources["ram_available_gb"] < 2:
        base_frames = max(3, base_frames // 2)
    elif resources["ram_percent_used"] > 80:
        base_frames = max(3, int(base_frames * 0.7))

    # Adjust if file is very large
    if file_size_mb > 500:
        base_frames = max(3, int(base_frames * 0.8))
    if file_size_mb > 2000:
        # but if >2 GB, we skip altogether (handled in analyze_video)
        base_frames = 3

    return min(MAX_FRAMES_TO_ANALYZE, max(MIN_FRAMES_FOR_ANALYSIS, base_frames))


def extract_keyframes_smart(video_path, num_frames):
    """
    Extract up to `num_frames` keyframes (uniformly sampled, or begin/mid/end),
    resize them if needed, save temporarily to disk, and run analyze_image_file on each.
    Returns a flat list of per-frame category predictions (e.g. ["people", "pets", ...]).
    """
    temp_files = []
    predictions = []
    clip = None

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration

        if duration > MAX_VIDEO_DURATION:
            print(
                f"⚠️ Video too long ({duration:.1f}s), analyzing first {MAX_VIDEO_DURATION}s"
            )
            duration = MAX_VIDEO_DURATION

        # Build a list of timestamps to sample:
        if duration <= 10:
            times = np.linspace(1, duration - 1, num_frames)
        else:
            # Split into three segments: beginning, middle, end
            segment_size = duration / 3
            times = []
            for segment in range(3):
                segment_start = segment * segment_size
                segment_frames = max(1, num_frames // 3)
                segment_times = np.linspace(
                    segment_start + 1, segment_start + segment_size - 1, segment_frames
                )
                times.extend(segment_times.tolist())
            times = times[:num_frames]

        print(f"🎬 Extracting {len(times)} frames from {os.path.basename(video_path)}")

        for t in times:
            try:
                if t >= duration:
                    continue

                frame = clip.get_frame(t)  # NumPy array (H × W × 3)
                pil_image = Image.fromarray(frame).convert("RGB")

                # Resize if bigger than MAX_FRAME_SIZE
                if (
                    pil_image.width > MAX_FRAME_SIZE[0]
                    or pil_image.height > MAX_FRAME_SIZE[1]
                ):
                    pil_image.thumbnail(MAX_FRAME_SIZE, Image.Resampling.LANCZOS)

                # Save to temp file (JPEG, medium quality)
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False, dir=tempfile.gettempdir()
                )
                temp_file.close()
                pil_image.save(temp_file.name, "JPEG", quality=85, optimize=True)
                temp_files.append(temp_file.name)

                # Analyze with the image classifier
                frame_predictions = analyze_image_file(temp_file.name)
                predictions.extend(frame_predictions)

                # Cleanup intermediate objects
                del frame, pil_image
                gc.collect()

                # Small delay to let the GPU cool if needed
                time.sleep(0.05)
            except Exception as e:
                msg = str(e)
                print(f"⚠️ Error extracting frame at {t:.1f}s: {msg[:50]}")
                continue

        return predictions

    except Exception as e:
        msg = str(e)
        print(f"❌ Error during keyframe extraction: {msg[:100]}")
        return ["uncategorized/extraction-error"]

    finally:
        # Delete all temporary JPEGs
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass

        # Close video clip
        if clip:
            try:
                clip.close()
                del clip
            except Exception:
                pass

        # Force garbage collection and free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_video_info_safe(video_path):
    """
    Try to load VideoFileClip only to read duration/fps/size.
    If it fails, return defaults.
    """
    try:
        clip = VideoFileClip(video_path)
        info = {
            "duration": clip.duration,
            "fps": clip.fps,
            "size": clip.size,
            "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
        }
        clip.close()
        return info
    except Exception as e:
        print(f"⚠️ Could not get video info: {str(e)[:50]}")
        return {
            "duration": 30,  # assume a short clip
            "fps": 30,
            "size": (640, 480),
            "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
        }


def analyze_video(video_path):
    """
    Main entry point for video analysis. Returns a single‐element list [category].

    1. Skip if file not found or >2 GB.
    2. Decide how many frames to sample based on duration and RAM.
    3. extract_keyframes_smart(...) → list of per‐frame predictions.
    4. Filter out error‐type predictions.
    5. OVERRIDE ORDER:
       a) if any 'memes' → ["memes"]
       b) elif any 'pets' → ["pets"]
       c) elif any 'people' → ["people"]
    6. Otherwise, pick the single most common valid label.
    """
    try:
        print(f"🎥 Analyzing video: {os.path.basename(video_path)}")

        if not os.path.exists(video_path):
            return ["uncategorized/file-not-found"]

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 2000:
            print(f"⚠️ Video too large ({file_size_mb:.1f}MB), skipping analysis")
            return ["uncategorized/too-large"]

        resources = check_system_resources()
        if resources["ram_percent_used"] > 85:
            print("⚠️ High memory usage, forcing minimal analysis (3 frames)")
            num_frames = 3
        else:
            info = get_video_info_safe(video_path)
            num_frames = get_optimal_frame_count(info["duration"], file_size_mb)

        predictions = extract_keyframes_smart(video_path, num_frames)
        if not predictions:
            print("❌ No predictions generated from video frames")
            return ["uncategorized/no-analysis"]

        # Filter out any “error‐like” captions
        valid_predictions = [
            p
            for p in predictions
            if not any(
                err in p.lower() for err in ["error", "invalid", "too-small", "blank"]
            )
        ]

        if not valid_predictions:
            # If every frame was “error …”, return whichever error is most common
            error_counter = Counter(predictions)
            return [error_counter.most_common(1)[0][0]]

        # ─── OVERRIDE A: MEMES ─────────────────────────────────────────────────────
        if any(p.lower() == "memes" for p in valid_predictions):
            print(
                "😂 Forcing 'memes' because at least one frame was classified as a meme."
            )
            return ["memes"]
        # ──────────────────────────────────────────────────────────────────────────

        # ─── OVERRIDE B: PETS ──────────────────────────────────────────────────────
        if any(p.lower() == "pets" for p in valid_predictions):
            print("🐾 Forcing 'pets' because at least one frame detected a pet.")
            return ["pets"]
        # ──────────────────────────────────────────────────────────────────────────

        # ─── OVERRIDE C: PEOPLE ───────────────────────────────────────────────────
        if any(p.lower() == "people" for p in valid_predictions):
            print("👤 Forcing 'people' because at least one frame detected a person.")
            return ["people"]
        # ──────────────────────────────────────────────────────────────────────────

        # Count how many times each category appeared, pick the top one
        counter = Counter(valid_predictions)
        most_common = counter.most_common(3)
        total_valid = len(valid_predictions)

        # If the top label covers >50%, choose it; otherwise pick the first anyway
        if most_common[0][1] > total_valid * 0.5:
            result = [most_common[0][0]]
        else:
            result = [most_common[0][0]]

        print(f"✅ Video categorized as: {result}")
        return result

    except Exception as e:
        msg = str(e)
        print(
            f"❌ Critical error analyzing video {os.path.basename(video_path)}: {msg[:100]}"
        )
        return ["uncategorized/critical-error"]

    finally:
        # Always attempt to clean up any loaded models
        cleanup_models()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.1)
