import os
import gc
import time
from tqdm import tqdm
from utils.file_utils import get_media_files, move_file_to_category
from analyzer.image_analyzer import analyze_image_file, cleanup_models
from analyzer.video_analyzer import analyze_video
from analyzer.gif_analyzer import analyze_gif
from utils.cache import load_cache, save_cache
import torch

SOURCE = "data"
DEST = "output"

# Memory management
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
BATCH_SIZE = 5  # Process files in small batches
CACHE_SAVE_INTERVAL = 10  # Save cache every 10 files


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_file(file_path, cache):
    """Process single file with memory management"""
    if file_path in cache:
        return file_path, cache[file_path], True  # True = from cache

    try:
        # Check file size first
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            result = ["uncategorized/too-large"]
            cache[file_path] = result
            return file_path, result, False

        # Determine file type and process
        file_ext = file_path.lower()
        if file_ext.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            categories = analyze_image_file(file_path)
        elif file_ext.endswith((".mp4", ".mov", ".avi", ".mkv")):
            categories = analyze_video(file_path)
        elif file_ext.endswith((".gif", ".gifv")):
            categories = analyze_gif(file_path)
        else:
            categories = ["uncategorized/unknown-type"]

        # Move file and cache result
        move_file_to_category(file_path, categories, DEST)
        cache[file_path] = categories

        return file_path, categories, False

    except Exception as e:
        error_msg = f"uncategorized/error-{str(e)[:20]}"
        cache[file_path] = [error_msg]
        return file_path, [error_msg], False


def process_batch(files_batch, cache):
    """Process a batch of files with memory cleanup"""
    results = []

    for file_path in files_batch:
        file_path, categories, from_cache = process_file(file_path, cache)
        results.append((file_path, categories, from_cache))

        # Cleanup after each file
        if not from_cache:
            cleanup_memory()
            time.sleep(0.1)  # Small delay to prevent GPU overheating

    return results


def main():
    print("———————————————— 🚀 Media Sorter ————————————————")
    print(f"📁 Source: {SOURCE}")
    print(f"📁 Destination: {DEST}")

    # Load cache
    cache = load_cache()

    # Get all media files
    print("📋 Scanning for media files...")
    files = get_media_files(SOURCE)
    total_files = len(files)

    if total_files == 0:
        print("❌ No media files found in the source directory!")
        return

    print(f"🔍 Found {total_files} media files")

    # process every file, cached or not, but skip double‐processing in process_file()
    processed, errors = 0, 0

    for file_path in tqdm(files, desc="Processing files", unit="file"):
        file_path, categories, from_cache = process_file(file_path, cache)

        if not from_cache:
            processed += 1
            if any("error" in c for c in categories):
                errors += 1
            cleanup_memory()
            time.sleep(0.1)

        # Optional: save cache every N files
        if processed and processed % CACHE_SAVE_INTERVAL == 0:
            save_cache(cache)

    # final flush
    save_cache(cache)
    cleanup_models()
    cleanup_memory()

    # Print summary
    print("\n" + "=" * 50)
    print("✅ Processing Complete!")
    print(f"📊 Total files: {total_files}")
    print(f"🔄 Processed: {processed}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Output directory: {DEST}")

    if errors > 0:
        print(f"⚠️  {errors} files had errors - check 'uncategorized/error-*' folders")


if __name__ == "__main__":
    main()
