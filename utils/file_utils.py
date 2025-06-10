import os
import shutil
import hashlib
from pathlib import Path
import time

# Supported file extensions (expanded list)
SUPPORTED_EXTENSIONS = {
    "images": {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".jfif",
        ".heic",
        ".heif",
    },
    "videos": {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".3gp",
        ".ogv",
    },
    "gifs": {".gif", ".gifv"},
}


def get_file_hash(file_path, chunk_size=8192):
    """Generate SHA256 hash for file to detect duplicates"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # Use first 16 chars for speed
    except Exception:
        return None


def is_file_accessible(file_path):
    """Check if file is accessible and not corrupted"""
    try:
        if not os.path.exists(file_path):
            return False

        # Check if file is not empty
        if os.path.getsize(file_path) == 0:
            return False

        # Try to open the file briefly to check accessibility
        with open(file_path, "rb") as f:
            f.read(1)
        return True
    except (PermissionError, OSError, IOError):
        return False


def get_media_files(folder, include_subdirs=True):
    """
    Recursively get all supported media files with optimizations
    """
    if not os.path.exists(folder):
        print(f"⚠️ Source folder '{folder}' does not exist!")
        return []

    media_files = []
    all_extensions = set()
    for ext_group in SUPPORTED_EXTENSIONS.values():
        all_extensions.update(ext_group)

    print(f"🔍 Scanning {folder} for media files...")

    def scan_directory(directory):
        try:
            entries = os.scandir(directory)
            for entry in entries:
                if entry.is_file():
                    file_ext = Path(entry.name).suffix.lower()
                    if file_ext in all_extensions:
                        file_path = entry.path
                        # Quick accessibility check
                        if is_file_accessible(file_path):
                            media_files.append(file_path)
                        else:
                            print(f"⚠️ Skipping inaccessible file: {entry.name}")
                elif entry.is_dir() and include_subdirs:
                    # Recursively scan subdirectories
                    scan_directory(entry.path)
        except PermissionError:
            print(f"⚠️ Permission denied accessing directory: {directory}")
        except Exception as e:
            print(f"⚠️ Error scanning directory {directory}: {e}")

    scan_directory(folder)

    # Sort files by size (process smaller files first for better progress feedback)
    try:
        media_files.sort(key=lambda x: os.path.getsize(x))
    except Exception:
        pass  # If sorting fails, continue with unsorted list

    return media_files


def get_file_type(file_path):
    """Determine the type of media file"""
    file_ext = Path(file_path).suffix.lower()

    for file_type, extensions in SUPPORTED_EXTENSIONS.items():
        if file_ext in extensions:
            return file_type

    return "unknown"


def create_safe_filename(filename):
    """Create a safe filename by removing/replacing problematic characters"""
    # Replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    safe_filename = filename
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, "_")

    # Limit filename length (Windows has 255 char limit)
    if len(safe_filename) > 200:
        name, ext = os.path.splitext(safe_filename)
        safe_filename = name[: 200 - len(ext)] + ext

    return safe_filename


def move_file_to_category(file_path, categories, dest_root, copy_mode=True):
    """
    Move or copy file to appropriate category folder with improvements
    """
    if not categories:
        categories = ["uncategorized"]

    success_count = 0
    errors = []

    try:
        source_filename = os.path.basename(file_path)
        safe_filename = create_safe_filename(source_filename)

        for category in categories:
            try:
                # Create nested directory structure if category contains "/"
                if "/" in category:
                    # For categories like "animals/dogs", create nested structure
                    category_parts = category.split("/")
                    dest_dir = os.path.join(dest_root, *category_parts)
                else:
                    dest_dir = os.path.join(dest_root, category)

                # Create destination directory
                os.makedirs(dest_dir, exist_ok=True)

                dest_path = os.path.join(dest_dir, safe_filename)

                # Handle filename conflicts
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(safe_filename)
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1

                    # Prevent infinite loop
                    if counter > 1000:
                        dest_path = os.path.join(
                            dest_dir, f"{name}_{int(time.time())}{ext}"
                        )
                        break

                # Copy or move the file
                if copy_mode:
                    shutil.copy2(file_path, dest_path)  # copy2 preserves metadata
                else:
                    shutil.move(file_path, dest_path)

                success_count += 1

                # Only print for first successful copy to avoid spam
                if success_count == 1:
                    operation = "Copied" if copy_mode else "Moved"
                    print(f"📁 {operation}: {source_filename} → {category}")

            except Exception as e:
                error_msg = f"Failed to process category '{category}': {str(e)}"
                errors.append(error_msg)
                continue

    except Exception as e:
        errors.append(f"General error processing {file_path}: {str(e)}")

    if errors and success_count == 0:
        # If all categories failed, try to put in uncategorized
        try:
            uncategorized_dir = os.path.join(dest_root, "uncategorized", "errors")
            os.makedirs(uncategorized_dir, exist_ok=True)

            safe_filename = create_safe_filename(os.path.basename(file_path))
            dest_path = os.path.join(uncategorized_dir, safe_filename)

            # Handle conflicts
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(safe_filename)
                dest_path = os.path.join(uncategorized_dir, f"{name}_{counter}{ext}")
                counter += 1

            if copy_mode:
                shutil.copy2(file_path, dest_path)
            else:
                shutil.move(file_path, dest_path)

            print(f"📁 Moved to errors folder: {os.path.basename(file_path)}")

        except Exception as fallback_error:
            print(f"❌ Critical error - could not save file anywhere: {fallback_error}")

    return success_count > 0, errors


def get_directory_size(directory):
    """Calculate total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    continue
    except Exception:
        pass
    return total_size


def cleanup_empty_directories(root_directory):
    """Remove empty directories after processing"""
    try:
        for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):
            # Skip root directory
            if dirpath == root_directory:
                continue

            try:
                # Try to remove directory if it's empty
                if not dirnames and not filenames:
                    os.rmdir(dirpath)
                    print(f"🗑️ Removed empty directory: {os.path.basename(dirpath)}")
            except OSError:
                # Directory not empty or permission denied
                continue
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")


def verify_file_integrity(source_path, dest_path):
    """Verify that copied file matches source (basic size check)"""
    try:
        source_size = os.path.getsize(source_path)
        dest_size = os.path.getsize(dest_path)
        return source_size == dest_size
    except Exception:
        return False
