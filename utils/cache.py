import json
import os
import time
import threading
from pathlib import Path
import hashlib

CACHE_FILE = "media_cache.json"
BACKUP_CACHE_FILE = "media_cache_backup.json"
CACHE_VERSION = "2.0"


class MediaCache:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.backup_file = BACKUP_CACHE_FILE
        self.cache_data = {}
        self.metadata = {
            "version": CACHE_VERSION,
            "created": time.time(),
            "last_modified": time.time(),
            "total_files": 0,
            "processing_stats": {"successful": 0, "errors": 0, "skipped": 0},
        }
        self._lock = threading.Lock()
        self.unsaved_changes = False
        self.load_cache()

    def _get_file_signature(self, file_path):
        """Create a unique signature for a file based on path, size, and mtime"""
        try:
            stat = os.stat(file_path)
            signature_string = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(signature_string.encode()).hexdigest()
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()

    def load_cache(self):
        """Load cache with error recovery and migration support"""
        cache_loaded = False

        # Try to load main cache file
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle old cache format (direct dict) vs new format (with metadata)
                if isinstance(data, dict) and "metadata" in data and "cache" in data:
                    # New format
                    self.cache_data = data.get("cache", {})
                    self.metadata.update(data.get("metadata", {}))
                    cache_loaded = True
                elif isinstance(data, dict):
                    # Old format - migrate
                    print("🔄 Migrating old cache format...")
                    self.cache_data = data
                    self.metadata["total_files"] = len(data)
                    cache_loaded = True
                    self.unsaved_changes = True  # Save in new format

            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                print(f"⚠️ Error loading cache: {e}")

        # Try backup if main cache failed
        if not cache_loaded and os.path.exists(self.backup_file):
            try:
                print("🔄 Attempting to restore from backup...")
                with open(self.backup_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "cache" in data:
                        self.cache_data = data["cache"]
                        self.metadata.update(data.get("metadata", {}))
                        cache_loaded = True
                        print("✅ Cache restored from backup")
                        self.unsaved_changes = True
            except Exception as e:
                print(f"⚠️ Backup restore failed: {e}")

        if not cache_loaded:
            print("📝 Starting with empty cache")
            self.cache_data = {}
        else:
            print(f"📖 Loaded cache with {len(self.cache_data)} entries")

        # Clean up invalid entries
        self._cleanup_invalid_entries()

    def _cleanup_invalid_entries(self):
        """Remove entries for files that no longer exist or have changed"""
        if not self.cache_data:
            return

        print("🧹 Cleaning up cache entries...")
        initial_count = len(self.cache_data)
        valid_entries = {}

        for file_path, entry in self.cache_data.items():
            try:
                if os.path.exists(file_path):
                    # Check if file has changed since caching
                    current_signature = self._get_file_signature(file_path)
                    cached_signature = (
                        entry.get("signature") if isinstance(entry, dict) else None
                    )

                    if cached_signature == current_signature:
                        valid_entries[file_path] = entry
                    elif cached_signature is None:
                        # Old cache entry without signature - keep for now but mark for update
                        if isinstance(entry, list):
                            valid_entries[file_path] = {
                                "categories": entry,
                                "timestamp": time.time(),
                                "signature": current_signature,
                            }
                        else:
                            valid_entries[file_path] = entry
                            valid_entries[file_path]["signature"] = current_signature
                else:
                    # File doesn't exist anymore
                    continue
            except Exception:
                # Skip problematic entries
                continue

        self.cache_data = valid_entries
        cleaned_count = initial_count - len(valid_entries)

        if cleaned_count > 0:
            print(f"🗑️ Removed {cleaned_count} stale cache entries")
            self.unsaved_changes = True
            self.metadata["last_modified"] = time.time()

    def get(self, file_path):
        """Get cached result for a file"""
        with self._lock:
            entry = self.cache_data.get(file_path)
            if entry is None:
                return None

            # Handle both old format (list) and new format (dict)
            if isinstance(entry, list):
                return entry
            elif isinstance(entry, dict) and "categories" in entry:
                return entry["categories"]

            return None

    def set(self, file_path, categories, processing_time=None):
        """Cache result for a file"""
        with self._lock:
            # Create comprehensive cache entry
            cache_entry = {
                "categories": categories,
                "timestamp": time.time(),
                "signature": self._get_file_signature(file_path),
                "processing_time": processing_time,
            }

            self.cache_data[file_path] = cache_entry
            self.unsaved_changes = True
            self.metadata["last_modified"] = time.time()
            self.metadata["total_files"] = len(self.cache_data)

            # Update processing stats
            if any("error" in str(cat).lower() for cat in categories):
                self.metadata["processing_stats"]["errors"] += 1
            else:
                self.metadata["processing_stats"]["successful"] += 1

    def contains(self, file_path):
        """Check if file is in cache"""
        with self._lock:
            return file_path in self.cache_data

    def save_cache(self, force=False):
        """Save cache to disk with atomic write and backup"""
        if not self.unsaved_changes and not force:
            return True

        with self._lock:
            try:
                # Prepare data structure
                cache_structure = {"metadata": self.metadata, "cache": self.cache_data}

                # Create backup of existing cache
                if os.path.exists(self.cache_file):
                    try:
                        if os.path.exists(self.backup_file):
                            os.remove(self.backup_file)
                        os.rename(self.cache_file, self.backup_file)
                    except Exception:
                        pass  # Backup creation is not critical

                # Atomic write using temporary file
                temp_file = f"{self.cache_file}.tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache_structure, f, indent=2, ensure_ascii=False)

                # Replace original file
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                os.rename(temp_file, self.cache_file)

                self.unsaved_changes = False
                return True

            except Exception as e:
                print(f"❌ Error saving cache: {e}")
                # Clean up temp file if it exists
                temp_file = f"{self.cache_file}.tmp"
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                return False

    def get_stats(self):
        """Get cache statistics"""
        with self._lock:
            stats = {
                "total_entries": len(self.cache_data),
                "cache_file_size": os.path.getsize(self.cache_file)
                if os.path.exists(self.cache_file)
                else 0,
                "created": self.metadata.get("created", 0),
                "last_modified": self.metadata.get("last_modified", 0),
                "processing_stats": self.metadata.get("processing_stats", {}),
            }

            # Calculate category distribution
            category_counts = {}
            for entry in self.cache_data.values():
                categories = (
                    entry.get("categories", []) if isinstance(entry, dict) else entry
                )
                if isinstance(categories, list):
                    for cat in categories:
                        main_cat = cat.split("/")[0] if "/" in cat else cat
                        category_counts[main_cat] = category_counts.get(main_cat, 0) + 1

            stats["category_distribution"] = category_counts
            return stats

    def clear(self):
        """Clear all cache data"""
        with self._lock:
            self.cache_data = {}
            self.metadata["total_files"] = 0
            self.metadata["processing_stats"] = {
                "successful": 0,
                "errors": 0,
                "skipped": 0,
            }
            self.metadata["last_modified"] = time.time()
            self.unsaved_changes = True

    def remove_entry(self, file_path):
        """Remove a specific entry from cache"""
        with self._lock:
            if file_path in self.cache_data:
                del self.cache_data[file_path]
                self.metadata["total_files"] = len(self.cache_data)
                self.metadata["last_modified"] = time.time()
                self.unsaved_changes = True
                return True
            return False

    def __len__(self):
        return len(self.cache_data)

    def __contains__(self, file_path):
        return self.contains(file_path)


# Global cache instance
_cache_instance = None


def get_cache_instance():
    """Get or create global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MediaCache()
    return _cache_instance


def load_cache():
    """Legacy function for compatibility"""
    cache = get_cache_instance()
    return cache.cache_data


def save_cache(cache_data=None):
    """Legacy function for compatibility"""
    cache = get_cache_instance()
    if cache_data is not None:
        # Update cache with provided data (legacy support)
        cache.cache_data.update(cache_data)
        cache.unsaved_changes = True
    return cache.save_cache()


def clear_cache():
    """Clear all cached data"""
    cache = get_cache_instance()
    cache.clear()
    cache.save_cache(force=True)


def get_cache_stats():
    """Get comprehensive cache statistics"""
    cache = get_cache_instance()
    return cache.get_stats()
