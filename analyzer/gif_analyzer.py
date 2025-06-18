# analyzer/gif_analyzer.py

import os
import gc
import time
import tempfile
import psutil
import imageio
from PIL import Image
import numpy as np
import torch
from collections import Counter

import analyzer.image_analyzer as ia  # refer to the module
from analyzer.image_analyzer import analyze_image_file, cleanup_models
from utils.label_mapper import MEME_INDICATOR_SET

# ─── Optimizations for GIFs ─────────────────────────────────────────────────────

MAX_GIF_SIZE_MB = 100  # If the GIF is larger than this, fallback to PIL method
MAX_FRAME_SIZE = (512, 512)  # Resize each frame to fit within this (width, height)
MIN_FRAMES_FOR_ANALYSIS = 2  # If memory is tight, analyze only this many frames
MAX_FRAMES_TO_ANALYZE = 6  # Never analyze more than this many frames


def check_gif_properties(gif_path):
    """
    Quickly probe a GIF’s file size and frame count without loading all frames.
    Returns a dict: { 'file_size_mb': float, 'frame_count': int, 'is_animated': bool }
    """
    try:
        file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        with Image.open(gif_path) as img:
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                pass

        return {
            "file_size_mb": file_size_mb,
            "frame_count": frame_count,
            "is_animated": (frame_count > 1),
        }
    except Exception as e:
        print(f"⚠️ Error checking GIF properties: {str(e)[:50]}")
        return {"file_size_mb": 0.0, "frame_count": 1, "is_animated": False}


def is_likely_meme_gif(gif_path, frame_count):
    """
    Heuristic to detect if a GIF is probably a meme/reaction, using:
      1) Substrings from MEME_INDICATOR_SET in the filename
      2) Short GIFs (2–50 frames) often are reaction/meme clips

    We require a total “meme_score” ≥ 3 to qualify as “likely meme.”
    """
    filename = os.path.basename(gif_path).lower()
    meme_score = 0

    # 1) Count how many MEME_INDICATOR_SET substrings appear in the filename
    for indicator in MEME_INDICATOR_SET:
        if indicator in filename:
            meme_score += 1

    # 2) Short looping GIFs add +1
    if 2 <= frame_count <= 50:
        meme_score += 1

    return meme_score >= 3


def extract_gif_frames_optimized(gif_path, num_frames, gif_properties):
    """
    Attempt to read up to `num_frames` frames via imageio, resizing each frame
    to fit within MAX_FRAME_SIZE, saving to a temp JPEG, and running analyze_image_file() on each.

    If ANY frame read throws:
      - ValueError about “all input arrays must have the same shape,”
      - OR an error containing "Cannot handle this data type",
    we immediately fall back to extract_gif_frames_pil().
    """
    temp_files = []
    predictions = []

    try:
        file_size_mb = gif_properties["file_size_mb"]
        frame_count = gif_properties["frame_count"]
        print(
            f"🎭 Extracting up to {num_frames} frames from GIF '{os.path.basename(gif_path)}' "
            f"({frame_count} frames, {file_size_mb:.1f}MB)"
        )

        # If the GIF is very large, skip imageio and go straight to the PIL fallback
        if file_size_mb > MAX_GIF_SIZE_MB:
            return extract_gif_frames_pil(gif_path, num_frames)

        try:
            gif_reader = imageio.get_reader(gif_path)
            total_frames = len(gif_reader)

            # Decide which frame indices to grab
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                if total_frames <= 10:
                    # Evenly spaced for short GIFs
                    frame_indices = np.linspace(
                        0, total_frames - 1, num_frames, dtype=int
                    )
                else:
                    # “Beginning, middle, end” strategy
                    frame_indices = []
                    # First two frames
                    frame_indices.extend(range(min(2, total_frames)))
                    # Two around the middle
                    mid_start = total_frames // 3
                    mid_end = 2 * total_frames // 3
                    frame_indices.extend(range(mid_start, min(mid_end, mid_start + 2)))
                    # Last two frames
                    frame_indices.extend(range(max(0, total_frames - 2), total_frames))
                    # Deduplicate & limit to num_frames
                    frame_indices = sorted(set(frame_indices))[:num_frames]

            # Extract each chosen frame
            for frame_idx in frame_indices:
                try:
                    if frame_idx >= total_frames:
                        continue

                    # Attempt to read via imageio
                    frame = gif_reader.get_data(frame_idx)

                    # Convert to PIL, resize if needed
                    pil_image = Image.fromarray(frame).convert("RGB")
                    if (
                        pil_image.size[0] > MAX_FRAME_SIZE[0]
                        or pil_image.size[1] > MAX_FRAME_SIZE[1]
                    ):
                        pil_image.thumbnail(MAX_FRAME_SIZE, Image.Resampling.LANCZOS)

                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False, dir=tempfile.gettempdir()
                    )
                    temp_file.close()
                    pil_image.save(temp_file.name, "JPEG", quality=85, optimize=True)
                    temp_files.append(temp_file.name)

                    # Analyze that frame (as a still image)
                    frame_preds = analyze_image_file(temp_file.name)
                    predictions.extend(frame_preds)

                    # Clean up this iteration
                    del frame, pil_image
                    gc.collect()
                    time.sleep(0.02)

                except ValueError as ve:
                    # “all input arrays must have the same shape”
                    print(f"⚠️ Imageio frame‐shape mismatch at frame {frame_idx}: {ve}")
                    gif_reader.close()
                    return extract_gif_frames_pil(gif_path, num_frames)

                except Exception as e:
                    msg = str(e)
                    if "Cannot handle this data type" in msg:
                        # PIL cannot handle this frame’s array dtype—fallback to PIL extractor
                        print(f"⚠️ PIL conversion error at frame {frame_idx}: {msg}")
                        gif_reader.close()
                        return extract_gif_frames_pil(gif_path, num_frames)
                    else:
                        # Some other frame‐specific error: log and continue
                        print(f"⚠️ Error processing GIF frame {frame_idx}: {msg[:50]}")
                        continue

            gif_reader.close()

        except Exception as e:
            # If imageio cannot open/read the GIF at all, fallback to PIL method
            print(f"⚠️ imageio.get_reader failed, falling back to PIL: {str(e)[:50]}")
            return extract_gif_frames_pil(gif_path, num_frames)

        return predictions

    except Exception as e:
        print(f"❌ Critical error during GIF frame extraction: {str(e)[:100]}")
        return ["uncategorized/extraction-error"]

    finally:
        # Delete any temporary JPEGs we created
        for tf in temp_files:
            try:
                if os.path.exists(tf):
                    os.unlink(tf)
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_gif_frames_pil(gif_path, num_frames):
    """
    PIL‐based fallback for GIFs that are extremely large or have shape mismatches.
    We composite each frame onto a full‐size RGBA background to avoid tile‐shape errors.
    We cap at reading 100 frames to avoid infinite loops, and pick up to `num_frames` frames to analyze.
    """
    temp_files = []
    predictions = []

    print("ℹ️ Using PIL fallback to extract frames")

    try:
        with Image.open(gif_path) as img:
            # Full GIF canvas size
            canvas_size = img.size  # (width, height)

            frame_count = 0
            frames_to_analyze = []

            try:
                while frame_count < 100:
                    img.seek(frame_count)

                    # Composite this frame onto a blank RGBA canvas
                    frame_rgba = img.convert("RGBA")
                    background = Image.new("RGBA", canvas_size)
                    background.paste(frame_rgba, (0, 0), frame_rgba)
                    frame_rgb = background.convert("RGB")

                    # Collect up to `num_frames` frames evenly spread
                    if len(frames_to_analyze) < num_frames:
                        if (
                            frame_count == 0
                            or frame_count % max(1, (100 // num_frames)) == 0
                        ):
                            frames_to_analyze.append(frame_rgb)

                    frame_count += 1

            except EOFError:
                pass

            # Process each collected frame
            for i, frame in enumerate(frames_to_analyze):
                try:
                    if (
                        frame.size[0] > MAX_FRAME_SIZE[0]
                        or frame.size[1] > MAX_FRAME_SIZE[1]
                    ):
                        frame.thumbnail(MAX_FRAME_SIZE, Image.Resampling.LANCZOS)

                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False, dir=tempfile.gettempdir()
                    )
                    temp_file.close()
                    frame.save(temp_file.name, "JPEG", quality=85)
                    temp_files.append(temp_file.name)

                    frame_preds = analyze_image_file(temp_file.name)
                    predictions.extend(frame_preds)

                    del frame
                    gc.collect()

                except Exception as e:
                    print(f"⚠️ Error processing PIL fallback frame {i}: {str(e)[:50]}")
                    continue

        return predictions

    except Exception as e:
        print(f"❌ PIL fallback failed: {str(e)[:100]}")
        return ["uncategorized/pil-error"]

    finally:
        for tf in temp_files:
            try:
                if os.path.exists(tf):
                    os.unlink(tf)
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze_gif(gif_path):
    """
    Main entry‐point for GIF analysis:

    1) Check file existence and size.
    2) If not animated (single frame), delegate to analyze_image_file().
    3) LOAD the first frame, run BLIP caption, and if that caption mentions
       “cartoon” or “animation,” immediately return ["memes"] (but only if BLIP loaded successfully).
    4) Decide how many frames to extract (based on frame count & memory usage).
    5) Run a filename + frame_count heuristic (using MEME_INDICATOR_SET) to detect
       “likely meme GIFs.” If so, and if any frame‐level category contains “meme,” return that.
    6) Otherwise, aggregate all frame‐level categories:
       - If top category > 60% of valid frames → return it.
       - If there’s a tie for first, return both.
       - Else, return the single most frequent category.
    7) Always call cleanup_models() + torch.cuda.empty_cache() at the end.
    """
    try:
        print(f"🎭 Analyzing GIF: {os.path.basename(gif_path)}")

        # 1) Ensure file exists
        if not os.path.exists(gif_path):
            return ["uncategorized/file-not-found"]

        gif_props = check_gif_properties(gif_path)

        # 2) Skip extremely large GIFs
        if gif_props["file_size_mb"] > 500:  # 500 MB limit
            print(f"⚠️ GIF too large ({gif_props['file_size_mb']:.1f} MB), skipping")
            return ["uncategorized/too-large"]

        # 3) If not animated, treat as a still image
        if not gif_props["is_animated"]:
            print("📸 Static GIF detected, routing to image analyzer")
            return analyze_image_file(gif_path)

        # ─── “CARTOON” OVERRIDE ────────────────────────────────────────────────
        # Load just the first frame via PIL, generate BLIP caption,
        # and if that caption mentions “cartoon” or “animation,” treat as ["memes"],
        # but only if BLIP models successfully loaded.
        try:
            ia.load_blip_models()
            if (ia.caption_processor is not None) and (ia.caption_model is not None):
                with Image.open(gif_path) as first_frame_img:
                    first_frame_img.seek(0)
                    frame_rgb = first_frame_img.convert("RGB")

                # Save this single frame to a temp JPEG so we can run BLIP on it
                temp_cartoon_file = tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False, dir=tempfile.gettempdir()
                )
                temp_cartoon_file.close()
                frame_rgb.save(temp_cartoon_file.name, "JPEG", quality=85)

                # Generate a BLIP caption on this frame
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = ia.caption_processor(frame_rgb, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = ia.caption_model.generate(**inputs, max_new_tokens=30)
                cartoon_caption = ia.caption_processor.decode(
                    out[0], skip_special_tokens=True
                ).lower()

                # Clean up BLIP models immediately after generating the caption
                cleanup_models("blip")

                # Delete the temporary JPEG now that we have the caption
                os.unlink(temp_cartoon_file.name)

                # If the caption mentions “cartoon” or “animation,” force “memes”
                if "cartoon" in cartoon_caption or "animation" in cartoon_caption:
                    print(
                        f"🎨 BLIP caption says \"{cartoon_caption}\" → forcing ['memes']"
                    )
                    return ["memes"]
            else:
                print(
                    "⚠️ BLIP models not available for cartoon override, skipping that check"
                )
        except Exception as e:
            print(f"⚠️ Could not run cartoon override BLIP check: {str(e)[:50]}")
        # ─────────────────────────────────────────────────────────────────────────────

        # 4) Decide how many frames to analyze based on memory & total frames
        mem = psutil.virtual_memory()
        if mem.percent > 85:
            print("⚠️ High RAM usage, analyzing minimal frames for GIF")
            num_frames = MIN_FRAMES_FOR_ANALYSIS
        else:
            total_frames = gif_props["frame_count"]
            if total_frames <= 5:
                num_frames = total_frames
            elif total_frames <= 20:
                num_frames = min(4, MAX_FRAMES_TO_ANALYZE)
            else:
                num_frames = MAX_FRAMES_TO_ANALYZE

        # 5) Filename + frame_count heuristic for “likely meme GIF”
        likely_meme = is_likely_meme_gif(gif_path, gif_props["frame_count"])

        # 6) Extract & analyze frames
        all_predictions = extract_gif_frames_optimized(gif_path, num_frames, gif_props)

        if not all_predictions:
            print("❌ No frame predictions generated, returning uncategorized")
            return ["uncategorized/no-analysis"]

        # 7) Filter out any “error” predictions
        valid_preds = [
            p
            for p in all_predictions
            if not any(
                err_kw in p.lower()
                for err_kw in ["error", "invalid", "too-small", "blank"]
            )
        ]
        if not valid_preds:
            # If all frames failed, return the single most common error
            err_count = Counter(all_predictions)
            return [err_count.most_common(1)[0][0]]

        # 8) If we flagged “likely meme,” see if any frame is already “meme.”
        if likely_meme:
            meme_preds = [p for p in valid_preds if "meme" in p.lower()]
            if meme_preds:
                print("🎭 GIF flagged as likely meme, prioritizing meme category")
                return [Counter(meme_preds).most_common(1)[0][0]]

        # 9) Otherwise, pick via simple majority/consensus
        freq = Counter(valid_preds)
        most_common = freq.most_common(3)
        total = len(valid_preds)

        # a) If top category > 60% of valid frames
        if most_common[0][1] > total * 0.6:
            result = [most_common[0][0]]
        # b) If tie for first place, return both top categories
        elif len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            result = [most_common[0][0], most_common[1][0]]
        else:
            # c) Otherwise, return the single most frequent
            result = [most_common[0][0]]

        print(f"✅ GIF categorized as: {result}")
        return result

    except Exception as e:
        print(
            f"❌ Critical error analyzing GIF {os.path.basename(gif_path)}: {str(e)[:100]}"
        )
        return ["uncategorized/critical-error"]

    finally:
        # 10) Final cleanup: unload any loaded models, free GPU RAM
        cleanup_models()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.05)
