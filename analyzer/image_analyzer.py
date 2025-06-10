# analyzer/image_analyzer.py

import warnings
import re
import io
import os
import sys
import gc
import torch
import logging
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import easyocr

from utils.label_mapper import CATEGORIES, PRIORITY_KEYWORDS, MEME_INDICATOR_SET

# Suppress the specific PyTorch TypedStorage warning
warnings.filterwarnings(
    "ignore",
    message=r".*TypedStorage is deprecated.*",
    category=UserWarning,
    module=r"torch\._utils",
)

# ─── Logging setup ─────────────────────────────────────────────────────────────

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

file_h = logging.FileHandler("log.txt", mode="w", encoding="utf-8")
file_h.setLevel(logging.INFO)
file_h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

console_stream = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
console_h = logging.StreamHandler(console_stream)
console_h.setLevel(logging.WARNING)
console_h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logging.root.setLevel(logging.INFO)
logging.root.addHandler(file_h)
logging.root.addHandler(console_h)


# ─── Global Model Variables ────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None  # CLIPProcessor
model = None  # CLIPModel
classifier_tokenizer = None
classifier_model = None
text_classification_pipeline = None
caption_processor = None  # BLIP processor
caption_model = None  # BLIP model

# Initialize EasyOCR reader for text extraction
reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"), verbose=False)
if DEVICE == "cuda":
    logging.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("CUDA not available. Using CPU; performance may be slower.")

# Pretrained model names
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"
CLASSIFIER_MODEL_PATH = "finiteautomata/bertweet-base-sentiment-analysis"
BLIP_MODEL_PATH = "Salesforce/blip-image-captioning-base"


# ─── MODEL LOADING & CLEANUP ──────────────────────────────────────────────────


def load_clip_models():
    """Load CLIP processor + model into memory (once)."""
    global processor, model
    if model is None or processor is None:
        try:
            logging.info(f"Loading CLIP model '{CLIP_MODEL_PATH}' → {DEVICE}")
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
            model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
            logging.info("CLIP model loaded.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            processor = None
            model = None


def load_classifier_models():
    """Load sentiment‐analysis pipeline (optional) for meme detection."""
    global classifier_tokenizer, classifier_model, text_classification_pipeline
    if classifier_model is None:
        try:
            logging.info(
                f"Loading text classifier '{CLASSIFIER_MODEL_PATH}' → {DEVICE}"
            )
            classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
            classifier_model = AutoModelForSequenceClassification.from_pretrained(
                CLASSIFIER_MODEL_PATH
            ).to(DEVICE)
            text_classification_pipeline = pipeline(
                "sentiment-analysis",
                model=classifier_model,
                tokenizer=classifier_tokenizer,
                device=0 if DEVICE == "cuda" else -1,
            )
            logging.info("Text classification model loaded.")
        except Exception as e:
            logging.warning(f"Failed to load text classification model: {e}")
            classifier_tokenizer = None
            classifier_model = None
            text_classification_pipeline = None


def load_blip_models():
    """Load BLIP processor + model for image captioning."""
    global caption_processor, caption_model
    if caption_model is None or caption_processor is None:
        try:
            logging.info(f"Loading BLIP model '{BLIP_MODEL_PATH}' → {DEVICE}")
            caption_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
            caption_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_PATH
            ).to(DEVICE)
            logging.info("BLIP model loaded.")
        except Exception as e:
            logging.error(f"Failed to load BLIP model: {e}")
            caption_processor = None
            caption_model = None


def cleanup_models(model_type=None):
    """
    Aggressively unload models from VRAM.
    model_type may be 'blip', 'clip', 'classifier', or None for all.
    """
    global processor, model
    global classifier_tokenizer, classifier_model, text_classification_pipeline
    global caption_processor, caption_model

    logging.info(f"Initiating model cleanup for {model_type or 'all'} → free VRAM")

    if model_type in (None, "blip"):
        if caption_processor:
            del caption_processor
            caption_processor = None
        if caption_model:
            del caption_model
            caption_model = None
        logging.info("BLIP models unloaded.")

    if model_type in (None, "clip"):
        if processor:
            del processor
            processor = None
        if model:
            del model
            model = None
        logging.info("CLIP models unloaded.")

    if model_type in (None, "classifier"):
        if classifier_tokenizer:
            del classifier_tokenizer
            classifier_tokenizer = None
        if classifier_model:
            del classifier_model
            classifier_model = None
        if text_classification_pipeline:
            del text_classification_pipeline
            text_classification_pipeline = None
        logging.info("Text classification models unloaded.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    logging.info("VRAM cleared.")


# ─── OCR & MEME DETECTION HELPERS ───────────────────────────────────────────────


def extract_text_easyocr_with_boxes(pil_img: Image.Image):
    """
    1) Upscale ×2
    2) EasyOCR readtext(detail=1)
    3) Count narrow vs wide text boxes
    4) Return (plain_text, is_meme_flag)
       - is_meme_flag = True if ≥ 2 wide boxes  (→ FORCE MEME)
       - We no longer auto‐flag “documents” here.
    """
    w, h = pil_img.size
    up = pil_img.resize((w * 2, h * 2), Image.BILINEAR)
    img_np = np.array(up)

    results = reader.readtext(img_np, detail=1, paragraph=False)
    filtered = [(bbox, txt, conf) for bbox, txt, conf in results if conf > 0.3]

    big_boxes = 0
    for bbox, txt, conf in filtered:
        xs = [pt[0] for pt in bbox]
        box_frac = (max(xs) - min(xs)) / w
        if box_frac > 0.5:
            big_boxes += 1

    is_meme_flag = big_boxes >= 2
    plain = " ".join(txt for _, txt, _ in filtered).strip()
    return plain, is_meme_flag


def detect_meme_indicators(caption, text_content=""):
    """
    Compute a “meme_score” based on:
      - presence of known meme keywords,
      - years (4-digit numbers),
      - short caption + very long word,
      - all‐caps tokens,
      - (optionally) strong sentiment from a text classifier.

    **We only force [“memes”] if meme_score >= 1.5.**
    This higher threshold prevents pure documents (with a single year or one all‐caps)
    from accidentally becoming memes.
    """
    meme_score = 0
    combined = (caption + " " + text_content).lower()

    # 1) MEME_OFFSET KEYWORDS
    for indicator in MEME_INDICATOR_SET:
        if indicator in combined:
            meme_score += 1.0

    # 2) YEAR‐LIKE NUMBERS → +0.5
    if re.search(r"\d{4}", combined):
        meme_score += 0.5

    # 3) SHORT CAPTION + VERY LONG WORD → +0.5
    words = caption.split()
    if len(words) < 10 and any(len(w) > 10 for w in words):
        meme_score += 0.5

    # 4) ALL‐CAP TOKEN → +0.5
    if re.search(r"\b[A-Z]{3,}\b", combined):
        meme_score += 0.5

    # 5) OPTIONAL SENTIMENT BOOST
    load_classifier_models()
    if text_classification_pipeline:
        try:
            results = text_classification_pipeline(caption)
            if results:
                label = results[0]["label"]
                score = results[0]["score"]
                # Strong positive → +1.2
                if label == "LABEL_1" and score > 0.8:
                    meme_score += 1.2
                # Strong negative/neutral → +0.7
                elif label in ("LABEL_0", "LABEL_2") and score > 0.85:
                    meme_score += 0.7
        except Exception as e:
            logging.debug(f"Sentiment analysis failed: {e}")
        finally:
            cleanup_models("classifier")

    return meme_score


# ─── CLIP SCORING HELPER ────────────────────────────────────────────────────────


def get_clip_scores(image, candidates):
    """
    Run a single CLIP inference on `image` vs. a list of textual `candidates`.
    Returns a dict {candidate_text: score}.
    """
    if not (model and processor):
        logging.error("CLIP models not loaded; returning empty scores.")
        return {}

    inputs = processor(
        images=image, text=candidates, return_tensors="pt", padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return {cand: probs[0, idx].item() for idx, cand in enumerate(candidates)}


# ─── Main Categorization Function ─────────────────────────────────────────────


def categorize_optimized(image, caption, text_content=""):
    """
    1. (OCR stage already handled “wide text → [‘memes’]” upstream.)
    2. Run detect_meme_indicators(...) → if score >= 1.5 → ["memes"]
    3. PET override → ["pets"]
    4. PEOPLE override → ["people"]
    5. MACHINE/LAUNDRY override → ["objects"]
    6. PRIORITY KEYWORD (sightseeing) → ["sightseeing"]
    7. Caption‐based meme check (cartoon + text) → ["memes"]
    8. GENERAL CLIP SIMILARITY → pick highest‐score category (including “documents”)
    9. Tie‐break: if best is “achievements” vs “people,” apply margin logic
    """
    try:
        logging.info("Analyzing image (from PIL object)")
        cap = caption.lower()

        # ─── 2) DETECT MEME VIA KEYWORD/HEURISTIC (STRONG ENOUGH?) ───────────────
        meme_score = detect_meme_indicators(cap, text_content)
        if meme_score >= 1.5:
            logging.info(f"Meme indicator score {meme_score:.2f} → forcing 'memes'")
            return ["memes"], meme_score
        # ───────────────────────────────────────────────────────────────────────────

        # ─── 3) PET OVERRIDE (generic) ─────────────────────────────────────────────
        pet_terms = PRIORITY_KEYWORDS.get("pets", [])
        for pet_kw in pet_terms:
            if re.search(rf"\b{re.escape(pet_kw.lower())}\b", cap):
                logging.info(f"Pet keyword '{pet_kw}' found → forcing 'pets'")
                return ["pets"], 1.0
        # ───────────────────────────────────────────────────────────────────────────

        # ─── 4) PEOPLE OVERRIDE ────────────────────────────────────────────────────
        child_terms = [
            r"\bboy\b",
            r"\bgirl\b",
            r"\bchild\b",
            r"\btoddler\b",
            r"\bbaby\b",
            r"\bperson\b",
            r"\bman\b",
            r"\bwoman\b",
        ]
        for pattern in child_terms:
            if re.search(pattern, cap):
                logging.info(
                    f"Child/person keyword '{pattern}' found → forcing 'people'"
                )
                return ["people"], 1.0
        # ───────────────────────────────────────────────────────────────────────────

        # ─── 5) MACHINE/LAUNDRY OVERRIDE → "objects" ───────────────────────────────
        machine_terms = [
            r"\bwashing\b",
            r"\blower\b",
            r"\bdryer\b",
            r"\bwasher\b",
            r"\blaundry\b",
            r"\bclothes\b",
            r"\bair[- ]?cooler\b",
            r"\bfan\b",
        ]
        for pattern in machine_terms:
            if re.search(pattern, cap):
                logging.info(
                    f"Machine/laundry term '{pattern}' found → forcing 'objects'"
                )
                return ["objects"], 1.0
        # ───────────────────────────────────────────────────────────────────────────

        # ─── 6) PRIORITY KEYWORD MATCHING → possibly "sightseeing" ────────────────
        people_terms = PRIORITY_KEYWORDS.get("people", [])
        sight_words = [
            "arch",
            "tower",
            "beach",
            "park",
            "bench",
            "building",
            "monument",
            "shrine",
            "landmark",
            "view",
            "tourist",
            "scenic",
        ]
        if any(re.search(rf"\b{re.escape(p)}\b", cap) for p in people_terms) and any(
            re.search(rf"\b{re.escape(s)}\b", cap) for s in sight_words
        ):
            logging.info(
                "Caption contains people‐term + sight‐word → forcing 'sightseeing'"
            )
            try:
                sight_scores = get_clip_scores(image, CATEGORIES["sightseeing"])
                conf = max(sight_scores.values()) if sight_scores else 1.0
            except Exception:
                conf = 1.0
            return ["sightseeing"], conf

        combined_text = (caption + " " + text_content).lower()
        logging.debug(f"Combined text for priority check: {combined_text!r}")

        matched_categories = {}
        for category, keywords in PRIORITY_KEYWORDS.items():
            hits = []
            for kw in keywords:
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, combined_text):
                    hits.append(kw)
            if hits:
                matched_categories[category] = hits
                logging.info(f"Found priority keywords for '{category}': {hits}")

        PRIORITY_THRESHOLD = 0.40
        best_cat, best_score = None, 0.0
        for category, _ in matched_categories.items():
            labels = CATEGORIES.get(category, [])
            clip_scores = (
                get_clip_scores(image, labels) if (model and processor) else {}
            )
            if clip_scores:
                max_score = max(clip_scores.values())
                logging.info(f"CLIP score for '{category}': {max_score:.4f}")
                if max_score > best_score:
                    best_cat, best_score = category, max_score

        if best_cat and best_score >= PRIORITY_THRESHOLD:
            logging.info(
                f"Priority category '{best_cat}' confirmed (score: {best_score:.4f})"
            )
            return [best_cat], best_score

        # ─── 7) CAPTION‐BASED MEME DETECTION → "memes" ─────────────────────────────
        if "cartoon" in cap and text_content:
            logging.info("Cartoon + text detected → forcing 'memes'")
            return ["memes"], 1.0

        # ─── 8) GENERAL CLIP SIMILARITY (no threshold) ────────────────────────────
        category_scores = {}
        all_category_labels = []
        for cat, labels in CATEGORIES.items():
            all_category_labels.extend(labels)
        all_category_labels = list(set(all_category_labels))

        if not all_category_labels:
            logging.error("No category labels defined. Returning uncategorized.")
            return ["uncategorized/no-labels-defined"], 0.0

        if model is None or processor is None:
            logging.error("CLIP models not available. Returning uncategorized.")
            return ["uncategorized/no-clip-model"], 0.0

        all_labels_clip_scores = get_clip_scores(image, all_category_labels)

        for cat, labels in CATEGORIES.items():
            avg_score = 0.0
            count = 0
            for lbl in labels:
                if lbl in all_labels_clip_scores:
                    avg_score += all_labels_clip_scores[lbl]
                    count += 1
            category_scores[cat] = (avg_score / count) if count > 0 else 0.0

        # Apply caption‐based boost (slightly raised)
        CAPTION_BOOST_AMOUNT = 0.09
        for cat, keywords in PRIORITY_KEYWORDS.items():
            if cat != "achievements":
                for kw in keywords:
                    if kw in combined_text and cat in category_scores:
                        orig = category_scores[cat]
                        boosted = min(orig + CAPTION_BOOST_AMOUNT, 1.0)
                        category_scores[cat] = boosted
                        logging.info(
                            f"Boosting '{cat}' score due to keyword '{kw}'. "
                            f"Orig: {orig:.4f}, Boosted: {boosted:.4f}"
                        )
                        break

        if not category_scores:
            return ["uncategorized/no-scores"], 0.0

        # ─── 9) PICK BEST CATEGORY & TIE‐BREAK ───────────────────────────────────
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]

        # Special tie‐break: Achievements vs People
        if best_category == "achievements" and "people" in category_scores:
            people_score = category_scores["people"]
            if people_score > best_score + 0.07 and people_score >= 0.26:
                logging.info(
                    f"Switching from achievements ({best_score:.4f}) to people ({people_score:.4f}) "
                    "due to clear people dominance"
                )
                best_category = "people"
                best_score = people_score
            elif best_score >= 0.42 and people_score < 0.21:
                logging.info(
                    f"Keeping achievements ({best_score:.4f}) as it's strong "
                    f"and people score ({people_score:.4f}) is low."
                )

        # ─── Final Debug Output ─────────────────────────────────────────────────
        image_log_name = (
            image.filename if hasattr(image, "filename") else "PIL_Image_Object"
        )
        logging.info(f"--- Debugging Scores for {image_log_name} ---")
        logging.info(f"Generated Caption: {caption}")
        logging.info(f"Extracted Text (OCR): {text_content or 'No text extracted'}")
        logging.info(f"Meme Indicator Score: {meme_score:.2f}")
        logging.info("Raw Category CLIP Scores:")
        for cat, score in category_scores.items():
            logging.info(f"    {cat}: {score:.4f}")
        logging.info(
            f"Best Category Candidate: {best_category} (Score: {best_score:.4f})"
        )
        logging.info(f"Selected category '{best_category}' with score {best_score:.4f}")
        logging.info("--- End Debugging Scores ---")

        return [best_category], best_score

    except Exception as e:
        logging.error(f"Error during image categorization: {e}", exc_info=True)
        logging.info("--- End Debugging Scores (Error) ---")
        return ["uncategorized/analysis-error"], 0.0


def analyze_image_file(image_path):
    """
    1. Load image via PIL.
    2. Run BLIP captioning → caption text.
    3. OCR via EasyOCR to detect ≥2 wide text boxes → if so, force ["memes"].
    4. Else call categorize_optimized(...) → return [category].
    5. Cleanup all models (BLIP, CLIP, classifier).
    """
    logging.info(f"Starting analysis for image: {os.path.basename(image_path)}")
    caption = ""
    text_content = ""
    image = None

    try:
        image = Image.open(image_path).convert("RGB")

        # 1) BLIP captioning
        load_blip_models()
        if caption_model and caption_processor:
            try:
                inputs = caption_processor(image, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = caption_model.generate(**inputs, max_new_tokens=30)
                caption = caption_processor.decode(out[0], skip_special_tokens=True)
                logging.info(f"Generated Caption: {caption}")
            except Exception as e:
                logging.error(f"BLIP captioning failed: {e}", exc_info=True)
                caption = ""
        else:
            logging.error("BLIP models not loaded; caption set to empty.")
            caption = ""
    except Exception as e:
        logging.error(
            f"Error opening image {os.path.basename(image_path)}: {e}", exc_info=True
        )
        caption = ""
    finally:
        cleanup_models("blip")  # free BLIP VRAM

    try:
        load_clip_models()

        # 2) OCR—detect if ≥2 wide text boxes → force ["memes"]
        text_content, is_meme_flag = extract_text_easyocr_with_boxes(image)
        if text_content:
            logging.info(f"Extracted Text (OCR): {text_content[:100]}...")
        if is_meme_flag:
            logging.info("OCR flagged ≥2 wide text boxes → returning ['memes']")
            return ["memes"]

        # 3) Otherwise call categorize_optimized(...) for everything else
        categories, confidence = categorize_optimized(image, caption, text_content)
        if not categories:
            categories = ["uncategorized"]

        logging.info(
            f"Final categories for {os.path.basename(image_path)}: {categories} (conf: {confidence:.4f})"
        )
        return categories

    except Exception as e:
        logging.error(
            f"Failed to analyze image {os.path.basename(image_path)}: {e}",
            exc_info=True,
        )
        return ["uncategorized/error-analysis-failed"]
    finally:
        cleanup_models()  # free CLIP & classifier VRAM
