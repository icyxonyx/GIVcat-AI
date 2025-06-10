# GIVcat-AI

A Python-based tool to automatically categorize and organize media files (GIFs, images, videos) into structured output directories using AI-driven content analysis.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [Configuration & Customization](#configuration--customization)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Project Overview

GIVcat-AI automatically scans a source folder of media files, analyzes each file’s content via deep learning (e.g., CLIP-based classifiers, optional OCR), and moves or copies them into category-based subdirectories in an output folder. It uses caching to avoid reprocessing, optimized memory handling (including GPU support via PyTorch), and provides detailed logging of successes and failures .

## Key Features

* **Recursive Scanning**: Collects all supported media files under a given source directory.
* **AI-driven Categorization**: Separate analyzers for images, videos (frame sampling + aggregation), and GIFs.
* **Caching Mechanism**: Skips files already processed, with signature-based invalidation and automatic backup.
* **Automatic Organization**: Moves or copies files into category folders under the output directory.
* **Memory & GPU Optimization**: Batch processing, aggressive cleanup, optional GPU acceleration via PyTorch.
* **Configurable & Extensible**: Command-line or code-based configuration; Windows convenience script (`run.bat`).
* **Detailed Logging**: Console logs with clear prefixes; optional file-based logging for deeper inspection.

## Technologies Used

* **Python 3.10+**&#x20;
* **PyTorch 2.1.0 (CUDA 11.8)** for model inference & GPU acceleration
* **Transformers 4.48.0 / HuggingFace Hub** for CLIP, BLIP models
* **TQDM, NumPy, PIL/Pillow** for progress bars, numerical ops, image I/O
* **MoviePy, ImageIO, OpenCV** for video & GIF handling
* **EasyOCR** (optional) for OCR fallback in document-like images
* **psutil, regex, ftfy** for system monitoring & text normalization
* **Accelerate** for memory optimization
* **Windows Batch Script** (`run.bat`) for easy setup on Windows

## Project Structure

```plaintext
media-sorter/
├── analyzer/
│   ├── __init__.py
│   ├── gif_analyzer.py
│   ├── image_analyzer.py
│   └── video_analyzer.py
├── data/
│   └── <your-source-media-files-here>
├── output/
│   └── <automatically-generated-categories-and-files>
├── utils/
│   ├── __init__.py
│   ├── cache.py
│   ├── file_utils.py
│   └── label_mapper.py
├── main.py
├── requirements.txt
└── run.bat
```

* **`analyzer/`**: Logic for determining categories per file type (images, videos, GIFs) .
* **`utils/`**: Supportive utilities:

  * `cache.py`: Persistent JSON-backed cache with signature checks, backup, metadata.
  * `file_utils.py`: File scanning, safe naming, moving/copying into category folders, integrity checks.
  * `label_mapper.py`: Category definitions and keyword lists guiding CLIP similarity scoring.
* **Root files**:

  * `main.py`: Orchestrates scanning, analyzing, caching, moving, memory cleanup, final summary.
  * `requirements.txt`: Pinned dependencies for reproducibility.
  * `run.bat`: Windows convenience script for environment setup & execution.

## Installation & Setup

1. **Clone or Download the Project**

   ```bash
   git clone https://github.com/icyxonyx/GIVcat-AI.git
   cd media-sorter
   ```



2. **Create & Activate Virtual Environment**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   python -m pip install --upgrade pip
   python -m pip install --upgrade pip wheel
   python -m pip install -r requirements.txt --use-pep517
   ```

4. **Prepare Data Folder**

   * Create a folder named `data/` in project root.
   * Place your media files inside (`.jpg`, `.png`, `.mp4`, `.avi`, `.gif`, etc.).

5. **Run the Tool**

   ```bash
   python main.py
   ```

   * On Windows, you may alternatively run `run.bat`, which will check prerequisites, set up venv if needed, and launch the script.&#x20;

6. **View Output**

   * Processed files (copied/moved) appear under `output/` in subdirectories named by detected category.
   * Console prints a summary: total files, processed count, errors (if any).

## Usage

* By default, `main.py` scans from `data/` and outputs to `output/`.

* You can modify constants in `main.py`:

  * `SOURCE` and `DEST` paths.
  * `MAX_FILE_SIZE` to skip or treat large files differently.
  * `BATCH_SIZE` for number of files processed before cleanup.
  * `CACHE_SAVE_INTERVAL` for how often cache is persisted.&#x20;

* **Copy vs. Move**:

  * By default, files are copied (`copy_mode=True`).
  * To move instead, set `copy_mode=False` in `move_file_to_category` calls.&#x20;

* **Logging**:

  * Console logs use clear prefixes (📁, ⚠️, ❌).
  * For file-based logging or more sophisticated reports (JSON/CSV), integrate Python’s `logging` or custom code iterating over cache data.&#x20;

## Configuration & Customization

1. **Adjust Category Keywords**

   * Edit `utils/label_mapper.py`. Modify or add keywords under each category to influence CLIP-based classification.&#x20;

2. **Tuning File Size Thresholds**

   * In `main.py`, adjust `MAX_FILE_SIZE` (default 100 MB) or `BATCH_SIZE`.&#x20;

3. **Caching Behavior**

   * The cache file `media_cache.json` persists categorization results.
   * To disable caching temporarily, comment out the `if file_path in cache:` check in `process_file`.
   * To clear cache manually, delete `media_cache.json` and `media_cache_backup.json` or call `clear_cache()`.
   * For advanced backends, replace JSON logic in `utils/cache.py`.&#x20;

4. **OCR Support**

   * Uncomment `pytesseract` in `requirements.txt` and install Tesseract on the system.
   * Modify `image_analyzer.py` to use EasyOCR or pytesseract for text-based categorization (e.g., receipts/documents).&#x20;

5. **GPU & Performance**

   * Ensure CUDA-enabled PyTorch wheel installed (e.g., `torch==2.1.0+cu118`).
   * The analyzer scripts automatically detect GPU availability and use it if present.
   * Tweak memory cleanup intervals or frame sampling parameters in `video_analyzer.py` for different system specs.

6. **Run Script Customization**

   * `run.bat` can be adapted for other OS (e.g., create a shell script for Linux/macOS).
   * Add additional checks or environment variables as needed.

## Troubleshooting

* **“Source folder ‘data’ does not exist”**
  Ensure `data/` directory exists in project root and contains supported media files.&#x20;

* **“No media files found in ‘data’ folder”**
  Verify file extensions and subdirectories. In a Python REPL, test `get_media_files("data")` to see detected files.&#x20;

* **GPU-Related Errors**
  If `torch.cuda.is_available()` is False, confirm you installed the correct CUDA-enabled PyTorch version. Check GPU drivers and CUDA toolkit compatibility.&#x20;

* **Cache Corruption**
  If JSON decode errors occur on `media_cache.json`, delete `media_cache.json` and `media_cache_backup.json` then rerun.&#x20;

* **“Too-Large” Files**
  Files > `MAX_FILE_SIZE` (100 MB) are placed under `uncategorized/too-large`. Adjust in `main.py` or pre-process large files.&#x20;

* **Permission or I/O Errors**
  Ensure read access to `data/` and write access to `output/`. On Windows, close applications locking files.&#x20;

* **Unexpected Categories**
  The CLIP-based classifier depends on keyword similarity and model outputs. To refine: update keywords in `label_mapper.py`, adjust thresholds or sampling logic in analyzers.&#x20;

## License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it in your own applications.
