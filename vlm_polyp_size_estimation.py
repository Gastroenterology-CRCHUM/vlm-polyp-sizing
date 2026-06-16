"""
VLM-based polyp size estimation and classification pipeline.

Runs a vision-language model (via Ollama) over cropped colonoscopy still
frames and asks it to (1) locate the polyp, (2) classify it into a clinical
size category, and (3) estimate its maximal diameter in mm, using a few-shot
prompting strategy (example images + reference category/diameter provided
in-context before each new query).

This matches the prompt and output format reported in the manuscript's
supplementary materials (Supplementary Materials 1s: Open-Source MLLM prompt).

Expected input layout:
    DATA_DIR/<record_id>/p<polyp_id>/best1/<image>.jpg|jpeg|png

Output:
    An Excel file with one row per image: record_id, polyp_id, parsed size
    category, parsed diameter estimate (mm), and the raw model response.
"""

import os
import glob
import traceback
import gc
import re

import pandas as pd
import ollama
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==== Configuration ====

# Root directory of this project (assumes script lives in the repo root,
# alongside the `data/` and `example_images/` folders below).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cropped polyp images, expected layout:
#   <DATA_DIR>/<record_id>/p<polyp_id>/best1/<image>.jpg|jpeg|png
DATA_DIR = os.path.join(PROJECT_DIR, "data", "cropped_5x4_1350x1080")

# Few-shot example images referenced in EXAMPLE_IMAGES below
EXAMPLES_DIR = os.path.join(PROJECT_DIR, "example_images")

# Output spreadsheet (final + periodic progress checkpoints)
OUT_XLSX = os.path.join(PROJECT_DIR, "results", "mllm_results.xlsx")

# This script was run once per evaluated VLM, changing MODEL each time:
#   Qwen3-VL:    "qwen3-vl:32b"
#   Llama4:      "llama4:scout"
#   Gemma3:      "gemma3:27b"
#   Mistral:     "mistral-small3.2:24b"
MODEL = "qwen3-vl:32b"  # <- change per run

# Prompt asks the model to both classify the polyp into a size category and
# give a continuous diameter estimate. Matches Supplementary Materials 1s.
PROMPT = """You are analyzing a colonoscopy image to estimate polyp size.

A polyp is an abnormal growth on the colon wall—typically a raised bump or dome-shaped lesion, often pink or reddish, distinct from surrounding mucosa.

Polyp size categories:
- Diminutive: 1–5 mm
- Small: 6–9 mm  
- Large: ≥10 mm

IMPORTANT:
There IS a polyp present in this image. Your task is to locate it and estimate its size.

- Carefully examine the entire image to identify the polyp
- Look for raised lesions, bumps, or dome-shaped structures
- The polyp may be small, subtle, or partially visible
- Only use NA if you genuinely cannot identify any polyp after thorough examination

If you locate the polyp:
- Estimate its maximal diameter in mm (give a single best estimate, not a range)
- Classify it into one of the three categories
- Adjust for scope distance (closer looks larger)

I will show you some example images first for reference, but you must analyze each NEW image independently based on its own visual features.

Output format (must follow exactly):
Category: [Diminutive/Small/Large/NA]
Estimated diameter: [number in mm or NA]
"""

# Deterministic decoding: temperature 0 + top_p 1 for reproducible outputs.
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 4096,
}

CLEANUP_INTERVAL = 10  # Run GC / clear GPU cache every N images
SAVE_INTERVAL = 5      # Write a progress checkpoint every N images

# Few-shot example images: (path, reference diameter in mm).
# These are shown to the model before each new query, as
# (user image -> assistant "Category: ...\nEstimated diameter: X") turns.
# Set EXAMPLE_IMAGES = None to disable few-shot examples (zero-shot mode).
EXAMPLE_IMAGES = [
    (os.path.join(EXAMPLES_DIR, "vid_01_5336_p1_t3.jpg"), 8),
    (os.path.join(EXAMPLES_DIR, "vid_01_5559_p1_t2.jpg"), 12),
    (os.path.join(EXAMPLES_DIR, "vid_01_5305_p1_t6.jpg"), 2),
]


# ==== Helper functions ====

def get_category(mm):
    """Convert a diameter in mm to its clinical size category
    (Diminutive: 1-5mm, Small: 6-9mm, Large: >=10mm). Returns None if mm is None.
    """
    if mm is None:
        return None
    if mm <= 5:
        return "Diminutive"
    elif mm <= 9:
        return "Small"
    else:
        return "Large"


def extract_ids(image_path, base_dir):
    """Extract (record_id, polyp_id) from a path of the form
    <base_dir>/<record_id>/p<polyp_id>/best1/<file>.
    Returns ("", "") if the path doesn't fit the expected structure.
    """
    image_path = os.path.normpath(image_path)
    base_dir = os.path.normpath(base_dir)

    try:
        rel_path = os.path.relpath(image_path, base_dir)
        parts = rel_path.split(os.sep)

        record_id = parts[0] if len(parts) > 0 else ""
        polyp_id = parts[1] if len(parts) > 1 else ""

        return record_id, polyp_id
    except ValueError:
        return "", ""


def find_images(base_dir):
    """Recursively collect all jpg/jpeg/png images under
    <base_dir>/*/p*/best1/, matching the expected dataset layout.
    """
    patterns = [
        os.path.join(base_dir, "*", "p*", "best1", "*.jpg"),
        os.path.join(base_dir, "*", "p*", "best1", "*.jpeg"),
        os.path.join(base_dir, "*", "p*", "best1", "*.png"),
    ]
    files = []
    for p in patterns:
        found = glob.glob(p, recursive=False)
        files.extend(found)
        print(f"Pattern: {p} -> Found {len(found)} files")
    return sorted(set(files))


def parse_response(response_text):
    """Parse the model's free-text response into (category, diameter_mm).

    Parsing strategy:
    1. Extract "Category: ..." (Diminutive/Small/Large), treating "NA" as None.
    2. Extract "Estimated diameter: ..." (NA -> None, otherwise a float).
    3. If neither field could be parsed from the expected format, fall back
       to scanning lines from the end for the last numeric token and
       deriving the category from it.
    """
    category = None
    diameter = None

    # 1) Category
    cat_match = re.search(r'Category:\s*(Diminutive|Small|Large|NA)', response_text, re.IGNORECASE)
    if cat_match:
        cat_value = cat_match.group(1)
        category = None if cat_value.upper() == 'NA' else cat_value.capitalize()

    # 2) Diameter
    if re.search(r'Estimated diameter:\s*NA', response_text, re.IGNORECASE):
        diameter = None
    else:
        diam_match = re.search(r'Estimated diameter:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        if diam_match:
            try:
                diameter = float(diam_match.group(1))
            except ValueError:
                pass

    # 3) Fallback: model didn't follow the expected format at all, try to
    # recover a number from the last line(s) containing digits and derive
    # the category from it.
    if diameter is None and category is None:
        lines = response_text.strip().split('\n')
        for line in reversed(lines):
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)
            if numbers:
                try:
                    diameter = float(numbers[0])
                    category = get_category(diameter)
                    break
                except ValueError:
                    pass

    return category, diameter


def call_model(image_path, prompt, model, options, example_images=None):
    """Query the VLM via Ollama for a single image.

    If example_images is provided, each (path, diameter) pair is added as a
    user/assistant turn before the real query, implementing the few-shot
    prompting strategy. The assistant turn includes both the category
    (derived from the reference diameter) and the diameter itself, matching
    the expected output format.
    """
    messages = []

    # Few-shot examples: one (user image, assistant answer) turn per example
    if example_images:
        for ex_img, ex_size in example_images:
            if os.path.exists(ex_img):
                messages.append({
                    "role": "user",
                    "content": "Estimate the maximal diameter of the polyp in this image:",
                    "images": [ex_img]
                })
                ex_category = get_category(ex_size)
                messages.append({
                    "role": "assistant",
                    "content": f"Category: {ex_category}\nEstimated diameter: {ex_size}"
                })

    # The actual image to be scored
    messages.append({
        "role": "user",
        "content": prompt,
        "images": [image_path]
    })

    response = ollama.chat(
        model=model,
        messages=messages,
        options=options,
        keep_alive=0,  # unload model from VRAM after each call
    )
    return response["message"]["content"]


def cleanup_memory():
    """Force garbage collection and clear the GPU cache (if torch/CUDA
    available). Called periodically to avoid memory buildup over long runs.
    """
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==== Processing ====

images = find_images(DATA_DIR)
print(f"\n{'='*60}")
print(f"Found {len(images)} images under {DATA_DIR}")
print(f"{'='*60}\n")

if len(images) == 0:
    print("No images found! Check your DATA_DIR path and folder structure.")
    exit(1)

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)

results = []

for idx, img in enumerate(tqdm(images, desc="Processing images")):
    record_id, polyp_id = extract_ids(img, DATA_DIR)
    try:
        response_text = call_model(img, PROMPT, MODEL, OLLAMA_OPTIONS, EXAMPLE_IMAGES)
        category, diameter_mm = parse_response(response_text)

        results.append({
            "record_id": record_id,
            "polyp_id": polyp_id,
            "size_category": category,
            "maximal_diameter_mm": diameter_mm,
            "model_response": response_text,
        })

        del response_text

    except Exception as e:
        # On failure, still record the image so it's clear which one
        # failed; the error message is stored in model_response.
        results.append({
            "record_id": record_id,
            "polyp_id": polyp_id,
            "size_category": None,
            "maximal_diameter_mm": None,
            "model_response": str(e),
        })
        print(f"\nError processing {img}:")
        traceback.print_exc()

    # Periodic memory cleanup
    if (idx + 1) % CLEANUP_INTERVAL == 0:
        cleanup_memory()
        tqdm.write(f"Cleaned up memory at image {idx + 1}/{len(images)}")

    # Periodic checkpoint save, in case the run is interrupted
    if (idx + 1) % SAVE_INTERVAL == 0:
        df_temp = pd.DataFrame(results)
        df_temp.to_excel(OUT_XLSX.replace('.xlsx', '_progress.xlsx'), index=False)
        tqdm.write(f"Saved progress: {idx + 1}/{len(images)} images")

# Final cleanup
cleanup_memory()

# ==== Save final results ====
df = pd.DataFrame(results)
df.to_excel(OUT_XLSX, index=False)
print(f"\n{'='*60}")
print(f"Saved {len(df)} results to {OUT_XLSX}")
print(f"{'='*60}\n")

# ==== Summary statistics ====
print("\nSummary Statistics:")
print(f"Total images processed: {len(df)}")
print(f"\nCategory distribution:")
print(df['size_category'].value_counts(dropna=False))
print(f"\nDiameter statistics:")
print(df['maximal_diameter_mm'].describe())
