# VLM Polyp Size Estimation

Pipeline for evaluating vision-language models (VLMs) on colorectal polyp size estimation and classification from colonoscopy still frames, using few-shot in-context learning.

## Overview

This script queries a vision-language model (via [Ollama](https://ollama.com)) with cropped colonoscopy images and asks it to:
1. Locate the polyp in the image
2. Classify it into a clinical size category (Diminutive: 1–5mm, Small: 6–9mm, Large: ≥10mm)
3. Estimate its maximal diameter in millimeters

A fixed set of three few-shot example images (with known reference diameters and categories) is shown to the model before each query, to calibrate its size estimates against known reference points.

This script was run once per evaluated model, with `MODEL` changed between runs:
- `qwen3-vl:32b`
- `llama4:scout`
- `gemma3:27b`
- `mistral-small3.2:24b`

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally, with the relevant model(s) pulled (e.g. `ollama pull qwen3-vl:32b`)
- See `requirements.txt` for Python dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data layout

Place your cropped images under `data/cropped_5x4_1350x1080/`, following this structure:
```
data/cropped_5x4_1350x1080/<record_id>/p<polyp_id>/best1/<image>.jpg
```

Place the three few-shot example images under `example_images/`.


## Usage

1. Set `MODEL` in `vlm_polyp_size_estimation.py` to the model you want to evaluate.
2. Run:
```bash
python vlm_polyp_size_estimation.py
```

Progress is checkpointed to `results/mllm_results_progress.xlsx` every 5 images. Final results are saved to `results/mllm_results.xlsx`, with one row per image containing:
- `record_id`, `polyp_id`
- `size_category` (parsed model classification)
- `maximal_diameter_mm` (parsed model estimate)
- `model_response` (raw model output, for auditing/debugging)

## Prompt

The full prompt used is documented in the manuscript's Supplementary Materials. See `vlm_polyp_size_estimation.py` (`PROMPT` variable) for the exact text used at inference time.
