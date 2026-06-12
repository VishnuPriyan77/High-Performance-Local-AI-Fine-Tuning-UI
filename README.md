# CoreForge-UI

CoreForge-UI is a production-focused local fine-tuning toolkit for instruction data workflows, inspired by LLaMA-Factory style tooling and redesigned for practical local usage. It combines:

- document-driven synthetic dataset generation (PDF/Text → Alpaca-style QA pairs),
- streamlined QLoRA training with a lightweight Gradio control plane,
- GGUF compilation + optional Ollama deployment,
- and a single terminal-first workflow for local iteration.

It is intentionally compact and modular, so you can run a full experiment loop in one place: **Data → Train → Compile → Deploy**.

## Project Layout

```text
coreforge-ui/
├── app.py                  # Main Gradio app entrypoint (3-tab dashboard)
├── requirements.txt        # Runtime dependencies
├── components/
│   ├── __init__.py
│   ├── dataset_gen.py      # Synthetic dataset generation engine (Text/PDF → Alpaca)
│   ├── trainer.py          # LoRA/QLoRA training wrapper (HF SFTTrainer)
│   └── quantizer.py        # GGUF compilation + Modelfile + Ollama helpers
└── utils/
    ├── __init__.py
    └── helpers.py          # Shared utilities
```

## Architecture Overview

### 1) Dataset Forge (`components/dataset_gen.py`)
- Reads source documents (`.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.log`).
- Splits content using `RecursiveCharacterTextSplitter`.
- Generates Alpaca schema samples (`instruction`, `input`, `output`) either:
  - using a local Ollama model endpoint (`/api/generate`), or
  - a local Hugging Face text-generation pipeline fallback.
- Persists final records to `data/<slug>-coreforge-alpaca.jsonl`.

### 2) Fine-Tuning Hub (`components/trainer.py`)
- Wraps a QLoRA run using `transformers` + `peft` + `trl.SFTTrainer`.
- Accepts base model + data path + hyperparameters:
  - Learning Rate
  - Batch Size
  - Epochs
  - LoRA Rank (`r`)
  - LoRA Alpha (`alpha`)
- Streams training status to the UI through callback-based logs.
- Writes adapter outputs under `outputs/<run-id>/adapter`.

### 3) Edge Compilation (`components/quantizer.py`)
- Merges PEFT adapters back into the base model.
- Converts merged checkpoints into GGUF using llama.cpp converter.
- Produces a local `Modelfile`.
- Optionally registers the model with a local Ollama daemon:
  - `ollama create <model-name> -f ./Modelfile`

### 4) Dashboard (`app.py`)
- Tabbed, single-page UI:
  1. **Dataset Forge** – upload, configure chunking, generate + preview data.
  2. **Fine-Tuning Hub** – configure training and launch with live logs.
  3. **Edge Compilation** – convert adapter and deploy via Ollama.

## Quick Start

From the `coreforge-ui` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open Gradio UI at `http://127.0.0.1:7860`.

## Notes

- CUDA support is controlled by your installed PyTorch build and available GPU hardware.
- If `llama.cpp` is not present, compilation returns a clear error with installation guidance.
- A running Ollama daemon is required for deployment step.
- By default, `unsloth-llama-3-8b` maps to `meta-llama/Meta-Llama-3-8B-Instruct`; training still runs with standard HF QLoRA today.

## Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers / PEFT / TRL / Accelerate
- Gradio (dark-themed local UI)
- bitsandbytes
- llama.cpp converter tooling (optional for GGUF export)
- Ollama (optional for local deployment)

## Git Push (Manual)

```bash
cd /Users/vishnupriyan/Codexx/QP/NEWPROJ3/coreforge-ui
git init
git add .
git commit -m "feat: build CoreForge-UI dataset generation, training, and GGUF pipeline"
git branch -M main

# Replace USER and REPO_NAME
git remote add origin https://github.com/USER/REPO_NAME.git
git push -u origin main
```

If you need to force-create on first push:

```bash
git push -u origin main --force
```

## Security / Operational Notes

- Keep tokens and API credentials out of repo history.
- Prefer isolated local environments and pinned versions.
- Validate generated synthetic instructions before production fine-tuning.
