
# Insurance FAQ Chatbot (Mistral + LoRA) - Colab-ready

## Overview
This repository provides a Google Colab–ready Jupyter Notebook to fine-tune **mistralai/Mistral-7B-Instruct-v0.2** using **LoRA (PEFT)** on a mock insurance FAQ dataset (20 examples). The notebook demonstrates dataset creation, LoRA configuration, training, and a side-by-side comparison of model outputs before and after fine-tuning.

**Files included**
- `FineTune_GPU.ipynb` — Colab-ready notebook containing all steps and code
- `faq_sample.json` — Mock insurance FAQ dataset (20 Q&A pairs)
- `comparison_results_FINETUNE_LLM_GPU.txt` — Example comparison file (structure-ready; you will generate real results after running the notebook in Colab)
- `FINETUNE_LLM_GPU_README.md` - Project Overview

## How to run (Google Colab)
1. Open Google Colab and upload or mount this notebook (`FineTune_GPU.ipynb`) or open directly if stored in Google Drive/GitHub.
2. Ensure you select a GPU runtime: `Runtime > Change runtime type > Hardware accelerator: GPU` (A100 / T4 recommended).
3. Run cells top-to-bottom. The notebook installs required packages (`transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`) and provides notes if you run out of memory.
4. After training completes, check `comparison_results_FINETUNE_LLM_GPU.txt` for before/after outputs and save your model or push to Hugging Face if desired.

## Notes & Tips
- The 7B model is large. Use `load_in_4bit=True` + `bitsandbytes` to reduce memory usage if you have a compatible GPU and the required CUDA setup on Colab.
- If you experience OOM issues on Colab free tiers, consider using a smaller model (e.g., 3B or 1.3B) for experimentation or use gradient accumulation and very small batch sizes.
- This notebook uses a tiny number of training steps for demo. Increase `max_steps` and dataset size for real fine-tuning.

## Deliverables included
- `README.md`
- `FineTune_GPU.ipynb`
- `faq_sample.json`
- `comparison_results_FINETUNE_LLM_GPU.txt`
- `FINETUNE_LLM_GPU_README.md`
