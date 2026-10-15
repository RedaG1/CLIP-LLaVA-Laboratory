# Vision-Language Models Lab: CLIP & LLaVA

Practical work for the *Agentic AI & Building AI Agents* module (PhD).

The lab explores two vision-language models:
- **CLIP** (`openai/clip-vit-base-patch32`): image–text similarity, zero-shot classification, accuracy evaluation, prompt engineering and text-to-image retrieval.
- **LLaVA** (`llava-hf/llava-1.5-7b-hf`, 4-bit): architecture, visual question answering, hallucination analysis, a comparison with CLIP, and a small multimodal assistant that combines both models.

**→ The full work, with code, outputs and answers to Q1–Q36, is in [`LAB_CLIP_LLaVA_Full.ipynb`](LAB_CLIP_LLaVA_Full.ipynb).**

## Results

Run on an NVIDIA RTX 5070 Ti (16 GB): CLIP ViT-B/32 in fp32, LLaVA-1.5-7B in 4-bit (5 GB of VRAM).

| Experiment | Result |
|---|---|
| CLIP zero-shot accuracy (Ex. 5, 30 CIFAR-10 images, 6 classes) | **90.0%** (27/30); errors: two ostrich-like birds, one dark cat |
| Best prompt (Ex. 6) | "a photograph showing a cat" (cosine 0.251, vs 0.227 for "a cat") |
| Retrieval for *"a person playing football"* (Ex. 8, 21 images) | the 4 football photos ranked 1–4; top-1 `football_02.jpg` (cosine 0.296) |
| LLaVA VQA accuracy (Ex. 11, 5 images × 5 questions, human-graded) | **60%** (15/25): counting 5/5, activity 4/5, objects 3/5, location 2/5, text 1/5 |
| Hallucination (Ex. 12, image with no bicycle) | Says "no bicycle", then: *"What color is the bicycle?"* → "The bicycle is **pink**." |

## Deliverables

| # | Deliverable | In the notebook | Saved file |
|---|---|---|---|
| 1 | Jupyter notebook | [`LAB_CLIP_LLaVA_Full.ipynb`](LAB_CLIP_LLaVA_Full.ipynb) | — |
| 2 | Code and outputs for every experiment | Exercises 1–13, all outputs saved in the notebook | [`results/`](results/) |
| 3 | CLIP classification accuracy | Exercise 5 (per-class accuracy, confusion matrix); prompt study in Ex. 6 | [`clip_classification.csv`](results/clip_classification.csv), [`clip_misclassified.png`](results/clip_misclassified.png) |
| 4 | Top-5 retrieval results | Exercise 8 | [`retrieval_top5.png`](results/retrieval_top5.png), [`retrieval_top5.csv`](results/retrieval_top5.csv) |
| 5 | LLaVA VQA results | Exercise 11 (25 questions, human evaluation, accuracy) | [`vqa_results.csv`](results/vqa_results.csv) |
| 6 | Hallucination example | Exercise 12 (independent, multi-turn and mitigated prompts) | [`hallucination_example.md`](results/hallucination_example.md) |
| 7 | CLIP / LLaVA comparison | Exercise 13 (comparison table + Q33–Q36) | [`clip_vs_llava.md`](results/clip_vs_llava.md) |
| 8 | Mini-project | *Final Mini-Project*: `MultimodalAssistant`, CLIP-retrieval + LLaVA bonus, optional Gradio UI | — |
| 9 | Short conclusion | Last section of the notebook | [`summary.csv`](results/summary.csv) |

## Repository structure

```text
├── LAB_CLIP_LLaVA_Full.ipynb        # the lab: code, outputs, answers
├── results/                         # deliverables saved by the notebook (CSV, PNG, MD)
├── prepare_assets.py                # downloads the lab images into assets/
├── requirements.txt
└── assets/
    ├── image.jpg          # main test image (two cats, no bicycle)    Ex. 2, 3, 4, 6, 12, 13
    ├── eval/<class>/      # labelled images, label = folder name       Ex. 5
    ├── database/          # retrieval database incl. football photos   Ex. 7, 8 + bonus
    └── vqa/               # images with people                          Ex. 10, 11
```

## Reproducing the results

Hardware used: NVIDIA RTX 5070 Ti (16 GB). Any CUDA GPU with ≥ 8 GB should work, because LLaVA is loaded in 4-bit.

**1. Environment.** RTX 50-series cards need **PyTorch ≥ 2.7 built for CUDA 12.8** (`cu128` wheels) and an NVIDIA driver R570 or newer.

```bash
py install 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" #this is done to check if cuda is available to use a GPU for accelerated learning
```

**2. Images.** They are already included in `assets/`. To download them again from scratch:

```bash
python prepare_assets.py --force
```

Sources:
- COCO val2017 #39769;
- CIFAR-10 test set (5 images per class);
- Flickr30k (football photos and everyday scenes).

**3. Run** the notebook top to bottom:

```bash
jupyter notebook LAB_CLIP_LLaVA_Full.ipynb
```

The first run downloads ~14 GB of LLaVA weights into the Hugging Face cache. Set `HF_HOME` to use another drive.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA error: no kernel image is available` / `sm_120` warning | Reinstall torch from the `cu128` index. |
| `bitsandbytes` error when loading LLaVA | `pip install -U bitsandbytes`. The notebook falls back to fp16 with CPU offload automatically (slower). |
| CUDA out of memory | Keep `LLAVA_PRECISION = "4bit"`, close other GPU apps, restart the kernel. |