"""
Run once, from this folder, with internet access:

    python prepare_assets.py            # download what is missing
    python prepare_assets.py --force    # re-download everything

Result:

    assets/
    ├── image.jpg            two cats on a sofa (COCO val2017 #39769), contains no bicycle
    ├── eval/<class>/        5 labelled CIFAR-10 test images per class (cat, dog, car, airplane, horse, bird)
    ├── database/            4 football/soccer photos + 16 everyday scenes (Flickr30k) + the cat image
    └── vqa/                 5 Flickr30k scenes with people, for visual question answering

You can replace any of these with your own photos afterwards; the notebook only reads the folders.
Downloads are cached in .cache/ (CIFAR-10 archive, ~170 MB), not in assets/.
"""

import argparse
import io
import os
import random
import shutil
import sys
import urllib.request

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
CACHE = os.path.join(ROOT, ".cache")
IMAGE_PATH = os.path.join(ASSETS, "image.jpg")
EVAL_DIR = os.path.join(ASSETS, "eval")
DB_DIR = os.path.join(ASSETS, "database")
VQA_DIR = os.path.join(ASSETS, "vqa")

COCO_CATS_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
CIFAR_TO_LAB = {"airplane": "airplane", "automobile": "car", "bird": "bird",
                "cat": "cat", "dog": "dog", "horse": "horse"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def n_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(f.lower().endswith(IMG_EXTS) for _, _, files in os.walk(folder) for f in files)


def step_main_image(force):
    if os.path.exists(IMAGE_PATH) and not force:
        print("[image.jpg] already there, skipping")
        return
    print("[image.jpg] downloading the COCO cat image ...")
    with urllib.request.urlopen(COCO_CATS_URL, timeout=60) as r:
        Image.open(io.BytesIO(r.read())).convert("RGB").save(IMAGE_PATH, quality=95)
    print("[image.jpg] saved", IMAGE_PATH)


def get_cifar():
    from torchvision.datasets import CIFAR10
    return CIFAR10(root=os.path.join(CACHE, "cifar10"), train=False, download=True)


def step_eval(force, n_per_class=5, seed=0):
    if n_images(EVAL_DIR) >= 10 and not force:
        print(f"[eval] {n_images(EVAL_DIR)} images already there, skipping")
        return
    print("[eval] building the labelled set from CIFAR-10 test images ...")
    rng = random.Random(seed)
    try:
        # 1st choice: torchvision (downloads from www.cs.toronto.edu, sometimes slow or blocked)
        ds = get_cifar()
        class_names, labels = ds.classes, ds.targets
        get_image = lambda i: ds[i][0]
    except Exception as e:
        # 2nd choice: the same test set from the Hugging Face hub
        print(f"    torchvision CIFAR-10 download failed ({type(e).__name__}: {str(e)[:100]})")
        print("    -> trying the Hugging Face copy (uoft-cs/cifar10) ...")
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar10", split="test", cache_dir=os.path.join(CACHE, "hf"))
        img_col = "img" if "img" in ds.column_names else "image"
        lab_col = "label" if "label" in ds.column_names else "labels"
        class_names, labels = ds.features[lab_col].names, ds[lab_col]
        get_image = lambda i: ds[i][img_col]
    idx_by_class = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(class_names[y], []).append(i)
    for cname, label in CIFAR_TO_LAB.items():
        out = os.path.join(EVAL_DIR, label)
        os.makedirs(out, exist_ok=True)
        for k, i in enumerate(rng.sample(idx_by_class[cname], n_per_class)):
            # CIFAR images are 32x32; upscale so they are viewable (CLIP resizes to 224 anyway)
            get_image(i).convert("RGB").resize((224, 224), Image.BICUBIC).save(os.path.join(out, f"{label}{k + 1}.png"))
    print(f"[eval] saved {n_images(EVAL_DIR)} images in {len(CIFAR_TO_LAB)} class folders")


def _to_pil(img):
    if isinstance(img, dict) and img.get("bytes") is not None:
        img = Image.open(io.BytesIO(img["bytes"]))
    return img.convert("RGB")


def stream_flickr(n_football, n_other, max_scan=8000):
    """Return (football_images, other_images) streamed from Flickr30k, or None if unavailable.
    Captions are only used to pick football photos for the pool; the notebook never sees them."""
    from datasets import load_dataset
    for repo in ["lmms-lab/flickr30k", "nlphuji/flickr30k"]:
        for split in ["test", "train"]:
            try:
                ds = load_dataset(repo, split=split, streaming=True)
                football, others = [], []
                for i, row in enumerate(ds):
                    if i >= max_scan or (len(football) >= n_football and len(others) >= n_other):
                        break
                    if row.get("image") is None:
                        continue
                    caps = row.get("caption", row.get("captions", ""))
                    caps = " ".join(caps) if isinstance(caps, list) else str(caps)
                    if any(w in caps.lower() for w in ("soccer", "football")):
                        if len(football) < n_football:
                            football.append(_to_pil(row["image"]))
                    elif len(others) < n_other:
                        others.append(_to_pil(row["image"]))
                    if i and i % 1000 == 0:
                        print(f"    scanned {i} rows: {len(football)} football, {len(others)} other")
                if football or others:
                    print(f"    using {repo} ({split})")
                    return football, others
            except Exception as e:
                print(f"    {repo}/{split} not usable: {type(e).__name__}: {str(e)[:120]}")
    return None


def step_database_and_vqa(force, n_football=4, n_db_other=16, n_vqa=5):
    need_db = n_images(DB_DIR) < 20 or force
    need_vqa = n_images(VQA_DIR) < 5 or force
    if not need_db and not need_vqa:
        print(f"[database] {n_images(DB_DIR)} images already there, skipping")
        print(f"[vqa] {n_images(VQA_DIR)} images already there, skipping")
        return
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(VQA_DIR, exist_ok=True)
    print("[database/vqa] streaming photos from Flickr30k (Hugging Face) ...")
    got = None
    try:
        got = stream_flickr(n_football, n_db_other + n_vqa)
    except ImportError:
        print("    the `datasets` package is missing: pip install datasets")
    if got:
        football, others = got
        if need_db:
            for k, im in enumerate(football):
                im.save(os.path.join(DB_DIR, f"football_{k + 1:02d}.jpg"), quality=92)
            for k, im in enumerate(others[:n_db_other]):
                im.save(os.path.join(DB_DIR, f"scene_{k + 1:02d}.jpg"), quality=92)
            if os.path.exists(IMAGE_PATH):
                shutil.copy(IMAGE_PATH, os.path.join(DB_DIR, "cats_on_sofa.jpg"))
            print(f"[database] {n_images(DB_DIR)} images ({len(football)} football)")
        if need_vqa:
            for k, im in enumerate(others[n_db_other:n_db_other + n_vqa]):
                im.save(os.path.join(VQA_DIR, f"vqa_{k + 1}.jpg"), quality=92)
            print(f"[vqa] {n_images(VQA_DIR)} images")
        if len(football) == 0:
            print("    WARNING: no football photo found - add 2-3 of your own to assets/database/")
        return
    # Fallback: CIFAR-10 thumbnails (no people, no football) so the notebook can still run
    print("    Flickr30k unavailable -> CIFAR-10 fallback. PLEASE add your own photos:")
    print("    a few football photos to assets/database/ and 5 photos with people to assets/vqa/")
    ds = get_cifar()
    if need_db:
        for k in range(20):
            img, y = ds[k * 97]
            img.resize((224, 224), Image.BICUBIC).save(os.path.join(DB_DIR, f"cifar_{ds.classes[y]}_{k:02d}.png"))
        if os.path.exists(IMAGE_PATH):
            shutil.copy(IMAGE_PATH, os.path.join(DB_DIR, "cats_on_sofa.jpg"))
    if need_vqa and os.path.exists(IMAGE_PATH):
        shutil.copy(IMAGE_PATH, os.path.join(VQA_DIR, "vqa_1_cats.jpg"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the folders are already filled")
    args = ap.parse_args()
    if args.force:
        for d in (EVAL_DIR, DB_DIR, VQA_DIR):
            shutil.rmtree(d, ignore_errors=True)
    for d in (ASSETS, EVAL_DIR, DB_DIR, VQA_DIR, CACHE):
        os.makedirs(d, exist_ok=True)

    failed = []
    for name, step in [("image.jpg", step_main_image), ("eval", step_eval), ("database/vqa", step_database_and_vqa)]:
        try:
            step(args.force)
        except Exception as e:
            failed.append(name)
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")

    print("\nSummary")
    print(f"  assets/image.jpg   {'OK' if os.path.exists(IMAGE_PATH) else 'MISSING'}")
    for label, d, need in [("eval", EVAL_DIR, 10), ("database", DB_DIR, 20), ("vqa", VQA_DIR, 5)]:
        n = n_images(d)
        print(f"  assets/{label + '/':10s} {n:3d} images {'OK' if n >= need else f'(need {need})'}")
    if failed:
        print("Some steps failed:", failed, "- check your internet connection or add the images by hand.")
        sys.exit(1)


if __name__ == "__main__":
    main()