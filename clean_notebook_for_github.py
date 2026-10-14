import json
import sys

WIDGET_MIME = "application/vnd.jupyter.widget-view+json"


def clean(path):
    nb = json.load(open(path, encoding="utf-8"))
    removed = 0
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        removed += 1
    for cell in nb["cells"]:
        cell.get("metadata", {}).pop("widgets", None)
        kept = []
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if WIDGET_MIME in data:
                data.pop(WIDGET_MIME)
                removed += 1
                if not data:
                    continue
            kept.append(out)
        if "outputs" in cell:
            cell["outputs"] = kept
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{path}: removed {removed} widget item(s)")


if __name__ == "__main__":
    clean(sys.argv[1] if len(sys.argv) > 1 else "LAB_CLIP_LLaVA_Full.ipynb")
