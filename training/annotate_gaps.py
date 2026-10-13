"""缺貨空位人工標註工具（驗收 1.8）。

SKU-110K 沒有空位標註，所以分兩步：
1. propose：用偵測模型 + shelf.py 從測試集挑出「看起來有空位」的圖，寫入候選框。
2. review：人工逐張確認。候選框只是起點，**模型漏掉的空位一定要自己補框**，
   否則召回率會被高估（只標模型找得到的空位，召回率永遠是 100%）。

review 操作：
  左鍵拖曳     新增空位框
  右鍵點框內   刪除該框
  n / 空白鍵   標記已確認並存檔，下一張
  b            上一張
  q / Esc      存檔並離開

結果存在 training/data/gap_annotations.json（只有檔名與座標，不含圖片，可以 commit）。
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import cv2

from training.common import DATA, MODELS, SKU
from vision.detector import OnnxDetector
from vision.io import load_image
from vision.shelf import analyze_shelf

OUT = DATA / "gap_annotations.json"
WINDOW = "gap review"
MAX_VIEW = (1500, 900)


def load() -> dict[str, Any]:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    return {"images": []}


def save(data: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def propose(n: int, scan: int) -> None:
    det = OnnxDetector(MODELS / "det-yolov8n-v1.onnx")
    names = (DATA / "splits" / "sku110k_test.txt").read_text().split()[:scan]
    scored = []
    for name in names:
        img = load_image(SKU / "images" / name)
        a = analyze_shelf(det.detect(img), img.shape)
        if a.gaps:
            scored.append((a.gap_ratio, name, [[g.x1, g.y1, g.x2, g.y2] for g in a.gaps]))
    scored.sort(reverse=True)
    data = load()
    existing = {i["image"] for i in data["images"]}
    for _, name, gaps in scored:
        if len(data["images"]) >= n:
            break
        if name not in existing:
            data["images"].append({"image": name, "gaps": gaps, "reviewed": False})
    save(data)
    print(f"候選圖 {len(data['images'])} 張 -> {OUT}")


def review() -> None:
    data = load()
    items = data["images"]
    if not items:
        print("沒有候選圖，先執行 propose")
        return
    i = 0
    state: dict[str, Any] = {}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        s = state["scale"]
        gx, gy = x / s, y / s
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (gx, gy)
        elif event == cv2.EVENT_LBUTTONUP and state.get("start"):
            x0, y0 = state.pop("start")
            if abs(gx - x0) > 5 and abs(gy - y0) > 5:
                items[i]["gaps"].append([min(x0, gx), min(y0, gy), max(x0, gx), max(y0, gy)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            items[i]["gaps"] = [
                g for g in items[i]["gaps"] if not (g[0] <= gx <= g[2] and g[1] <= gy <= g[3])
            ]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    img = None
    loaded = -1
    while True:
        if loaded != i:
            img = load_image(SKU / "images" / items[i]["image"])
            state["scale"] = min(MAX_VIEW[0] / img.shape[1], MAX_VIEW[1] / img.shape[0], 1.0)
            loaded = i
        assert img is not None
        s = state["scale"]
        view = cv2.resize(img, None, fx=s, fy=s)
        for g in items[i]["gaps"]:
            cv2.rectangle(
                view, (int(g[0] * s), int(g[1] * s)), (int(g[2] * s), int(g[3] * s)), (0, 0, 255), 2
            )
        done = sum(1 for it in items if it.get("reviewed"))
        status = f"{i + 1}/{len(items)}  reviewed {done}  gaps {len(items[i]['gaps'])}"
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord(" ")):
            items[i]["reviewed"] = True
            save(data)
            i = min(i + 1, len(items) - 1)
        elif key == ord("b"):
            i = max(i - 1, 0)
        elif key in (ord("q"), 27):
            save(data)
            break
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--scan", type=int, default=400)
    sub.add_parser("review")
    args = ap.parse_args()
    if args.cmd == "propose":
        propose(args.n, args.scan)
    else:
        review()


if __name__ == "__main__":
    main()
