"""缺貨空位人工標註工具（驗收 1.8）。

SKU-110K 沒有空位標註，所以分兩步：
1. propose：用偵測模型 + shelf.py 從測試集挑出「看起來有空位」的圖，寫入候選框（見 propose 說明）。
2. review：人工逐張確認。**刪掉不是空位的框，並補上漏掉的空位**，
   否則召回率會被高估（只標模型找得到的空位，召回率永遠是 100%）。
   空位的定義：貨架上本來應該放商品、現在空著、看得到背板或層板的地方。
   商品之間的正常小縫、畫面邊緣的牆或走道都不算。
   上下堆疊：同一層隔板上旁邊的商品堆兩層、只有某格上層空掉時，框上層空掉的那一塊
   （shelf.py 依框的高度分層，上下堆疊會被分成兩層，標註要和這個判斷方式一致）；
   整條隔板本來就只放一層、看不出能堆兩層時不框。判斷不出來就不框。
   前排有商品、後排空（商品被往前推）：不框。這是「庫存偏低」不是「缺貨空位」，
   正面照片看不到後排深度，框了會讓標註標準不一致；但若前排本身也露出空位（看得到層板）就要框。
   判斷原則：正面看過去，那個位置有沒有看到商品。

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

SETS = {
    # eval：報告成績用，來自 SKU-110K test；tune：只用來改演算法與選參數，來自 val。
    # 兩組圖分開，參數才不會是對著評估圖調出來的。
    "eval": (DATA / "gap_annotations.json", "sku110k_test.txt"),
    "tune": (DATA / "gap_annotations_tune.json", "sku110k_val.txt"),
}
OUT, SPLIT_FILE = SETS["eval"]
WINDOW = "gap review"
MAX_VIEW = (1500, 900)


def load() -> dict[str, Any]:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    return {"images": []}


def save(data: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def propose(n: int, scan: int, min_density: float) -> None:
    """挑候選圖並預填寬鬆版的空位框。

    兩個刻意的設計，都是為了不讓評估被高估：
    - 依測試清單順序挑，不依空位比例排序：依比例排序會專挑「偵測器漏掉一大片」的圖，
      那些「空位」其實是漏偵測，無法評估空位演算法。改用陳列密度過濾掉偵測失敗的圖。
    - 預填的是不做紋理驗證的寬鬆結果（候選比較多），由人刪掉不是空位的框；
      而不是預填要被評估的演算法本身的輸出，避免標註者被它的答案牽著走。
    已人工確認過的圖會保留，只重新產生未確認的候選。
    """
    det = OnnxDetector(MODELS / "det-yolov8n-v1.onnx")
    names = (DATA / "splits" / SPLIT_FILE).read_text().split()[:scan]
    data = load()
    data["images"] = [i for i in data["images"] if i.get("reviewed")]
    existing = {i["image"] for i in data["images"]}
    for name in names:
        if len(data["images"]) >= n:
            break
        if name in existing:
            continue
        img = load_image(SKU / "images" / name)
        a = analyze_shelf(det.detect(img), img.shape)  # 不傳 image：不做紋理驗證
        if a.gaps and a.density >= min_density:
            gaps = [
                [round(g.x1, 1), round(g.y1, 1), round(g.x2, 1), round(g.y2, 1)] for g in a.gaps
            ]
            data["images"].append({"image": name, "gaps": gaps, "reviewed": False})
    save(data)
    print(f"候選圖 {len(data['images'])} 張 -> {OUT}")


def review() -> None:
    data = load()
    items = data["images"]
    if not items:
        print("沒有候選圖，先執行 propose")
        return
    # 從第一張還沒確認的圖開始，中斷後重開或追加候選時不用從頭按過已標好的圖
    i = next((k for k, it in enumerate(items) if not it.get("reviewed")), 0)
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
    p.add_argument("--scan", type=int, default=600)
    p.add_argument("--min-density", type=float, default=0.5)
    sub.add_parser("review")
    ap.add_argument("--set", choices=sorted(SETS), default="eval")
    args = ap.parse_args()
    global OUT, SPLIT_FILE
    OUT, SPLIT_FILE = SETS[args.set]
    if args.cmd == "propose":
        propose(args.n, args.scan, args.min_density)
    else:
        review()


if __name__ == "__main__":
    main()
