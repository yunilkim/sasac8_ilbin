#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_preds.py  --  models/<모델>/ 로 db/<데이터셋> 에 예측을 생성한다
================================================================================

 실행:
     .venv\\Scripts\\python.exe make_preds.py <모델이름> [--split train] [--conf 0.05]

 예:
     .venv\\Scripts\\python.exe make_preds.py y11s_n150_random

 출력:
     models/<모델>/preds/<split>/<stem>.txt      "<cls> <cx> <cy> <w> <h> <conf>"
     models/<모델>/preds/preds.json              생성 기록

--------------------------------------------------------------------------------
 [ 왜 predict 에 배열을 넘기는가 ]
--------------------------------------------------------------------------------
 model.predict(경로) 는 파일을 한 장씩 순차로 읽고 디코드한다. 측정해보면 전체
 시간의 73% 가 여기 걸리고 GPU 는 27% 만 일한다(RTX 5070 Ti / Ryzen 9700X).

 그래서 디코드만 스레드로 미리 돌리고 predict 에는 이미 풀린 배열을 넘긴다.
 전처리·추론·NMS 는 전부 ultralytics 코드 그대로라 결과가 바뀌지 않는다.
 512장으로 대조 검증: 박스 좌표와 신뢰도가 전부 비트 단위로 동일, 129 → 239 img/s.

 GPU 디코드(nvJPEG)로 가면 4배까지 가지만 디코더가 달라져 라벨이 7.4% 흔들린다.
 학습도 같은 디코더로 바꾸기 전에는 쓰면 안 된다. 자세한 측정은 대화 기록 참조.

--------------------------------------------------------------------------------
 [ 학습에 쓴 이미지는 반드시 제외한다 ]
--------------------------------------------------------------------------------
 모델이 이미 본 이미지에는 과하게 자신 있는 예측을 낸다. 그걸 리뷰 대상에 넣으면
 자동승인율과 편집비용이 실제보다 좋게 나온다. model.json 의 train_image_list 를
 읽어 자동으로 뺀다.
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402

DATA = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CHUNK = 64          # predict 한 번에 넘길 장수
DECODE_WORKERS = 12  # 9700X(8코어/16스레드)에서 측정한 포화 지점


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="모델로 db 이미지에 예측 생성")
    ap.add_argument("model", help="models/ 아래 폴더 이름")
    ap.add_argument("--split", default="train", help="대상 스플릿 (기본 train)")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="신뢰도 하한. 낮게 잡아야 기각 티어가 남는다 (기본 0.05)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (기본 0.7)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=DECODE_WORKERS)
    ap.add_argument("--include-train-images", action="store_true",
                    help="학습에 쓴 이미지도 포함 (권장하지 않음)")
    a = ap.parse_args()

    model_dir = os.path.join(DATA, "models", a.model)
    meta_path = os.path.join(model_dir, "model.json")
    weights = os.path.join(model_dir, "weights", "best.pt")
    if not os.path.isfile(weights):
        print(f"[!] 가중치가 없습니다: {weights}")
        return 1
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    ds = meta.get("source_dataset", "db/vest-helmet")
    img_dir = os.path.join(DATA, ds.replace("/", os.sep), a.split, "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(DATA, ds.replace("/", os.sep), "images")
    if not os.path.isdir(img_dir):
        print(f"[!] 이미지 폴더를 못 찾음: {img_dir}")
        return 1

    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(IMG_EXTS))
    used = set(meta.get("train_image_list") or [])
    if used and not a.include_train_images:
        before = len(files)
        files = [f for f in files if f not in used]
        print(f"학습에 쓴 {before - len(files)}장 제외")
    if not files:
        print("[!] 대상 이미지가 없습니다.")
        return 1

    out_dir = os.path.join(model_dir, "preds", a.split)
    os.makedirs(out_dir, exist_ok=True)
    print(f"모델   : {a.model}")
    print(f"대상   : {img_dir}")
    print(f"장수   : {len(files)}  · conf≥{a.conf} · 디코드 스레드 {a.workers}")
    print(f"출력   : {out_dir}\n")

    from ultralytics import YOLO
    model = YOLO(weights)
    pool = ThreadPoolExecutor(max_workers=a.workers)

    def decode(names):
        return list(pool.map(lambda n: cv2.imread(os.path.join(img_dir, n)), names))

    t0 = time.perf_counter()
    n_box = 0
    hist: dict[float, int] = {}
    # 다음 청크 디코드를 현재 청크 추론과 겹친다
    pending = pool.submit(decode, files[:CHUNK])
    for i in range(0, len(files), CHUNK):
        names = files[i:i + CHUNK]
        imgs = pending.result()
        nxt = files[i + CHUNK:i + 2 * CHUNK]
        pending = pool.submit(decode, nxt) if nxt else None

        results = model.predict(imgs, verbose=False, device=0, conf=a.conf,
                                iou=a.iou, imgsz=a.imgsz)
        for name, r in zip(names, results):
            lines = []
            b = r.boxes
            if b is not None and len(b):
                xywhn = b.xywhn.cpu().numpy()
                confs = b.conf.cpu().numpy()
                clss = b.cls.cpu().numpy().astype(int)
                for (cx, cy, w, h), c, k in zip(xywhn, confs, clss):
                    lines.append(f"{k} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.4f}")
                    n_box += 1
                    bucket = round(float(c) // 0.05 * 0.05, 2)
                    hist[bucket] = hist.get(bucket, 0) + 1
            with open(os.path.join(out_dir, stem(name) + ".txt"),
                      "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

        done = min(i + CHUNK, len(files))
        if (i // CHUNK) % 20 == 0 or done == len(files):
            el = time.perf_counter() - t0
            print(f"  {done}/{len(files)}  {el:5.0f}s  ({done / max(el, 1e-9):.0f} img/s)")
    pool.shutdown()

    el = time.perf_counter() - t0
    print(f"\n완료: {len(files)}장 {el:.1f}초 ({len(files) / el:.0f} img/s)")
    print(f"박스 {n_box}개 (장당 {n_box / len(files):.2f})\n")
    print("conf 분포:")
    for k in sorted(hist, reverse=True):
        print(f"  {k:.2f}~{k + 0.05:.2f}  {hist[k]:6d}")

    with open(os.path.join(model_dir, "preds", "preds.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "model": a.model, "split": a.split, "images": len(files),
            "excluded_train_images": len(used) if not a.include_train_images else 0,
            "conf_floor": a.conf, "iou": a.iou, "imgsz": a.imgsz,
            "boxes": n_box, "seconds": round(el, 1),
            "img_per_sec": round(len(files) / el, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "cls cx cy w h conf",
        }, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
