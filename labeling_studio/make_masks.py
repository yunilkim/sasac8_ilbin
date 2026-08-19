#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_masks.py  --  기존 bbox 를 SAM 에 먹여 폴리곤(마스크)으로 승격
================================================================================

 실행:
     .venv\\Scripts\\python.exe make_masks.py vest-helmet_crop
     .venv\\Scripts\\python.exe make_masks.py vest-helmet_crop --from-workspace vest-helmet_02
     .venv\\Scripts\\python.exe make_masks.py vest-helmet_crop --split train --limit 200

     # 학습 가능한 seg 데이터셋으로 통째로 뽑기
     .venv\\Scripts\\python.exe make_masks.py vest-helmet_crop_dedup \\
         --as-dataset vest-helmet_crop_dedup_seg --sam sam2.1_b.pt

--------------------------------------------------------------------------------
 [ 무엇을 하는가 ]
--------------------------------------------------------------------------------
 박스를 프롬프트로 넣으면 SAM 이 그 안의 물체 경계를 따 준다.
 사람은 이미 "여기에 조끼가 있다"는 정보를 박스로 지불했고, SAM 은 모양만
 올려준다. 사람 노동이 추가로 들지 않는다.

 출력은 두 가지다.
   (기본)         models/<이름>/preds/<split>/  -- 검수용 예측. 신뢰도가 붙는다.
   --as-dataset   db/<이름>/                    -- 학습 가능한 seg 데이터셋.
                                                   이미지까지 복사하고 신뢰도는 뺀다.

 예측은 스튜디오에서 시드로 얹어 사람이 검수하는 용도고, 데이터셋은
 ultralytics 에 그대로 물릴 수 있는 순수 YOLO seg 형식이다.

--------------------------------------------------------------------------------
 [ 출처 -- 이 데이터셋은 정답지가 아니다 ]
--------------------------------------------------------------------------------
 박스의 위치와 클래스는 원본에서 왔고(GT 면 B, 사람이면 A), 마스크의 '모양'만
 SAM 이 만들었다. SAM 은 이 데이터셋을 본 적이 없으므로 zero-shot(D) 이다.

 즉 --as-dataset 으로 나온 seg 라벨은 사람이 한 번도 확인하지 않은 기계 출력이다.
 박스가 GT 에서 왔다면 그 데이터셋은 GT 의 지위를 그대로 물려받는다 -- 채점과
 시뮬레이션 용도이고, 자동 라벨러의 학습 데이터로 쓰면 같은 전제 위반이 된다.
 manifest 에 이 사실이 기록된다.

--------------------------------------------------------------------------------
 [ 실패를 어떻게 다루는가 ]
--------------------------------------------------------------------------------
 SAM 이 박스 개수와 다른 개수의 마스크를 돌려주면 클래스가 어긋나므로 그 장은
 통째로 버린다. 그런데 '버린다'의 뜻이 두 출력에서 다르다.

   예측     빈 파일을 쓴다. 검수 화면에서 '예측 없음'으로 보이면 그만이다.
   데이터셋 이미지를 아예 뺀다. 빈 라벨 파일은 YOLO 에게 "여기엔 아무것도
            없다"는 뜻이고, 그건 사실이 아니다. 학습에 거짓을 먹이게 된다.

 원본 라벨이 원래 비어 있던 장(진짜 배경)은 그대로 빈 채로 남긴다. 이 둘은
 완전히 다르며 manifest 에서 구분해 센다.
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DATA = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BATCH_LOG = 500
# 폴리곤 단순화 강도. SAM 은 점을 수백 개 뱉는데 그대로 두면 라벨 파일이 비대해지고
# 편집기에서 다루기도 어렵다. 이미지 대각선의 이 비율만큼을 허용 오차로 쓴다.
SIMPLIFY = 0.0015

# 마스크가 프롬프트 박스를 얼마나 벗어나도 되는가.
#
# 왜 필요한가: 첫 시험 40장에서 08930(조끼 제품 사진)의 마스크가 조끼가 아니라
# '화면 전체'로 나왔다. 박스가 이미지 대부분을 덮는 경우 SAM 이 물체가 아니라
# 배경을 잡아 버린다. 개수는 맞으니 개수 검사로는 안 걸린다.
#
# 그래서 면적으로 본다. 폴리곤 넓이 중 프롬프트 박스 안에 든 비율이 이 값보다
# 낮으면 그 마스크는 박스가 가리킨 물체가 아니다.
# 살짝 삐져나가는 것은 정상이다 -- 박스가 물체보다 조금 작게 그려졌을 수 있고,
# 폴리곤 단순화로도 몇 픽셀은 밀린다. 그래서 1.0 이 아니라 0.85 다.
MIN_INSIDE = 0.85


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def imread_u(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except OSError:
        return None


def read_boxes(lp: str):
    """(cls, x0,y0,x1,y1) 정규화 목록. 박스 줄만 취한다."""
    out = []
    if not os.path.isfile(lp):
        return out
    for line in open(lp, "r", encoding="utf-8", errors="replace"):
        t = line.split()
        if len(t) < 5:
            continue
        try:
            cls = int(float(t[0]))
            v = [float(x) for x in t[1:]]
        except ValueError:
            continue
        if len(v) in (4, 5):          # 5 번째는 신뢰도
            cx, cy, w, h = v[:4]
            out.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def poly_line(cls: int, pts: np.ndarray, w: int, h: int,
              conf: float, with_conf: bool = True) -> str:
    """픽셀 폴리곤 -> 정규화 YOLO seg 한 줄.

    with_conf=False 면 신뢰도를 뺀 순수 YOLO seg 형식이다. ultralytics 는
    'cls 다음은 전부 좌표쌍'으로 읽으므로 꼬리에 신뢰도가 붙어 있으면 좌표
    개수가 홀수가 되어 파싱이 깨진다. 학습용 데이터셋에는 반드시 빼야 한다.
    """
    eps = SIMPLIFY * float(np.hypot(w, h))
    ap = cv2.approxPolyDP(pts.astype(np.int32).reshape(-1, 1, 2), eps, True)
    ap = ap.reshape(-1, 2).astype(np.float64)
    if len(ap) < 3:
        return ""
    ap[:, 0] = np.clip(ap[:, 0] / w, 0.0, 1.0)
    ap[:, 1] = np.clip(ap[:, 1] / h, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in ap.reshape(-1))
    return f"{cls} {coords}" + (f" {conf:.4f}" if with_conf else "")


def rmtree_force(path: str) -> None:
    """읽기전용 파일이 섞여 있어도 확실히 지운다.

    db 원본은 GT 봉인으로 OS 읽기전용이고, copy2 가 그 속성까지 복사한다.
    그래서 파생본에도 읽기전용 파일이 생기는데, shutil.rmtree 는 그걸 못 지운다.
    ignore_errors=True 를 쓰면 실패를 삼켜서 '지웠다고 생각했는데 남아 있는'
    상태가 되고, 다음 실행이 그 파일에서 PermissionError 로 죽는다.
    (실제로 그렇게 죽었다 -- data.yaml 이 살아남아 있었다.)
    """
    if not os.path.isdir(path):
        return

    def onerr(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    try:                       # 3.12+ 는 onexc, 그 이전은 onerror
        shutil.rmtree(path, onexc=lambda f, p, e: onerr(f, p, e))
    except TypeError:
        shutil.rmtree(path, onerror=lambda f, p, e: onerr(f, p, e))


def unlock(path: str) -> None:
    """복사하면서 딸려온 읽기전용 속성을 푼다. 파생본까지 잠글 이유가 없다."""
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def inside_frac(pts: np.ndarray, box, w: int, h: int) -> float:
    """폴리곤 넓이 중 프롬프트 박스 안에 든 비율.

    다각형 클리핑을 직접 하지 않고 마스크 두 장을 그려 픽셀로 센다. 폴리곤이
    자기 자신과 교차하는 경우가 있어서(SAM 출력은 종종 그렇다) 기하 연산보다
    래스터가 안전하다. 640×640 한 장이라 비용도 무시할 만하다.
    """
    p = np.asarray(pts, np.int32).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [p], 1)
    tot = int(m.sum())
    if tot == 0:
        return 0.0
    x0, y0, x1, y1 = (int(round(box[0])), int(round(box[1])),
                      int(round(box[2])), int(round(box[3])))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(m[y0:y1, x0:x1].sum()) / tot


def find_splits(root: str):
    found = []

    def walk(d, depth):
        if depth > 3:
            return
        try:
            subs = [e for e in os.scandir(d) if e.is_dir()]
        except OSError:
            return
        names = {e.name.lower(): e.path for e in subs}
        if "images" in names and "labels" in names:
            found.append((os.path.relpath(d, root).replace("\\", "/"),
                          names["images"], names["labels"]))
            return
        for e in subs:
            walk(e.path, depth + 1)

    walk(root, 0)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="bbox -> SAM 마스크 승격")
    ap.add_argument("dataset", help="db/ 아래 데이터셋 이름")
    ap.add_argument("--from-workspace", default=None,
                    help="박스를 가져올 작업공간 (기본: 데이터셋의 db 라벨)")
    ap.add_argument("--out", default=None, help="models/ 아래 결과 이름")
    ap.add_argument("--sam", default="mobile_sam.pt",
                    help="mobile_sam.pt(기본·빠름) / sam2.1_t.pt / sam_b.pt")
    ap.add_argument("--split", default=None, help="특정 스플릿만")
    ap.add_argument("--limit", type=int, default=0, help="앞 N 장만 (시험용)")
    ap.add_argument("--preview", type=int, default=0, help="확인용 그림 N 건")
    ap.add_argument("--as-dataset", default=None, metavar="이름",
                    help="db/<이름> 에 학습 가능한 seg 데이터셋으로 뽑는다")
    ap.add_argument("--link", action="store_true",
                    help="--as-dataset 일 때 이미지를 복사 대신 하드링크")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-inside", type=float, default=MIN_INSIDE,
                    help=f"마스크 넓이 중 프롬프트 박스 안에 들어야 하는 최소 비율 "
                         f"(기본 {MIN_INSIDE}). 0 이면 검사 끔")
    ap.add_argument("--on-escape", choices=("box", "drop"), default="box",
                    help="박스를 벗어난 마스크를 어떻게 할지. "
                         "box=박스를 사각 폴리곤으로 대신 씀(기본) · "
                         "drop=그 장을 통째로 뺌")
    a = ap.parse_args()

    src = os.path.join(DATA, "db", a.dataset)
    if not os.path.isdir(src):
        print(f"[!] 데이터셋 없음: {src}")
        return 1
    as_ds = bool(a.as_dataset)
    ds_root = os.path.join(DATA, "db", a.as_dataset) if as_ds else None
    if as_ds:
        if os.path.normpath(ds_root) == os.path.normpath(src):
            print("[!] 출력이 원본과 같습니다. 원본은 절대 덮지 않습니다.")
            return 1
        if os.path.isdir(ds_root) and not a.overwrite:
            print(f"[!] 이미 있습니다: {ds_root}   (--overwrite 로 덮어쓰기)")
            return 1
        if os.path.isdir(ds_root) and a.overwrite:
            rmtree_force(ds_root)
    splits = find_splits(src)
    if a.split:
        splits = [s for s in splits if s[0] == a.split]
    if not splits:
        print("[!] images/labels 쌍을 못 찾음")
        return 1

    ws_dir = None
    box_src = f"db/{a.dataset}"
    if a.from_workspace:
        ws_dir = os.path.join(DATA, "work", a.from_workspace)
        if not os.path.isdir(ws_dir):
            print(f"[!] 작업공간 없음: {ws_dir}")
            return 1
        box_src = f"work/{a.from_workspace}"

    name = a.out or f"sam_{a.dataset}" + (f"_{a.from_workspace}" if a.from_workspace else "")
    out_root = os.path.join(DATA, "models", name)

    print(f"이미지  : db/{a.dataset}")
    print(f"박스 출처: {box_src}")
    print(f"SAM     : {a.sam}")
    if as_ds:
        print(f"결과    : db/{a.as_dataset}/   (학습 가능한 seg 데이터셋)")
        print(f"          이미지 {'하드링크' if a.link else '실제 복사'} · "
              f"라벨은 신뢰도 없는 순수 YOLO seg")
    else:
        print(f"결과    : models/{name}/preds/")
    print()

    from ultralytics import SAM
    model = SAM(a.sam)

    t0 = time.perf_counter()
    n_img = n_box = n_mask = n_fail = 0
    n_bg = n_dropped_img = 0        # 원래 빈 장 / 실패해서 뺀 장
    n_escaped = 0                   # 박스를 벗어난 마스크
    n_boxfill = 0                   # 그중 사각 폴리곤으로 대체한 것
    previews = []
    dropped: list = []
    confs: list = []
    insides: list = []
    per_class: Counter = Counter()

    def place(sp: str, fn: str, ipath: str) -> None:
        """데이터셋 모드에서 이미지를 결과 폴더에 놓는다."""
        d = os.path.join(ds_root, sp, "images", fn)
        if os.path.exists(d):
            return
        if a.link:
            try:
                os.link(ipath, d)
                return
            except OSError:
                pass
        shutil.copy2(ipath, d)
        unlock(d)

    for label, idir, ldir in splits:
        files = sorted(f for f in os.listdir(idir)
                       if f.lower().endswith(IMG_EXTS))
        if a.limit:
            files = files[:a.limit]
        if as_ds:
            odir = os.path.join(ds_root, label, "labels")
            os.makedirs(os.path.join(ds_root, label, "images"), exist_ok=True)
        else:
            odir = os.path.join(out_root, "preds", label)
        os.makedirs(odir, exist_ok=True)
        print(f"[{label}] {len(files)}장")

        for i, fn in enumerate(files):
            lp = (os.path.join(ws_dir, "labels", label, stem(fn) + ".txt")
                  if ws_dir else os.path.join(ldir, stem(fn) + ".txt"))
            boxes = read_boxes(lp)
            op = os.path.join(odir, stem(fn) + ".txt")
            ipath = os.path.join(idir, fn)
            if not boxes:
                # 원본이 원래 비어 있던 장 = 진짜 배경. 데이터셋에서도 살린다.
                # 이건 SAM 실패와 완전히 다르다.
                open(op, "w", encoding="utf-8").close()
                if as_ds:
                    place(label, fn, ipath)
                    n_bg += 1
                continue
            img = imread_u(ipath)
            if img is None:
                continue
            n_img += 1
            n_box += len(boxes)
            h, w = img.shape[:2]
            xyxy = [[b[1] * w, b[2] * h, b[3] * w, b[4] * h] for b in boxes]

            try:
                res = model(img, bboxes=xyxy, verbose=False, device=0)
            except Exception as e:  # noqa: BLE001
                print(f"    실패 {fn[:30]}: {type(e).__name__}")
                n_fail += len(boxes)
                if as_ds:
                    n_dropped_img += 1
                    dropped.append({"split": label, "image": fn,
                                    "boxes": len(boxes),
                                    "reason": type(e).__name__})
                    if os.path.isfile(op):
                        os.remove(op)
                else:
                    open(op, "w", encoding="utf-8").close()
                continue

            lines = []
            r = res[0]
            polys = list(r.masks.xy) if r.masks is not None else []
            # SAM 은 입력 박스 순서대로 마스크를 돌려준다. 개수가 어긋나면
            # 클래스가 엉뚱하게 붙으므로 그때는 통째로 버린다.
            mismatch = len(polys) != len(boxes)
            escaped = 0
            if mismatch:
                n_fail += len(boxes)
            else:
                for (cls, *bx), pts, conf in zip(
                        boxes, polys,
                        (r.masks.conf.tolist() if getattr(r.masks, "conf", None)
                         is not None else [None] * len(polys))):
                    xy = [bx[0] * w, bx[1] * h, bx[2] * w, bx[3] * h]
                    frac = inside_frac(pts, xy, w, h)
                    insides.append(round(frac, 4))
                    use = np.asarray(pts)
                    if frac < a.min_inside:
                        # 박스가 가리킨 물체가 아니라 배경을 잡았다.
                        escaped += 1
                        n_escaped += 1
                        if a.on_escape == "drop":
                            continue
                        # 박스를 사각 폴리곤으로 대신 쓴다. 모양은 거칠지만
                        # 위치와 클래스는 맞다. 아래 주석 참조.
                        use = np.array([[xy[0], xy[1]], [xy[2], xy[1]],
                                        [xy[2], xy[3]], [xy[0], xy[3]]],
                                       np.float64)
                        n_boxfill += 1
                    ln = poly_line(cls, use, w, h,
                                   1.0 if conf is None else float(conf),
                                   with_conf=not as_ds)
                    if ln:
                        lines.append(ln)
                        n_mask += 1
                        per_class[cls] += 1
                        if conf is not None:
                            confs.append(round(float(conf), 4))
                    else:
                        n_fail += 1
                # 사각 폴리곤으로 대체했으면 라벨은 남았으므로 실패가 아니다.
                # 벗어난 개수는 n_escaped 로 따로 세고 있다.
                if escaped and a.on_escape == "drop":
                    n_fail += escaped

            if as_ds and (mismatch or len(lines) != len(boxes)):
                # 라벨이 원본보다 적어진 장은 데이터셋에 넣지 않는다.
                # 빈/모자란 라벨은 "여기엔 없다"는 거짓을 학습시킨다.
                n_mask -= len(lines)
                for ln in lines:
                    per_class[int(ln.split()[0])] -= 1
                del confs[len(confs) - len(lines):]
                n_dropped_img += 1
                dropped.append({"split": label, "image": fn,
                                "boxes": len(boxes), "masks": len(lines),
                                "reason": "mask_count_mismatch" if mismatch
                                          else ("mask_escaped_box" if escaped
                                                else "polygon_too_small")})
                if os.path.isfile(op):
                    os.remove(op)
                continue

            with open(op, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            if as_ds:
                place(label, fn, ipath)

            if a.preview and len(previews) < a.preview and lines:
                previews.append((fn, img.copy(), polys, xyxy))
            if (i + 1) % BATCH_LOG == 0:
                el = time.perf_counter() - t0
                print(f"    {i + 1}/{len(files)}  {el:.0f}s "
                      f"({n_img / max(el, 1e-9):.1f} img/s)")

    el = time.perf_counter() - t0
    lost = sum(d.get("boxes", 0) for d in dropped)
    print(f"\n{'=' * 60}")
    print(f"이미지 {n_img}장 · {el:.0f}초 ({n_img / max(el, 1e-9):.1f} img/s)")
    print(f"박스 {n_box}개 → 마스크 {n_mask}개")
    if as_ds and lost:
        # 여기가 헷갈리기 쉽다. 박스 하나가 실패하면 그 장을 통째로 빼므로,
        # 잃는 박스는 실패한 박스보다 훨씬 많다. 둘을 따로 보여준다.
        print(f"  마스크가 박스를 벗어남 {n_escaped}개 "
              f"→ 그 {n_dropped_img}장을 빼면서 박스 {lost}개 손실")
        print(f"  {n_box} − {lost} = {n_mask}  (맞아야 정상)")
    elif n_fail:
        print(f"  실패 {n_fail}개")
    if confs:
        c = np.asarray(confs)
        print(f"마스크 신뢰도  중앙값 {np.median(c):.3f} · "
              f"하위10% {np.percentile(c, 10):.3f} · 최소 {c.min():.3f}")
    else:
        print("마스크 신뢰도  이 SAM 은 마스크별 신뢰도를 주지 않는다 "
              "(품질 판단은 아래 '박스 안 비율'로 한다)")
    if insides:
        v = np.asarray(insides)
        print(f"박스 안 비율   중앙값 {np.median(v):.3f} · "
              f"하위10% {np.percentile(v, 10):.3f} · 최소 {v.min():.3f}")
        print(f"               기준 {a.min_inside} 미달 {n_escaped}개 "
              f"({n_escaped / max(len(insides), 1):.1%}) — 배경을 잡은 마스크")
        if n_boxfill:
            print(f"               그중 {n_boxfill}개는 박스를 사각 폴리곤으로 대체")
    if as_ds:
        print(f"배경(원래 빈 장) {n_bg}장 유지 · "
              f"실패해서 뺀 장 {n_dropped_img}장")

    if previews:
        cell = 320
        cols = min(4, len(previews))
        rows = (len(previews) + cols - 1) // cols
        sheet = np.full((rows * (cell + 22) + 6, cols * cell, 3), 28, np.uint8)
        for i, (fn, img, polys, xyxy) in enumerate(previews):
            vis = img.copy()
            for pts in polys:
                p = np.asarray(pts, np.int32)
                cv2.fillPoly(vis, [p], (60, 220, 120))
                cv2.polylines(vis, [p], True, (30, 160, 80), 2)
            vis = cv2.addWeighted(vis, 0.45, img, 0.55, 0)
            for b in xyxy:
                cv2.rectangle(vis, (int(b[0]), int(b[1])),
                              (int(b[2]), int(b[3])), (80, 160, 255), 2)
            r_, c_ = divmod(i, cols)
            oy, ox = r_ * (cell + 22) + 20, c_ * cell
            sheet[oy:oy + cell, ox:ox + cell] = cv2.resize(
                vis, (cell, cell), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet, fn[:26], (ox + 3, oy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 210), 1,
                        cv2.LINE_AA)
        pv = os.path.join(DATA, "docs", "sam_preview.jpg")
        os.makedirs(os.path.dirname(pv), exist_ok=True)
        cv2.imwrite(pv, sheet)
        print(f"확인용 그림 -> {pv}  (파랑=입력 박스 / 초록=SAM 마스크)")

    meta = {
        "name": a.as_dataset or name,
        "kind": "sam_seg_dataset" if as_ds else "sam_mask_from_box",
        "sam": a.sam, "source_dataset": f"db/{a.dataset}",
        "box_source": box_src,
        "box_provenance_note":
            "박스의 위치·클래스는 위 출처에서 왔고 마스크 모양만 SAM 이 만들었다. "
            "SAM 은 이 데이터셋을 본 적이 없으므로 zero-shot(D) 이다.",
        "images": n_img, "boxes": n_box, "masks": n_mask, "failed": n_fail,
        "masks_per_class": {str(k): v for k, v in sorted(per_class.items())},
        "mask_conf": ({"median": round(float(np.median(confs)), 4),
                       "p10": round(float(np.percentile(confs, 10)), 4),
                       "min": round(float(np.min(confs)), 4),
                       "max": round(float(np.max(confs)), 4)} if confs else
                      {"note": "이 SAM 은 마스크별 신뢰도를 제공하지 않는다"}),
        "inside_frac": ({"threshold": a.min_inside,
                         "median": round(float(np.median(insides)), 4),
                         "p10": round(float(np.percentile(insides, 10)), 4),
                         "min": round(float(np.min(insides)), 4),
                         "below_threshold": n_escaped} if insides else {}),
        "escaped_masks": n_escaped,
        "escaped_policy": a.on_escape,
        "box_fallback_masks": n_boxfill,
        "box_fallback_note":
            "SAM 이 배경을 잡은 마스크는 프롬프트 박스를 사각 폴리곤으로 대신 썼다. "
            "모양은 거칠지만 위치와 클래스는 맞다. 그 장을 통째로 빼는 방식도 있지만, "
            "박스가 많은(붐비는) 장일수록 실패 확률이 높아 쉬운 장만 남는 편향이 생긴다.",
        "simplify_eps_ratio": SIMPLIFY,
        "seconds": round(el, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": ("cls x1 y1 x2 y2 ... xn yn  (정규화, 순수 YOLO seg)" if as_ds
                   else "cls x1 y1 x2 y2 ... xn yn conf  (정규화)"),
    }

    if not as_ds:
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "model.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"\n-> {out_root}")
        return 0

    # ---- 데이터셋 마무리 ---------------------------------------------------
    # data.yaml 은 원본 것을 그대로 쓴다. 클래스 이름과 스플릿 구조가 같다.
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith(".json"):
            d = os.path.join(ds_root, f)
            shutil.copy2(p, d)
            unlock(d)

    print("\n[검증] 쓴 것을 다시 읽어 원본과 맞춰 본다")
    bad = {"count": 0, "range": 0, "cls": 0, "odd": 0, "orphan": 0}
    checked = 0
    kept_per_split: Counter = Counter()
    for label, idir, ldir in splits:
        odir = os.path.join(ds_root, label, "labels")
        oidir = os.path.join(ds_root, label, "images")
        if not os.path.isdir(odir):
            continue
        imgs = {stem(f) for f in os.listdir(oidir)}
        kept_per_split[label] = len(imgs)
        for f in os.listdir(odir):
            st = stem(f)
            if st not in imgs:
                bad["orphan"] += 1
                continue
            srcp = (os.path.join(ws_dir, "labels", label, f) if ws_dir
                    else os.path.join(ldir, f))
            want = read_boxes(srcp)
            got = [ln.split() for ln in
                   open(os.path.join(odir, f), encoding="utf-8")
                   if ln.strip()]
            checked += 1
            if len(got) != len(want):
                bad["count"] += 1
                continue
            for ln, (cls, *_) in zip(got, want):
                if int(float(ln[0])) != cls:
                    bad["cls"] += 1
                v = [float(x) for x in ln[1:]]
                if len(v) % 2:
                    bad["odd"] += 1
                if v and (min(v) < 0.0 or max(v) > 1.0):
                    bad["range"] += 1
    tot_bad = sum(bad.values())
    print(f"  라벨 {checked}개 대조 — "
          + " · ".join(f"{k} {v}" for k, v in bad.items()))
    print(f"  {'전부 일치' if tot_bad == 0 else '[!] 불일치가 있습니다'}")

    meta.update({
        "output": ds_root,
        "images_written": dict(kept_per_split),
        "images_total": sum(kept_per_split.values()),
        "background_images_kept": n_bg,
        "images_dropped": n_dropped_img,
        "dropped_files": dropped,
        "image_mode": "hardlink" if a.link else "copy",
        "verify": {"labels_checked": checked, **bad, "total_bad": tot_bad},
        "WARNING":
            "이 데이터셋의 마스크는 사람이 한 번도 확인하지 않은 SAM 출력이다. "
            "박스가 GT 에서 왔다면 이 데이터셋은 GT 의 지위를 물려받는다 -- "
            "채점·시뮬레이션 용도이며 자동 라벨러의 학습 데이터로 쓰면 전제 위반이다.",
    })
    with open(os.path.join(ds_root, "seg_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"\n{'=' * 60}")
    print(f"db/{a.as_dataset}")
    print("=" * 60)
    for sp in sorted(kept_per_split):
        print(f"  {sp:<8}{kept_per_split[sp]:>7}장")
    print(f"  {'합계':<8}{sum(kept_per_split.values()):>7}장 · 마스크 {n_mask}개")
    print(f"\n-> {ds_root}")
    print(f"   seg_manifest.json 에 신뢰도·제외 목록·검증 결과")
    print(f"   학습: train_model.py --labels gt --dataset {a.as_dataset} "
          f"--base yolo11s-seg.pt")
    print(f"         (seg 라벨이므로 베이스도 -seg 모델이어야 한다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
