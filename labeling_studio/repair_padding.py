#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 repair_padding.py  --  반사 패딩을 회색으로 되돌린다
================================================================================

 실행:
     .venv\\Scripts\\python.exe repair_padding.py vest-helmet
     .venv\\Scripts\\python.exe repair_padding.py vest-helmet --dry-run
     .venv\\Scripts\\python.exe repair_padding.py vest-helmet --out vest-helmet_gray

--------------------------------------------------------------------------------
 [ 왜 필요한가 ]
--------------------------------------------------------------------------------
 이 데이터셋의 이미지 상당수는 정사각형으로 맞추면서 모자란 가장자리를 반사
 (거울)로 채워 넣었다. Roboflow 이전 단계에서 이미 들어간 것이라 문서에 없다.

 분류에서는 무해하지만 검출에서는 해롭다. 가장자리의 조끼가 반사되면 조끼가
 하나 더 생기는데 거기엔 라벨이 없다. 모델이 받는 신호는 "조끼와 똑같이 생긴
 이것은 배경이다" -- 중립이 아니라 모순된 지도다.

 YOLO 의 표준 letterbox 는 회색 (114,114,114) 이다. 반사 띠를 그 회색으로
 덮으면 패딩 영역에 객체를 닮은 것이 사라져 "여기 라벨 없음"이 사실과 맞는다.

--------------------------------------------------------------------------------
 [ 안전장치 ]
--------------------------------------------------------------------------------
 1. db 의 원본은 열지도 쓰지도 않는다. 새 폴더에 파생본을 만든다.
 2. 라벨 박스와 겹치는 영역은 절대 덮지 않는다. 반사 검출이 틀렸을 때
    진짜 객체를 지우는 것이 이 도구가 낼 수 있는 최악의 사고다.
    겹치면 띠를 줄이고, 그래도 겹치면 그 변은 건너뛴다.
 3. 밋밋한 영역(흰 벽·하늘)은 어떤 축으로도 대칭이라 반사로 치지 않는다.
 4. 좌표는 바뀌지 않으므로 라벨 파일은 그대로 복사된다. 기존 라벨과
    이미 만든 작업공간의 라벨이 전부 그대로 유효하다.
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Optional

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DATA = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
GRAY = (114, 114, 114)   # ultralytics letterbox 기본값

# ---- 판정 기준 -------------------------------------------------------------
# 방침: 애매한 것은 건드리지 않는다. "화면이 쪼개진" 수준으로 명백한 것만 덮는다.
# 놓친 반사는 그냥 남을 뿐이지만, 잘못 덮으면 진짜 사진을 지운다. 대칭이 아니다.
#
# 임계값을 0 근처로 잡으면 안 된다는 것을 실측으로 확인했다. 06109 를 뜯어보니
# 반사는 분명 프로그램이 만든 것인데도 평균차가 4.6, 최대차가 47 이었다.
# 이유는 두 가지가 겹쳐서다.
#   1) 거울축이 두 행 사이(반 픽셀)에 있다 -- OpenCV 의 BORDER_REFLECT 와
#      BORDER_REFLECT_101 차이. 규약을 하나만 보면 10 이상으로 벌어진다.
#   2) 반사 후 JPEG 로 저장되고 Roboflow 가 640 으로 다시 인코딩했다.
#      압축 잔차 때문에 애초에 0 이 될 수 없다.
#
# 최대차로 거르면 안 된다는 것도 실측으로 알았다. 00181 은 비계·천막처럼 고주파
# 무늬가 많은데, 명백한 반사인데도 최대차가 191 이었다. JPEG 압축 오차는 날카로운
# 경계에서 크게 튄다 -- 픽셀 하나가 190 어긋나는 건 반사가 가짜라서가 아니다.
# 대신 상위 백분위를 쓴다. 소수의 튀는 화소에 휘둘리지 않는다.
MIN_BAND = 64        # 640 의 10%. 이보다 좁으면 '쪼개졌다'고 보지 않는다
TOL_1D = 3.0         # 행 평균 프로파일 일치도. 진짜 반사는 1~4 로 매우 낮다
TOL = 15.0           # 2D 평균 절대차 (실측 4.6~9.5 + JPEG 여유)
PCT = 99.0           # 이 백분위의 차이를 본다 (최대차 대신)
PCT_LIMIT = 110.0    # 백분위 차이가 이보다 크면 우연한 대칭으로 본다
MIN_TEXTURE = 20.0   # 이보다 밋밋하면 대칭이어도 반사로 안 봄 (표준편차)
MAX_FRAC = 0.45      # 한 변에서 이 비율을 넘으면 검출 실패로 간주


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def imread_u(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except OSError:
        return None


def imwrite_u(path: str, img) -> bool:
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False
    buf.tofile(path)
    return True


def detect_band(gray: np.ndarray) -> int:
    """위쪽 가장자리의 반사 패딩 높이를 px 로 돌려준다. 없으면 0.

    패딩 높이가 p 라면 rows[0:p] 는 rows[p:2p] 를 뒤집은 것과 같다.
    큰 p 부터 훑어 가장 넓은 것을 취한다.

    2단계로 나눈 이유: 대부분의 이미지에는 반사가 아예 없다. 그런데 후보 p 마다
    전체 배열을 비교하면 한 장에 수백 번의 (p×640) 연산이 든다. 먼저 행 평균
    1차원 프로파일로 후보를 좁히면 대부분의 이미지가 그 단계에서 끝난다.
    """
    h = gray.shape[0]
    limit = min(int(h * MAX_FRAC), h // 2)
    if limit < MIN_BAND:
        return 0

    prof = gray.mean(axis=1)          # 행 평균 (640개 실수)
    cands = []
    for p in range(limit, MIN_BAND - 1, -1):
        for off in (0, 1):            # 두 반사 규약 (REFLECT / REFLECT_101)
            if 2 * p + off > h:
                continue
            a = prof[0:p]
            b = prof[p + off:2 * p + off][::-1]
            m1 = float(np.abs(a - b).mean())
            if m1 < TOL_1D:
                # break 하면 안 된다. 두 규약이 1차 필터는 둘 다 통과하는데
                # 2차에서는 한쪽만 맞는 경우가 있다 (실측: 10.6 대 4.7).
                cands.append((m1, p, off))
    if not cands:
        return 0

    # 1D 일치도가 좋은 순으로 본다. 이게 반사 여부를 가르는 가장 강한 신호다.
    cands.sort()
    for _m1, p, off in cands:
        pad = gray[0:p]
        src = gray[p + off:2 * p + off][::-1]
        if pad.shape != src.shape:
            continue
        if float(src.std()) < MIN_TEXTURE:
            continue                   # 밋밋한 영역은 판정 불가
        d = np.abs(pad - src)
        # 평균만 보면 대부분 맞고 일부만 크게 어긋나는 '우연한 대칭'이 통과한다.
        # 그렇다고 최댓값으로 자르면 JPEG 잡음에 진짜 반사가 걸러진다.
        if float(d.mean()) < TOL and float(np.percentile(d, PCT)) < PCT_LIMIT:
            return p
    return 0


def detect_all_sides(img: np.ndarray) -> dict:
    """네 변의 반사 패딩 두께. 회전시켜 같은 함수를 네 번 쓴다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "top": detect_band(g),
        "bottom": detect_band(g[::-1]),
        "left": detect_band(g.T),
        "right": detect_band(g.T[::-1]),
    }


def load_boxes_line(line: str):
    """라벨 한 줄 -> (x0,y0,x1,y1) 정규화. 못 읽으면 None."""
    t = line.split()
    if len(t) < 5:
        return None
    try:
        v = [float(x) for x in t[1:]]
    except ValueError:
        return None
    if len(v) in (4, 5):
        cx, cy, bw, bh = v[:4]
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    if len(v) >= 6:
        n = len(v) - (len(v) % 2)
        pts = np.asarray(v[:n], np.float32).reshape(-1, 2)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))
    return None


def load_boxes(label_path: str) -> list:
    """라벨을 (x0,y0,x1,y1) 정규화로. 파싱 실패한 줄은 조용히 건너뛴다."""
    out = []
    if not label_path or not os.path.isfile(label_path):
        return out
    try:
        for line in open(label_path, "r", encoding="utf-8", errors="replace"):
            t = line.split()
            if len(t) < 5:
                continue
            try:
                v = [float(x) for x in t[1:]]
            except ValueError:
                continue
            if len(v) == 4:
                cx, cy, w, h = v
                out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            elif len(v) >= 6 and len(v) % 2 == 0:
                pts = np.asarray(v, np.float32).reshape(-1, 2)
                out.append((float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max()), float(pts[:, 1].max())))
    except OSError:
        pass
    return out


def inside_band(box, bands: dict, w: int, h: int) -> bool:
    """박스가 반사 띠 안에 '완전히' 들어가 있는가.

    그렇다면 그 박스는 원본에 존재하지 않는 영역에 있는 것이다 -- 반사가 만들어낸
    유령 객체를 GT 가 라벨한 경우다. 규약 R1(실제 객체여야 한다)에 따라 지운다.
    걸쳐 있는 박스는 실제 객체일 수 있으므로 여기 해당하지 않는다.
    """
    x0, y0, x1, y1 = box
    return ((bands["top"] > 0 and y1 * h <= bands["top"]) or
            (bands["bottom"] > 0 and y0 * h >= h - bands["bottom"]) or
            (bands["left"] > 0 and x1 * w <= bands["left"]) or
            (bands["right"] > 0 and x0 * w >= w - bands["right"]))


def clip_to_labels(bands: dict, boxes: list, w: int, h: int) -> tuple:
    """라벨과 겹치지 않도록 띠를 줄인다.

    반환 (조정된 bands, 줄어든 변 목록, 유령 라벨 수)

    이 도구가 낼 수 있는 최악의 사고는 반사 검출이 틀려서 진짜 객체를 지우는
    것이다. 그래서 띠에 '걸쳐 있는' 라벨에는 무조건 양보한다.

    다만 띠 안에 완전히 들어간 라벨은 다르다. 그건 반사가 만들어낸 유령이고,
    GT 가 그것까지 라벨해 둔 경우가 있다(실측 00181). 그런 라벨 때문에 양보하면
    정작 고쳐야 할 이미지를 못 고친다. 이 경우만 예외로 둔다.
    """
    b = dict(bands)
    shrunk = []
    # 수렴할 때까지 반복해야 한다. 유령 판정은 '현재 띠'에 의존하는데, 다른 박스
    # 때문에 띠가 줄면 아까 유령이던 박스가 유령이 아니게 된다. 한 번만 돌면 그
    # 박스는 양보를 못 받은 채 띠에 걸쳐 남고, 나중에 좌표 변환에서 조용히 사라진다.
    # (실측 08886: 106px 띠에서 y1=106 인 박스가 유령으로 분류됐다가 띠가 104 로
    #  줄면서 걸침 상태가 됐다)
    for _ in range(10):
        before = dict(b)
        for box in boxes:
            if inside_band(box, b, w, h):
                continue                # 유령 -- 양보하지 않는다
            x0, y0, x1, y1 = box
            if b["top"] > 0 and y0 * h < b["top"]:
                b["top"] = max(0, int(y0 * h))
                shrunk.append("top")
            if b["bottom"] > 0 and (1 - y1) * h < b["bottom"]:
                b["bottom"] = max(0, int((1 - y1) * h))
                shrunk.append("bottom")
            if b["left"] > 0 and x0 * w < b["left"]:
                b["left"] = max(0, int(x0 * w))
                shrunk.append("left")
            if b["right"] > 0 and (1 - x1) * w < b["right"]:
                b["right"] = max(0, int((1 - x1) * w))
                shrunk.append("right")
        for k in list(b):
            if b[k] < MIN_BAND:
                b[k] = 0
        if b == before:
            break
    ghosts = sum(1 for box in boxes if inside_band(box, b, w, h))
    return b, sorted(set(shrunk)), ghosts


def fill(img: np.ndarray, bands: dict) -> tuple:
    """띠를 회색으로 덮고 (결과, 바뀐 픽셀 수) 를 돌려준다."""
    out = img.copy()
    h, w = out.shape[:2]
    n = 0
    if bands["top"]:
        out[0:bands["top"], :] = GRAY
        n += bands["top"] * w
    if bands["bottom"]:
        out[h - bands["bottom"]:, :] = GRAY
        n += bands["bottom"] * w
    if bands["left"]:
        out[:, 0:bands["left"]] = GRAY
        n += bands["left"] * h
    if bands["right"]:
        out[:, w - bands["right"]:] = GRAY
        n += bands["right"] * h
    return out, n


def crop(img: np.ndarray, bands: dict) -> tuple:
    """띠를 잘라내고 (결과, 잘린 픽셀 수) 를 돌려준다."""
    h, w = img.shape[:2]
    y0, y1 = bands["top"], h - bands["bottom"]
    x0, x1 = bands["left"], w - bands["right"]
    out = img[y0:y1, x0:x1].copy()
    return out, h * w - out.shape[0] * out.shape[1]


def remap_label_line(line: str, bands: dict, w: int, h: int) -> Optional[str]:
    """라벨 한 줄을 잘라낸 좌표계로 옮긴다. 옮길 수 없으면 None.

    잘라내기는 화소만 버리는 게 아니라 좌표계를 바꾼다. 이 변환을 빠뜨리면
    모든 박스가 조용히 어긋난다 -- 이미지가 멀쩡해 보여서 눈치채기도 어렵다.
    """
    t = line.split()
    if len(t) < 5:
        return None
    try:
        cls = t[0]
        v = [float(x) for x in t[1:]]
    except ValueError:
        return None

    nw = w - bands["left"] - bands["right"]
    nh = h - bands["top"] - bands["bottom"]
    if nw <= 0 or nh <= 0:
        return None

    def mv(nx, ny):
        return ((nx * w - bands["left"]) / nw, (ny * h - bands["top"]) / nh)

    tail = ""
    if len(v) == 5 or (len(v) >= 7 and len(v) % 2 == 1):
        tail = f" {v[-1]:.4f}"       # 꼬리 신뢰도 보존
        v = v[:-1]

    if len(v) == 4:
        cx, cy, bw, bh = v
        ncx, ncy = mv(cx, cy)
        nbw, nbh = bw * w / nw, bh * h / nh
        # 안전장치가 제대로 돌았다면 박스는 잘린 영역 안에 온전히 남는다
        if ncx - nbw / 2 < -1e-6 or ncy - nbh / 2 < -1e-6 or \
           ncx + nbw / 2 > 1 + 1e-6 or ncy + nbh / 2 > 1 + 1e-6:
            return None
        return f"{cls} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}{tail}"

    if len(v) >= 6 and len(v) % 2 == 0:
        pts = []
        for i in range(0, len(v), 2):
            px, py = mv(v[i], v[i + 1])
            if not (-1e-6 <= px <= 1 + 1e-6 and -1e-6 <= py <= 1 + 1e-6):
                return None
            pts.append(f"{min(max(px, 0.0), 1.0):.6f} "
                       f"{min(max(py, 0.0), 1.0):.6f}")
        return f"{cls} " + " ".join(pts) + tail
    return None


def remap_label_file(src_lp: str, dst_lp: str, bands: dict,
                     w: int, h: int) -> tuple:
    """라벨 파일 하나를 옮겨 쓴다. 반환 (옮긴 줄 수, 버린 줄 수)."""
    kept, dropped = [], 0
    if os.path.isfile(src_lp):
        try:
            for line in open(src_lp, "r", encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                out = remap_label_line(line, bands, w, h)
                if out is None:
                    dropped += 1
                else:
                    kept.append(out)
        except OSError:
            pass
    with open(dst_lp, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    return len(kept), dropped


def find_splits(root: str) -> list:
    """images/ 와 labels/ 가 나란한 폴더를 찾는다 (studio 와 같은 규칙)."""
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
    global MIN_BAND, TOL   # --min-band / --tol 로 덮어쓸 수 있게
    ap = argparse.ArgumentParser(description="반사 패딩을 회색으로 되돌린다")
    ap.add_argument("dataset", help="db/ 아래 데이터셋 폴더 이름")
    ap.add_argument("--mode", choices=("crop", "gray"), default="crop",
                    help="crop=반사 띠를 잘라냄(기본) · gray=회색으로 덮음")
    ap.add_argument("--out", default=None, help="결과 폴더명 (기본 <이름>_<모드>)")
    ap.add_argument("--dry-run", action="store_true", help="세기만 하고 안 씀")
    ap.add_argument("--min-band", type=int, default=MIN_BAND)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--limit", type=int, default=0, help="앞 N 장만 (시험용)")
    ap.add_argument("--preview", type=int, default=0,
                    help="검출 N 건을 원본/수정본 나란히 붙인 확인용 그림으로 저장")
    a = ap.parse_args()
    MIN_BAND, TOL = a.min_band, a.tol

    src = os.path.join(DATA, "db", a.dataset)
    if not os.path.isdir(src):
        print(f"[!] 데이터셋을 찾을 수 없습니다: {src}")
        return 1
    dst = os.path.join(DATA, "db", a.out or f"{a.dataset}_{a.mode}")
    if not a.dry_run and os.path.normpath(dst) == os.path.normpath(src):
        print("[!] 출력이 원본과 같습니다. 원본은 절대 덮지 않습니다.")
        return 1

    splits = find_splits(src)
    if not splits:
        print(f"[!] images/labels 쌍을 찾지 못했습니다: {src}")
        return 1

    print(f"원본 : {src}")
    print(f"결과 : {dst}{'   (dry-run, 쓰지 않음)' if a.dry_run else ''}")
    print(f"설정 : 최소 띠 {MIN_BAND}px · 허용차 {TOL} · 질감 하한 {MIN_TEXTURE}")
    print(f"       라벨과 겹치는 영역은 덮지 않음\n")

    t0 = time.perf_counter()
    stats = {"images": 0, "repaired": 0, "px": 0, "shrunk": 0, "skipped_all": 0,
             "labels_moved": 0, "labels_dropped": 0, "ghost_labels": 0,
             "unexpected_drops": 0}
    per_side = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    records = []
    previews = []

    for label, idir, ldir in splits:
        files = sorted(f for f in os.listdir(idir)
                       if f.lower().endswith(IMG_EXTS))
        if a.limit:
            files = files[:a.limit]
        oi = os.path.join(dst, label, "images")
        ol = os.path.join(dst, label, "labels")
        if not a.dry_run:
            os.makedirs(oi, exist_ok=True)
            os.makedirs(ol, exist_ok=True)
        print(f"[{label}] {len(files)}장")

        for i, fn in enumerate(files):
            img = imread_u(os.path.join(idir, fn))
            if img is None:
                continue
            stats["images"] += 1
            h, w = img.shape[:2]
            raw = detect_all_sides(img)
            boxes = load_boxes(os.path.join(ldir, stem(fn) + ".txt"))
            bands, shrunk, ghosts = clip_to_labels(raw, boxes, w, h)
            changed = sum(bands.values()) > 0
            if changed:
                stats["ghost_labels"] += ghosts

            if changed:
                out, npx = (crop(img, bands) if a.mode == "crop"
                            else fill(img, bands))
                stats["repaired"] += 1
                stats["px"] += npx
                for k, v in bands.items():
                    if v:
                        per_side[k] += 1
                records.append({"split": label, "image": fn,
                                "detected": raw, "applied": bands,
                                "shrunk_by_label": shrunk, "pixels": npx,
                                "ghost_labels": ghosts, "size": [w, h],
                                "new_size": [out.shape[1], out.shape[0]]})
                if a.preview and len(previews) < a.preview:
                    previews.append((fn, img.copy(), out.copy(), bands))
            else:
                out = img
                if any(raw.values()):
                    stats["skipped_all"] += 1
            if shrunk:
                stats["shrunk"] += 1

            if not a.dry_run:
                imwrite_u(os.path.join(oi, fn), out)
                lp = os.path.join(ldir, stem(fn) + ".txt")
                dlp = os.path.join(ol, stem(fn) + ".txt")
                if a.mode == "crop" and changed:
                    # 잘라내면 좌표계가 바뀐다. 라벨을 반드시 함께 옮긴다.
                    # 유령 라벨은 잘린 영역 밖으로 나가므로 여기서 자동으로 빠진다.
                    k, dr = remap_label_file(lp, dlp, bands, w, h)
                    stats["labels_moved"] += k
                    stats["labels_dropped"] += dr
                    # 유령으로 센 것보다 많이 사라졌다면 어딘가 잘못된 것이다.
                    # 조용히 넘어가면 라벨이 소리 없이 줄어든다.
                    if dr != ghosts:
                        stats["unexpected_drops"] += dr - ghosts
                        records[-1]["unexpected_drop"] = dr - ghosts
                elif a.mode == "gray" and changed:
                    # 좌표는 그대로지만 유령 라벨은 회색을 가리키게 되므로 뺀다
                    kept = []
                    if os.path.isfile(lp):
                        for line in open(lp, encoding="utf-8", errors="replace"):
                            if not line.strip():
                                continue
                            bx = load_boxes_line(line)
                            if bx and inside_band(bx, bands, w, h):
                                stats["labels_dropped"] += 1
                                continue
                            kept.append(line.strip())
                    with open(dlp, "w", encoding="utf-8") as f:
                        f.write("\n".join(kept) + ("\n" if kept else ""))
                    stats["labels_moved"] += len(kept)
                elif os.path.isfile(lp):
                    shutil.copy2(lp, dlp)

            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(files)}  "
                      f"{time.perf_counter() - t0:.0f}s")

    # data.yaml 등 부속 파일도 복사
    if not a.dry_run:
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(dst, f))

    el = time.perf_counter() - t0
    print(f"\n{'=' * 62}")
    print(f"이미지 {stats['images']}장 · {el:.0f}초")
    print(f"  반사 검출·수정   {stats['repaired']}장 "
          f"({stats['repaired'] / max(stats['images'], 1):.1%})")
    print(f"  변별 적용        " + " · ".join(f"{k} {v}" for k, v in per_side.items()))
    print(f"  라벨 때문에 축소 {stats['shrunk']}장")
    print(f"  라벨 때문에 취소 {stats['skipped_all']}장")
    if stats["repaired"]:
        avg = stats["px"] / stats["repaired"] / (640 * 640)
        verb = "잘라낸" if a.mode == "crop" else "덮은"
        print(f"  {verb} 면적        평균 {avg:.1%} / 장")
    print(f"  라벨 좌표 이동   {stats['labels_moved']}개")
    print(f"  유령 라벨 삭제   {stats['labels_dropped']}개"
          f"  (반사 띠 안에 완전히 들어간 GT 박스 = 실재하지 않는 객체)")
    if stats["unexpected_drops"]:
        print(f"  ⚠ 예상 밖 삭제   {stats['unexpected_drops']}개 "
              f"-- 유령이 아닌데 사라진 라벨이 있다. manifest 의 "
              f"unexpected_drop 항목을 확인할 것")

    if previews:
        cell = 300
        cols = 3
        rows_n = (len(previews) + cols - 1) // cols
        sheet = np.full((rows_n * (cell + 26) + 6, cols * (cell * 2 + 14), 3),
                        28, np.uint8)
        for i, (fn, before, after, bd) in enumerate(previews):
            r, c = divmod(i, cols)
            oy = r * (cell + 26) + 24
            ox = c * (cell * 2 + 14)
            for j, im in enumerate((before, after)):
                s = cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA)
                sheet[oy:oy + cell, ox + j * (cell + 6):ox + j * (cell + 6) + cell] = s
            tag = " ".join(f"{k}{v}" for k, v in bd.items() if v)
            cv2.putText(sheet, f"{fn[:26]}  [{tag}]", (ox + 2, oy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1,
                        cv2.LINE_AA)
        pv = os.path.join(DATA, "docs", "repair_preview.jpg")
        os.makedirs(os.path.dirname(pv), exist_ok=True)
        imwrite_u(pv, sheet)
        print(f"\n  확인용 그림 (왼쪽=원본 / 오른쪽=수정본) {len(previews)}건")
        print(f"  -> {pv}")

    if not a.dry_run:
        mf = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": src, "output": dst,
            "params": {"min_band": MIN_BAND, "tol": TOL,
                       "min_texture": MIN_TEXTURE, "gray": list(GRAY),
                       "max_frac": MAX_FRAC},
            "stats": stats, "per_side": per_side,
            "note": "좌표는 바뀌지 않았다. 원본 라벨과 기존 작업공간 라벨이 "
                    "그대로 유효하다.",
            "records": records,
        }
        with open(os.path.join(dst, "repair_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(mf, f, ensure_ascii=False, indent=1)
        print(f"\n-> {dst}")
        print(f"   repair_manifest.json 에 장별 기록")
    return 0


if __name__ == "__main__":
    sys.exit(main())
