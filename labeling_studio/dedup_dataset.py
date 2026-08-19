#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 dedup_dataset.py  --  스플릿을 넘나드는 같은 사진을 걷어낸다
================================================================================

 실행:
     .venv\\Scripts\\python.exe dedup_dataset.py vest-helmet_crop --dry-run
     .venv\\Scripts\\python.exe dedup_dataset.py vest-helmet_crop --also vest-helmet
     .venv\\Scripts\\python.exe dedup_dataset.py vest-helmet_crop --within

--------------------------------------------------------------------------------
 [ 왜 필요한가 ]
--------------------------------------------------------------------------------
 vest-helmet 은 스톡 사진을 긁어 모은 것이라 같은 사진이 여러 번 들어와 있다.
 파일명은 helmet_jacket_00000 ~ _10498 연번이고 스플릿은 번호를 통째로 잘랐다
 (train #0~7349 · test #7350~8924 · valid #8925~10498). 자르기 전 목록이 이미
 섞여 있었으므로, 같은 사진의 사본이 서로 다른 블록에 떨어졌다.

 바이트가 같은 파일은 한 장도 없다 -- Roboflow 가 전부 640 으로 다시 인코딩해서
 해시가 다 다르다. 그래서 파일 해시로는 안 걸린다. 그림을 봐야 걸린다.

 실측: valid 의 12.0%, test 의 10.9% 가 train 에 같은 사진을 갖고 있다.
 이 상태로 잰 mAP 는 그만큼 후하다. 모델이 답을 외운 것을 실력으로 세는 것이다.

--------------------------------------------------------------------------------
 [ 어떻게 찾는가 ]
--------------------------------------------------------------------------------
 2단계. 먼저 dHash 64비트로 후보를 좁히고, 후보만 실제 픽셀로 확인한다.
 9526장을 전부 픽셀로 맞대면 4500만 번인데, 해시로 거르면 수천 번으로 준다.

   1) dHash: 9x8 회색조로 줄여 가로 이웃 밝기 비교 -> 64비트.
      해밍거리 <= HAMMING 이면 후보.
   2) 확인: 64x64 회색조 평균절대차(MAD)가 MAD_MAX 미만이면 같은 사진.

--------------------------------------------------------------------------------
 [ 임계값의 근거 ]
--------------------------------------------------------------------------------
 HAMMING = 5
   5 에서 8 로 늘려 봤다. MAD<8 인 확실한 쌍이 313 -> 324 로 11 쌍 늘 뿐이다
   (96.6% 가 이미 5 안에 있다). 더 넓혀도 얻는 게 없고 후보만 3 배로 는다.

 MAD_MAX = 18
   눈으로 확인한 값이다. 후보쌍을 MAD 순으로 늘어놓고 구간별로 뽑아 봤다.
     MAD  8~18  무작위 12쌍 -> 12쌍 모두 같은 장면 (같은 사진의 재인코딩,
                              같은 촬영분의 다른 컷, 살짝 다른 크롭)
     MAD 14~18  4쌍 -> 4쌍 모두 같은 장면
     MAD 20 이상 8쌍 -> 5쌍이 남남. 같은 조끼를 입었을 뿐인 다른 사진.
   즉 경계는 18 과 20 사이에 있다. 18 을 쓴다.

   이 방향의 오차는 비대칭이다. 아닌 것을 지우면 평가용 이미지가 한 장 줄 뿐이고,
   맞는 것을 놓치면 누출이 그대로 남는다. 그래도 무작정 넓히지는 않는다.
   지운 것이 한쪽으로 쏠리면(=train 과 닮은 쉬운 것만 빠지면) 평가셋 성격이
   바뀌기 때문이다.

--------------------------------------------------------------------------------
 [ 무엇을 남기는가 ]
--------------------------------------------------------------------------------
 무리마다 우선순위가 가장 높은 스플릿의 사본을 전부 남기고, 그보다 낮은
 스플릿의 사본을 전부 뺀다. 기본 우선순위는 train > valid > test.

 train 을 1순위로 두는 이유: 목적이 평가셋의 순도이기 때문이다. train 은
 사람이 라벨을 붙인 비싼 자원이라 한 장도 잃지 않고, 대신 답지 쪽을 깎는다.
 valid 를 test 보다 위에 두는 이유: valid 는 모델 고르는 데 쓰이므로 test 가
 valid 와도 독립이어야 한다.

 같은 스플릿 안의 중복은 기본적으로 건드리지 않는다. 누출이 아니기 때문이다.
 (train 안의 중복은 가벼운 증강에 가깝다.) --within 을 주면 그것도 한 장만
 남긴다.

--------------------------------------------------------------------------------
 [ 안전장치 ]
--------------------------------------------------------------------------------
 1. 원본은 읽기만 한다. 새 폴더에 파생본을 만든다.
 2. 남긴 라벨 파일은 바이트 그대로 복사한다. 끝나고 SHA-256 으로 대조한다.
    좌표를 건드리는 도구가 아니므로 한 글자라도 달라지면 사고다.
 3. 뺀 파일은 dedup_manifest.json 에 전부 이름으로 남는다. 무리 단위로
    누가 누구 때문에 빠졌는지 추적된다.
 4. --also 로 다른 파생본(예: 크롭 전 원본)의 판정을 합칠 수 있다. 크롭으로
    패딩이 사라져야 비로소 같아지는 쌍이 있고, 반대로 크롭 때문에 어긋나는
    쌍도 있다. 둘을 합집합으로 보면 어느 쪽도 놓치지 않는다.
================================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DATA = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

HAMMING = 5        # dHash 후보 문턱 (64비트 중 몇 비트까지 달라도 되는가)
MAD_MAX = 18.0     # 64x64 회색조 평균절대차. 이 미만이면 같은 사진
VERIFY = 64        # 확인용 축소 크기

POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def imread_gray(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dhash(path: str) -> int:
    """9x8 로 줄여 가로 이웃끼리 밝기를 비교한 64비트.

    밝기의 절대값이 아니라 '왼쪽보다 밝은가'만 보므로 노출·감마·재인코딩이
    달라도 값이 유지된다. 이 데이터셋처럼 전부 다시 인코딩된 경우에 맞다.
    """
    im = imread_gray(path)
    if im is None:
        return -1
    g = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    v = 0
    for b in (g[:, 1:] > g[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


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


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def scan_pairs(root: str, keys: list, index: dict, hamming: int,
               mad_max: float, workers: int) -> list:
    """한 데이터셋 안에서 같은 사진 쌍을 찾아 (i, j, mad) 로 돌려준다.

    keys 는 (split, filename) 의 전역 목록이고 index 는 그 역인덱스다.
    여러 파생본의 결과를 합치려면 좌표계가 같아야 하므로 전역 색인을 쓴다.
    """
    paths, ids = [], []
    for label, idir, _ in find_splits(root):
        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            k = (label, fn)
            if k in index:                      # 다른 파생본에 없는 파일은 건너뜀
                paths.append(os.path.join(idir, fn))
                ids.append(index[k])
    if not paths:
        return []

    with ThreadPoolExecutor(workers) as ex:
        hs = list(ex.map(dhash, paths))
    ok = [i for i, h in enumerate(hs) if h >= 0]
    arr = np.array([hs[i] for i in ok], dtype=np.uint64)
    gid = np.array([ids[i] for i in ok], dtype=np.int64)

    cache: dict[int, np.ndarray] = {}

    def small(loc: int) -> np.ndarray:
        if loc not in cache:
            im = imread_gray(paths[ok[loc]])
            cache[loc] = cv2.resize(im, (VERIFY, VERIFY),
                                    interpolation=cv2.INTER_AREA).astype(np.float32)
        return cache[loc]

    out = []
    n = len(arr)
    rng = np.arange(n)
    for i in range(n):
        x = arr[i] ^ arr
        d = POP[x.view(np.uint8).reshape(-1, 8)].sum(1)
        for j in np.nonzero((d <= hamming) & (rng > i))[0]:
            j = int(j)
            m = float(np.abs(small(i) - small(j)).mean())
            if m < mad_max:
                out.append((int(gid[i]), int(gid[j]), m))
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{n}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="스플릿을 넘나드는 같은 사진을 걷어낸다")
    ap.add_argument("dataset", help="db/ 아래 데이터셋 폴더 이름")
    ap.add_argument("--out", default=None, help="결과 폴더명 (기본 <이름>_dedup)")
    ap.add_argument("--also", action="append", default=[],
                    help="판정을 합칠 다른 파생본 (파일명이 같아야 함). 반복 가능")
    ap.add_argument("--priority", default="train,valid,test",
                    help="남길 우선순위. 앞에 있을수록 살아남는다")
    ap.add_argument("--within", action="store_true",
                    help="같은 스플릿 안의 중복도 한 장만 남긴다")
    ap.add_argument("--hamming", type=int, default=HAMMING)
    ap.add_argument("--mad", type=float, default=MAD_MAX)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="세기만 하고 안 씀")
    ap.add_argument("--preview", type=int, default=0,
                    help="뺀 것 N 건을 남긴 짝과 나란히 붙여 저장")
    a = ap.parse_args()

    src = os.path.join(DATA, "db", a.dataset)
    if not os.path.isdir(src):
        print(f"[!] 데이터셋을 찾을 수 없습니다: {src}")
        return 1
    dst = os.path.join(DATA, "db", a.out or f"{a.dataset}_dedup")
    if not a.dry_run and os.path.normpath(dst) == os.path.normpath(src):
        print("[!] 출력이 원본과 같습니다. 원본은 절대 덮지 않습니다.")
        return 1

    splits = find_splits(src)
    if not splits:
        print(f"[!] images/labels 쌍을 찾지 못했습니다: {src}")
        return 1
    prio = [s.strip() for s in a.priority.split(",") if s.strip()]
    rank = {s: i for i, s in enumerate(prio)}

    # 전역 색인 -------------------------------------------------------------
    keys: list = []
    for label, idir, _ in splits:
        for fn in sorted(os.listdir(idir)):
            if fn.lower().endswith(IMG_EXTS):
                keys.append((label, fn))
    index = {k: i for i, k in enumerate(keys)}
    n_all = len(keys)

    print(f"원본 : {src}")
    print(f"결과 : {dst}{'   (dry-run, 쓰지 않음)' if a.dry_run else ''}")
    print(f"설정 : 해밍<={a.hamming} · MAD<{a.mad} · 우선순위 {' > '.join(prio)}"
          f"{' · 스플릿 안 중복도 정리' if a.within else ''}")
    print(f"       이미지 {n_all}장 · " +
          " · ".join(f"{s}={sum(1 for k in keys if k[0] == s)}" for s in prio
                     if any(k[0] == s for k in keys)))

    # 판정 ------------------------------------------------------------------
    t0 = time.perf_counter()
    roots = [(a.dataset, src)] + [(x, os.path.join(DATA, "db", x)) for x in a.also]
    pairs_by_root: dict = {}
    for name, root in roots:
        if not os.path.isdir(root):
            print(f"[!] --also 대상을 못 찾음, 건너뜀: {root}")
            continue
        print(f"\n[{name}] 같은 사진 찾는 중")
        p = scan_pairs(root, keys, index, a.hamming, a.mad, a.workers)
        pairs_by_root[name] = p
        print(f"    쌍 {len(p)}개")

    dsu = DSU(n_all)
    all_pairs = {}
    for name, ps in pairs_by_root.items():
        for i, j, m in ps:
            k = (min(i, j), max(i, j))
            if k not in all_pairs or m < all_pairs[k][0]:
                all_pairs[k] = (m, name)
            dsu.union(i, j)
    for (i, j) in all_pairs:
        dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(n_all):
        groups[dsu.find(i)].append(i)
    multi = [v for v in groups.values() if len(v) > 1]
    print(f"\n합집합: 쌍 {len(all_pairs)}개 · 무리 {len(multi)}개 · "
          f"연루 {sum(len(v) for v in multi)}장")
    if len(pairs_by_root) > 1:
        print("  파생본별 기여")
        for name, ps in pairs_by_root.items():
            own = {(min(i, j), max(i, j)) for i, j, _ in ps}
            others = set()
            for n2, p2 in pairs_by_root.items():
                if n2 != name:
                    others |= {(min(i, j), max(i, j)) for i, j, _ in p2}
            print(f"    {name:<22} {len(own):>5}쌍 (이쪽에서만 {len(own - others)})")

    # 남길 것 고르기 --------------------------------------------------------
    drop: dict = {}      # idx -> (사유, 남은 짝 idx)
    cross_groups = 0
    for v in multi:
        sps = {keys[i][0] for i in v}
        best = min(sps, key=lambda s: rank.get(s, 99))
        keep_pool = sorted([i for i in v if keys[i][0] == best],
                           key=lambda i: keys[i][1])
        if len(sps) > 1:
            cross_groups += 1
        for i in v:
            if keys[i][0] != best:
                drop[i] = ("cross_split", keep_pool[0])
        if a.within:
            for i in keep_pool[1:]:
                drop[i] = ("within_split", keep_pool[0])

    kept_n = Counter()
    drop_n = Counter()
    for i, (sp, fn) in enumerate(keys):
        (drop_n if i in drop else kept_n)[sp] += 1

    print(f"\n{'=' * 66}")
    print("빼는 것")
    print("=" * 66)
    print(f"  스플릿을 넘는 무리 {cross_groups}개")
    r = Counter(v[0] for v in drop.values())
    print(f"  제외 총 {len(drop)}장  " +
          " · ".join(f"{k}={v}" for k, v in r.items()))
    print(f"\n  {'스플릿':<8}{'원본':>8}{'제외':>8}{'남음':>8}{'제외율':>9}")
    for sp in prio:
        if kept_n[sp] or drop_n[sp]:
            tot = kept_n[sp] + drop_n[sp]
            print(f"  {sp:<8}{tot:>8}{drop_n[sp]:>8}{kept_n[sp]:>8}"
                  f"{drop_n[sp] / tot:>8.1%}")
    tot_all = sum(kept_n.values()) + sum(drop_n.values())
    print(f"  {'합계':<8}{tot_all:>8}{len(drop):>8}"
          f"{tot_all - len(drop):>8}{len(drop) / tot_all:>8.1%}")

    # 누출이 실제로 없어졌는지 확인 ------------------------------------------
    left = 0
    for v in multi:
        alive = [i for i in v if i not in drop]
        if len({keys[i][0] for i in alive}) > 1:
            left += 1
    print(f"\n  처리 후 스플릿을 넘는 무리 {left}개 "
          f"{'(정상: 0)' if left == 0 else '<- 남았다, 확인 필요'}")

    # 미리보기 ---------------------------------------------------------------
    if a.preview:
        sel = sorted(drop.items())[:a.preview]
        rowsimg = []
        for i, (why, j) in sel:
            cells = []
            for x, tag in ((j, "남김"), (i, "뺌")):
                sp, fn = keys[x]
                p = os.path.join(src, sp, "images", fn)
                im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                im = cv2.resize(im, (220, 220))
                cv2.putText(im, f"{sp}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0) if tag == "남김" else (0, 90, 255), 2)
                cells.append(im)
            rowsimg.append(np.hstack([cells[0],
                                      np.full((220, 5, 3), 255, np.uint8),
                                      cells[1]]))
        if rowsimg:
            pv = os.path.join(DATA, "docs", "dedup_preview.jpg")
            os.makedirs(os.path.dirname(pv), exist_ok=True)
            ok, buf = cv2.imencode(".jpg", np.vstack(rowsimg),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                buf.tofile(pv)
                print(f"\n  확인용 그림 (왼쪽=남김 / 오른쪽=뺌) -> {pv}")

    if a.dry_run:
        print(f"\n{time.perf_counter() - t0:.0f}초 · dry-run 이라 쓰지 않았습니다.")
        return 0

    # 복사 -------------------------------------------------------------------
    print(f"\n복사 중...")
    copied = {"images": 0, "labels": 0, "missing_labels": 0}
    label_sha = {}
    for label, idir, ldir in splits:
        oi = os.path.join(dst, label, "images")
        ol = os.path.join(dst, label, "labels")
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        for fn in sorted(os.listdir(idir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            i = index[(label, fn)]
            if i in drop:
                continue
            shutil.copy2(os.path.join(idir, fn), os.path.join(oi, fn))
            copied["images"] += 1
            lp = os.path.join(ldir, stem(fn) + ".txt")
            dlp = os.path.join(ol, stem(fn) + ".txt")
            if os.path.isfile(lp):
                shutil.copy2(lp, dlp)
                copied["labels"] += 1
                label_sha[f"{label}/{stem(fn)}"] = (sha256(lp), sha256(dlp))
            else:
                copied["missing_labels"] += 1
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))

    bad = [k for k, (x, y) in label_sha.items() if x != y]
    print(f"  이미지 {copied['images']}장 · 라벨 {copied['labels']}개")
    if copied["missing_labels"]:
        print(f"  라벨 없는 이미지 {copied['missing_labels']}장 (원본대로)")
    print(f"  라벨 SHA-256 대조: {len(label_sha)}개 중 불일치 {len(bad)}개"
          f"{'  (정상)' if not bad else '  <- 사고'}")

    mf = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": src, "output": dst, "also": a.also,
        "params": {"hamming": a.hamming, "mad_max": a.mad,
                   "verify_size": VERIFY, "priority": prio,
                   "within": bool(a.within)},
        "stats": {
            "source_images": n_all,
            "groups": len(multi),
            "cross_split_groups": cross_groups,
            "pairs": len(all_pairs),
            "dropped": len(drop),
            "dropped_by_reason": dict(r),
            "kept_per_split": dict(kept_n),
            "dropped_per_split": dict(drop_n),
            "cross_groups_left": left,
            "label_sha_mismatch": len(bad),
            "seconds": round(time.perf_counter() - t0, 1),
        },
        "note": "이미지와 라벨을 그대로 복사했다. 좌표는 손대지 않았다. "
                "기존 작업공간의 라벨은 남은 파일에 대해 그대로 유효하다.",
        "dropped_files": [
            {"split": keys[i][0], "file": keys[i][1], "reason": why,
             "kept_instead": {"split": keys[j][0], "file": keys[j][1]}}
            for i, (why, j) in sorted(drop.items())
        ],
        "groups": [
            {"members": [{"split": keys[i][0], "file": keys[i][1],
                          "kept": i not in drop} for i in sorted(v)]}
            for v in sorted(multi, key=len, reverse=True)
        ],
    }
    with open(os.path.join(dst, "dedup_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(mf, f, ensure_ascii=False, indent=1)
    print(f"\n{time.perf_counter() - t0:.0f}초")
    print(f"-> {dst}")
    print(f"   dedup_manifest.json 에 뺀 파일 {len(drop)}개와 무리 {len(multi)}개 기록")
    return 0


if __name__ == "__main__":
    sys.exit(main())
