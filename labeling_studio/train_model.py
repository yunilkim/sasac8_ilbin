#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 train_model.py  --  라벨셋 하나로 모델을 학습하고, 그 내력을 함께 남긴다
================================================================================

 실행:
     .venv\\Scripts\\python.exe train_model.py --labels manual_bbox --dry-run
     .venv\\Scripts\\python.exe train_model.py --labels manual_bbox
     .venv\\Scripts\\python.exe train_model.py --labels gt --dataset vest-helmet_crop_dedup \\
         --n 150 --seed 0 --name y11s_n150_dedup

--------------------------------------------------------------------------------
 [ 왜 필요한가 ]
--------------------------------------------------------------------------------
 이 프로젝트에서 모델은 결과물이 아니라 계측 결과다. "150장 라벨하면 66% 절감"
 같은 문장은 그 150장이 어디서 왔고 누가 만들었는지가 붙어 있어야만 뜻이 있다.

 그래서 이 도구는 가중치만 뱉지 않는다. models/<이름>/model.json 에 다음을 남긴다.

   · 학습에 쓴 이미지 목록 전부 (train_image_list)
   · 라벨의 출처 분해 -- A 사람 / B GT / C 학습모델 / D zero-shot 각 몇 박스
   · 그 라벨을 만드는 데 사람이 실제로 쓴 시간 (작업공간 기록에서 합산)
   · 평가셋과의 중복 검사 결과
   · 점수와 가중치 해시

 마지막 두 줄이 핵심이다. "사람 N분 들여 mAP M" 이 한 줄로 읽혀야 다음 라운드에
 몇 장을 더 할지 정할 수 있다.

--------------------------------------------------------------------------------
 [ 라벨은 어디서 오는가 ]
--------------------------------------------------------------------------------
 --labels <작업공간이름>   work/<이름>/labels/<split>/ 의 사람 편집 결과. 정규 경로.
 --labels gt              db 데이터셋의 정답지. 시뮬레이션 전용이며 명시해야만 쓴다.

 GT 를 자동 라벨러의 학습 데이터로 쓰는 것은 이 프로젝트의 전제 위반이다
 (정답지를 먹고 정답을 맞히는 순환이 되고, 라벨 없는 새 데이터셋에는 아무것도
 해주지 못한다). 사람 주석자 대역으로 쓰는 시뮬레이션만 허용되며, 그때도
 label_source 에 반드시 기록된다. 실수로 섞이는 일이 없도록 기본값은 거부다.

--------------------------------------------------------------------------------
 [ 두 가지 함정과 그에 대한 방어 ]
--------------------------------------------------------------------------------
 1. 검수 안 한 이미지를 '객체 없음'으로 학습시키는 것
    작업공간에서 아직 손대지 않은 이미지는 라벨 파일이 없다. 그걸 그냥 넘기면
    YOLO 는 "이 사진에는 아무것도 없다"로 배운다 -- 사실이 아니다.
    → 검수 기록(meta)이 있는 이미지만 쓴다. 빈 라벨 파일은 '봤는데 없었다'는
      뜻이므로 그대로 쓴다. 이 둘은 완전히 다르다.

 2. 평가셋에 같은 사진이 있는 채로 학습하는 것
    2026-08-16 에 valid 의 12.0%, test 의 10.9% 가 train 에 같은 사진을 갖고
    있는 것이 드러났다. 바이트가 같은 파일은 0장이라 해시 검사로는 안 잡힌다.
    → 학습 직전에 선택된 학습 이미지와 평가 이미지를 dHash 로 대조한다.
      데이터셋에 dedup_manifest.json 이 있으면 그 결과를 신뢰하고 건너뛴다.

--------------------------------------------------------------------------------
 [ 배치 ]
--------------------------------------------------------------------------------
 models/<이름>/
     weights/best.pt · last.pt      ultralytics 산출물
     args.yaml · results.csv        ultralytics 산출물
     data/train/images|labels/      학습용 임시 배치 (이미지는 하드링크 = 0바이트)
     data/val/images|labels/
     dataset.yaml                   ultralytics 에 넘긴 설정
     model.json                     이 도구가 남기는 내력
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
from collections import Counter
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

DATA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA)

import yolo_dataset_studio as S  # noqa: E402

DEFAULT_BASE = "yolo11s.pt"
DEFAULT_EPOCHS = 80
DEFAULT_IMGSZ = 640

# 중복 판정 기준 -- dedup_dataset.py 와 같은 값을 쓴다. 두 도구가 다른 잣대를
# 쓰면 "dedup 은 통과했는데 여기선 걸린다" 같은 일이 생긴다.
HAMMING = 5
MAD_MAX = 18.0


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_hms(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분 {s}초" if h else (f"{m}분 {s}초" if m else f"{s}초")


# ------------------------------------------------------------------ 라벨 공급원
class LabelSource:
    """학습에 쓸 라벨을 어디서 가져올지에 대한 공통 창구.

    작업공간과 GT 는 '어떤 이미지를 쓸 수 있는가' 의 뜻이 근본적으로 다르다.
    작업공간은 검수한 것만 쓸 수 있고, GT 는 전부 쓸 수 있되 라벨 파일이 없으면
    배경(객체 없음)을 뜻한다. 그 차이를 여기서 흡수한다.
    """

    def __init__(self, kind: str, name: str):
        self.kind = kind          # "workspace" | "gt"
        self.name = name

    def usable(self, split: S.Split, split_name: str) -> List[str]:
        raise NotImplementedError

    def label_path(self, split: S.Split, split_name: str,
                   fn: str) -> Optional[str]:
        raise NotImplementedError

    def provenance(self, split_name: str, files: List[str]) -> dict:
        """출처 태그별 박스 수와 사람이 쓴 시간. 모르면 빈 dict."""
        return {}


class WorkspaceLabels(LabelSource):
    def __init__(self, ws: "S.Workspace"):
        super().__init__("workspace", ws.name)
        self.ws = ws

    def usable(self, split: S.Split, split_name: str) -> List[str]:
        # 검수 기록이 있는 것만. 라벨 파일 유무로 판단하면 '아직 안 본 이미지'와
        # '보고 나니 객체가 없던 이미지'가 구별되지 않는다.
        done = {fn for (sp, fn) in self.ws.records if sp == split_name}
        return [f for f in split.image_files if f in done]

    def label_path(self, split, split_name, fn):
        p = self.ws.label_path(split_name, fn)
        return p if os.path.isfile(p) else None

    def provenance(self, split_name: str, files: List[str]) -> dict:
        want = set(files)
        tags: Counter = Counter()
        seconds = 0.0
        ops = S.EditOps()
        n = 0
        for (sp, fn), rec in self.ws.records.items():
            if sp != split_name or fn not in want:
                continue
            n += 1
            for t in (rec.get("src") or []):
                tags[t] += 1
            o = rec.get("ops") or {}
            seconds += float(o.get("seconds") or 0.0)
            ops.add(S.EditOps.from_dict(o))
        return {"images_with_record": n, "boxes_by_tag": dict(tags),
                "human_seconds": round(seconds, 1),
                "ops": ops.to_dict()}


class GtLabels(LabelSource):
    def __init__(self, dataset_name: str):
        super().__init__("gt", f"GT:{dataset_name}")

    def usable(self, split: S.Split, split_name: str) -> List[str]:
        return list(split.image_files)

    def label_path(self, split, split_name, fn):
        return split.label_path_for(fn)   # 없으면 None = 배경 이미지


# ------------------------------------------------------------------ 중복 검사
def leak_check(dataset_root: str, train_imgs: List[Tuple[str, str]],
               val_imgs: List[Tuple[str, str]], project: S.Project) -> dict:
    """학습 이미지와 평가 이미지에 같은 사진이 있는가.

    데이터셋에 dedup_manifest.json 이 있으면 이미 정리된 것이므로 그 사실만
    기록하고 넘어간다. 없으면 직접 센다 -- 선택된 학습분만 보면 되므로
    (학습 150 × 평가 1258) 정도라 몇 초면 끝난다.
    """
    mf = os.path.join(dataset_root, "dedup_manifest.json")
    if os.path.isfile(mf):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
            st = d.get("stats", {})
            return {"method": "dedup_manifest", "manifest": mf,
                    "cross_groups_left": st.get("cross_groups_left"),
                    "dropped": st.get("dropped"),
                    "pairs": 0,
                    "note": "이 데이터셋은 dedup_dataset.py 로 정리되어 있다"}
        except (OSError, ValueError):
            pass

    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"method": "skipped", "reason": "cv2/numpy 없음"}

    pop = np.array([bin(i).count("1") for i in range(256)], np.uint8)

    def dhash(p: str) -> int:
        try:
            buf = np.fromfile(p, dtype=np.uint8)
        except OSError:
            return -1
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None
        if im is None:
            return -1
        g = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
        v = 0
        for b in (g[:, 1:] > g[:, :-1]).flatten():
            v = (v << 1) | int(b)
        return v

    small: dict = {}

    def sm(p: str):
        if p not in small:
            buf = np.fromfile(p, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            small[p] = cv2.resize(im, (64, 64),
                                  interpolation=cv2.INTER_AREA).astype(np.float32)
        return small[p]

    def paths(items):
        out = []
        for sp_name, fn in items:
            sp = project.split_by_name(sp_name)
            if sp:
                out.append(os.path.join(sp.images_dir, fn))
        return out

    tp, vp = paths(train_imgs), paths(val_imgs)
    th = np.array([dhash(p) for p in tp], dtype=np.uint64)
    vh = np.array([dhash(p) for p in vp], dtype=np.uint64)
    hits = []
    for i in range(len(th)):
        x = th[i] ^ vh
        d = pop[x.view(np.uint8).reshape(-1, 8)].sum(1)
        for j in np.nonzero(d <= HAMMING)[0]:
            j = int(j)
            if float(np.abs(sm(tp[i]) - sm(vp[j])).mean()) < MAD_MAX:
                hits.append({"train": train_imgs[i][1], "val": val_imgs[j][1],
                             "hamming": int(d[j])})
    return {"method": "dhash", "hamming": HAMMING, "mad_max": MAD_MAX,
            "train_checked": len(tp), "val_checked": len(vp),
            "pairs": len(hits), "examples": hits[:20]}


# ------------------------------------------------------------------ 데이터 배치
def stage(dst_root: str, tag: str, project: S.Project, src: LabelSource,
          items: List[Tuple[str, str]]) -> dict:
    """학습/평가용 images|labels 폴더를 만든다.

    이미지는 하드링크한다. 같은 볼륨이면 0바이트이고 원본과 항상 같다.
    (복사하면 수천 장에서 GB 단위가 낭비되고, 원본이 바뀌면 조용히 어긋난다.)
    링크가 안 되는 환경이면 복사로 물러난다.
    """
    idir = os.path.join(dst_root, tag, "images")
    ldir = os.path.join(dst_root, tag, "labels")
    os.makedirs(idir, exist_ok=True)
    os.makedirs(ldir, exist_ok=True)
    linked = copied = empty = 0
    boxes = 0
    for sp_name, fn in items:
        sp = project.split_by_name(sp_name)
        if not sp:
            continue
        s = os.path.join(sp.images_dir, fn)
        d = os.path.join(idir, fn)
        if not os.path.exists(d):
            try:
                os.link(s, d)
                linked += 1
            except OSError:
                shutil.copy2(s, d)
                copied += 1
        lp = src.label_path(sp, sp_name, fn)
        dl = os.path.join(ldir, S.stem(fn) + ".txt")
        if lp and os.path.isfile(lp):
            # 신뢰도가 붙은 라벨(6토큰)은 ultralytics 가 못 읽는다. 앞 5토큰만 남긴다.
            keep = []
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    t = line.split()
                    if len(t) < 5:
                        continue
                    if len(t) == 6:          # cls cx cy w h conf
                        t = t[:5]
                    elif len(t) > 6 and len(t) % 2 == 0:   # 폴리곤 + conf
                        t = t[:-1]
                    keep.append(" ".join(t))
                    boxes += 1
            with open(dl, "w", encoding="utf-8") as f:
                f.write("\n".join(keep) + ("\n" if keep else ""))
            if not keep:
                empty += 1
        else:
            open(dl, "w", encoding="utf-8").close()
            empty += 1
    return {"images": len(items), "linked": linked, "copied": copied,
            "boxes": boxes, "empty_labels": empty,
            "images_dir": idir, "labels_dir": ldir}


# ------------------------------------------------------------------ 본체
def main() -> int:
    ap = argparse.ArgumentParser(
        description="라벨셋 하나로 모델을 학습하고 내력을 남긴다")
    ap.add_argument("--labels", required=True,
                    help="work/ 아래 작업공간 이름, 또는 'gt' (시뮬레이션 전용)")
    ap.add_argument("--dataset", default=None,
                    help="db/ 아래 데이터셋 이름. 작업공간을 쓰면 그쪽 설정을 따름")
    ap.add_argument("--split", default="train", help="학습에 쓸 스플릿")
    ap.add_argument("--val-split", default="valid", help="채점에 쓸 스플릿")
    ap.add_argument("--val-labels", default="gt",
                    help="평가 라벨의 출처. 기본 gt (채점은 GT 로 하는 것이 규칙)")
    ap.add_argument("--val-n", type=int, default=0,
                    help="평가셋을 N 장으로 줄임 (0=전부). 채점까지 같이 줄어든다")
    ap.add_argument("--val-fast-n", type=int, default=0,
                    help="학습 중 검증만 N 장으로 줄임 (0=끄기). 채점은 전량 그대로")
    ap.add_argument("--val-period", type=int, default=1,
                    help="학습 중 검증 주기(에폭). 5 면 5에폭마다. 마지막 에폭은 항상")
    ap.add_argument("--mode", default="random",
                    choices=("random", "stride", "all", "list"))
    ap.add_argument("--n", type=int, default=0, help="학습 장수 (0=쓸 수 있는 전부)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--from-list", default=None, help="파일명 목록 텍스트")
    ap.add_argument("--seed", type=int, default=0, help="선별 시드 겸 학습 시드")
    ap.add_argument("--name", default=None, help="모델 이름 (기본 자동 생성)")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=0,
                    help="조기 종료 (0=끄기). 곡선 비교에는 끄는 편이 공정하다")
    ap.add_argument("--note", default="")
    ap.add_argument("--allow-leak", action="store_true",
                    help="평가셋과 같은 사진이 있어도 강행")
    ap.add_argument("--allow-task-mismatch", action="store_true",
                    help="라벨 종류(box/seg)와 베이스 모델이 안 맞아도 강행")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="계획과 검사 결과만 보고 학습은 안 함")
    a = ap.parse_args()

    lay = S.Layout(DATA)
    if not os.path.isdir(lay.db_dir):
        print(f"[!] db 폴더가 없습니다: {lay.db_dir}")
        return 1

    # ---- 라벨 공급원과 데이터셋 결정 --------------------------------------
    ws: Optional[S.Workspace] = None
    if a.labels.lower() == "gt":
        if not a.dataset:
            print("[!] --labels gt 를 쓰려면 --dataset 을 지정해야 합니다.")
            return 1
        ds_root = os.path.join(lay.db_dir, a.dataset)
        src: LabelSource = GtLabels(a.dataset)
        print("=" * 70)
        print(" 경고 — GT 를 학습 라벨로 사용합니다")
        print("=" * 70)
        print(" 이 프로젝트에서 GT 는 채점 전용이다. 자동 라벨러의 학습 데이터로")
        print(" 쓰면 정답지를 먹고 정답을 맞히는 순환이 되고, 라벨 없는 새")
        print(" 데이터셋에는 아무것도 해주지 못한다.")
        print(" 사람 주석자 대역으로 쓰는 시뮬레이션만 허용되며, 이 실행은")
        print(" model.json 에 label_source='gt' 로 영구히 남는다.")
        print("=" * 70 + "\n")
    else:
        wsp = os.path.join(lay.work_dir, S.safe_name(a.labels))
        if not os.path.isfile(os.path.join(wsp, S.MANIFEST)):
            print(f"[!] 작업공간을 찾을 수 없습니다: {wsp}")
            names = [os.path.basename(d) for d in lay.workspace_dirs()]
            if names:
                print("    있는 것: " + " · ".join(names))
            return 1
        ws = S.Workspace.load(wsp)
        ds_root = os.path.join(lay.db_dir, a.dataset) if a.dataset else ws.source
        src = WorkspaceLabels(ws)

    if not os.path.isdir(ds_root):
        print(f"[!] 데이터셋을 찾을 수 없습니다: {ds_root}")
        return 1
    project = S.Project(ds_root)
    ds_name = os.path.basename(ds_root)

    tr = project.split_by_name(a.split)
    va = project.split_by_name(a.val_split)
    if tr is None:
        print(f"[!] 스플릿 '{a.split}' 이 없습니다. "
              f"있는 것: {[s.name for s in project.splits]}")
        return 1
    if va is None:
        print(f"[!] 평가 스플릿 '{a.val_split}' 이 없습니다.")
        return 1

    # ---- 학습 이미지 고르기 ------------------------------------------------
    pool = src.usable(tr, a.split)
    steps = [f"{src.kind} 에서 쓸 수 있는 {a.split} 이미지 {len(pool)}장"]
    if src.kind == "workspace":
        steps.append(f"  (검수 기록이 있는 것만. 전체 {len(tr.image_files)}장 중)")
    picked, more = S.build_worklist(
        pool, mode=("all" if a.mode == "list" else a.mode), n=a.n,
        stride=a.stride, seed=a.seed,
        from_list=a.from_list if a.mode == "list" else None)
    steps += ["  " + s for s in more]
    if not picked:
        print("[!] 학습에 쓸 이미지가 없습니다.")
        if src.kind == "workspace":
            print("    작업공간에 검수 기록이 없습니다 — 라벨링을 먼저 하세요.")
        return 1

    # ---- 평가 이미지 ------------------------------------------------------
    if a.val_labels.lower() == "gt":
        vsrc: LabelSource = GtLabels(ds_name)
    else:
        vwp = os.path.join(lay.work_dir, S.safe_name(a.val_labels))
        if not os.path.isfile(os.path.join(vwp, S.MANIFEST)):
            print(f"[!] 평가용 작업공간을 찾을 수 없습니다: {vwp}")
            return 1
        vsrc = WorkspaceLabels(S.Workspace.load(vwp))
    vpool = vsrc.usable(va, a.val_split)
    if a.val_n > 0:
        vpool, _ = S.build_worklist(vpool, mode="random", n=a.val_n, seed=a.seed)
    if not vpool:
        print(f"[!] 평가 이미지가 없습니다 ({a.val_split}).")
        return 1

    name = a.name or (f"{os.path.splitext(a.base)[0].replace('yolo', 'y')}"
                      f"_n{len(picked)}_{a.mode}_{S.safe_name(a.labels)}")
    out_dir = os.path.join(DATA, "models", S.safe_name(name))

    # ---- 계획 출력 --------------------------------------------------------
    print(f"모델 이름   {name}")
    print(f"데이터셋    db/{ds_name}")
    print(f"라벨 출처   {src.name}  ({src.kind})")
    print(f"평가 라벨   {vsrc.name}")
    print(f"베이스      {a.base} · {a.epochs}에폭 · imgsz {a.imgsz} · "
          f"batch {a.batch} · seed {a.seed}")
    print("\n[학습 이미지 선별]")
    for s in steps:
        print("  " + s)
    print(f"  → 학습 {len(picked)}장 · 평가 {len(vpool)}장 ({a.val_split})")

    train_items = [(a.split, f) for f in picked]
    val_items = [(a.val_split, f) for f in vpool]

    # ---- 겹침 검사 --------------------------------------------------------
    same = set(picked) & set(vpool) if a.split == a.val_split else set()
    if same:
        print(f"\n[!] 학습셋과 평가셋에 같은 파일 {len(same)}장이 있습니다. 중단합니다.")
        return 1

    print("\n[평가셋 중복 검사]")
    lk = leak_check(ds_root, train_items, val_items, project)
    if lk.get("method") == "dedup_manifest":
        print(f"  dedup_manifest.json 확인 — 이 데이터셋은 정리되어 있습니다 "
              f"(제외 {lk.get('dropped')}장 · 남은 교차무리 "
              f"{lk.get('cross_groups_left')}개)")
    elif lk.get("method") == "dhash":
        print(f"  직접 대조 {lk['train_checked']}×{lk['val_checked']} "
              f"(해밍≤{HAMMING} · MAD<{MAD_MAX})")
        if lk["pairs"]:
            print(f"  [!] 같은 사진 {lk['pairs']}쌍 발견 — 평가셋에 답이 들어 있습니다")
            for h in lk["examples"][:5]:
                print(f"      {h['train'][:40]}  ↔  {h['val'][:40]}")
            if not a.allow_leak:
                print("\n  중단합니다. dedup_dataset.py 로 데이터셋을 먼저 정리하거나,")
                print("  이유가 있다면 --allow-leak 로 강행하세요 (기록에 남습니다).")
                return 1
            print("  --allow-leak 로 강행합니다. model.json 에 기록됩니다.")
        else:
            print("  같은 사진 없음")
    else:
        print(f"  건너뜀 ({lk.get('reason', '?')})")

    # ---- 라벨 내력 --------------------------------------------------------
    prov = src.provenance(a.split, picked)
    if prov:
        tags = prov.get("boxes_by_tag") or {}
        total = sum(tags.values())
        print("\n[라벨 내력]")
        print(f"  박스 {total}개  " +
              " · ".join(f"{k}={v}({v / max(total, 1):.0%})"
                         for k, v in sorted(tags.items())))
        print(f"  사람이 쓴 시간 {fmt_hms(prov.get('human_seconds', 0))}"
              f"  ({prov.get('human_seconds', 0) / max(len(picked), 1):.1f}초/장"
              f" · {prov.get('human_seconds', 0) / max(total, 1):.2f}초/박스)")
        o = prov.get("ops") or {}
        if o:
            print(f"  그리기 {o.get('drawn', 0)} · 삭제 {o.get('deleted', 0)} · "
                  f"조정 {o.get('adjusted', 0)} · 무편집 {o.get('kept', 0)}")

    if a.dry_run:
        print("\ndry-run 이라 여기서 멈춥니다.")
        return 0

    if os.path.isdir(out_dir) and not a.overwrite:
        print(f"\n[!] 이미 있습니다: {out_dir}")
        print("    덮어쓰려면 --overwrite, 아니면 --name 으로 다른 이름을 주세요.")
        return 1
    if os.path.isdir(out_dir) and a.overwrite:
        shutil.rmtree(out_dir, ignore_errors=True)

    # ---- 배치 -------------------------------------------------------------
    t0 = time.perf_counter()
    data_root = os.path.join(out_dir, "data")
    print("\n[배치]")
    st_tr = stage(data_root, "train", project, src, train_items)
    st_va = stage(data_root, "val", project, vsrc, val_items)
    for tag, st in (("train", st_tr), ("val", st_va)):
        print(f"  {tag:<6} 이미지 {st['images']}장 "
              f"(하드링크 {st['linked']} / 복사 {st['copied']}) · "
              f"박스 {st['boxes']} · 빈 라벨 {st['empty_labels']}")
    if st_tr["boxes"] == 0:
        print("\n[!] 학습 라벨에 박스가 하나도 없습니다. 중단합니다.")
        return 1

    # ---- 라벨 종류와 모델 종류가 맞는가 ------------------------------------
    # 이게 어긋나도 ultralytics 는 조용히 돌아간다. seg 라벨을 detect 모델에
    # 주면 폴리곤을 박스로 접어서 학습해 버린다. 에러가 안 나므로 몇 시간 뒤
    # "왜 마스크가 안 나오지" 하고서야 알게 된다. 여기서 막는다.
    kinds = Counter()
    for fn in os.listdir(st_tr["labels_dir"])[:400]:
        with open(os.path.join(st_tr["labels_dir"], fn), encoding="utf-8") as f:
            for line in f:
                t = line.split()
                if len(t) == 5:
                    kinds["box"] += 1
                elif len(t) >= 7 and (len(t) - 1) % 2 == 0:
                    kinds["poly"] += 1
    label_kind = "segment" if kinds["poly"] > kinds["box"] else "detect"
    from ultralytics import YOLO  # noqa: E402
    base_task = YOLO(a.base).task          # 'detect' | 'segment' | ...
    want = "segment" if label_kind == "segment" else "detect"
    print(f"\n[종류] 라벨 {label_kind} (폴리곤 {kinds['poly']} · 박스 {kinds['box']})"
          f" · 베이스 모델 {base_task}")
    if base_task != want:
        print(f"\n[!] 라벨은 {want} 인데 베이스 모델은 {base_task} 입니다.")
        if want == "segment":
            alt = os.path.splitext(a.base)[0] + "-seg.pt"
            print(f"    seg 라벨로 학습하려면 seg 모델이 필요합니다: --base {alt}")
            print(f"    지금 이대로 두면 폴리곤이 박스로 접혀 detect 학습이 됩니다.")
        else:
            print(f"    박스 라벨에 seg 모델을 주면 마스크 손실을 계산할 수 없습니다.")
        if not a.allow_task_mismatch:
            print("    중단합니다. 정말 이대로 하려면 --allow-task-mismatch")
            return 1
        print("    --allow-task-mismatch 로 강행합니다. model.json 에 기록됩니다.")

    def write_yaml(path: str, val_dir: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"path: {data_root}\ntrain: train/images\nval: {val_dir}/images\n")
            f.write(f"nc: {project.nc}\n")
            f.write("names: [" + ", ".join(f"'{n}'" for n in project.names) + "]\n")
        return path

    yml = write_yaml(os.path.join(out_dir, "dataset.yaml"), "val")

    # ---- 학습 중 검증만 따로 (채점은 아래에서 전량으로 다시 한다) ----------
    #  학습 에폭 시간의 80% 가 검증이다(tools/bench_train_time.py). 학습 중 검증은
    #  best.pt 를 고르는 용도라 표본이 전량일 필요가 없고, 보고용 숫자는 어차피
    #  학습이 끝난 뒤 전량으로 다시 잰다.
    train_yml, fast_n = yml, 0
    if 0 < a.val_fast_n < len(vpool):
        fast_names, _ = S.build_worklist(vpool, mode="random", n=a.val_fast_n,
                                         seed=a.seed)
        st_vf = stage(data_root, "val_fast", project, vsrc,
                      [(a.val_split, f) for f in fast_names])
        fast_n = st_vf["images"]
        train_yml = write_yaml(os.path.join(out_dir, "dataset_fast.yaml"), "val_fast")
        print(f"  학습 중 검증 {fast_n}장 (채점은 {len(vpool)}장 전량)")

    # ---- 학습 -------------------------------------------------------------
    print(f"\n[학습] {a.base} · {a.epochs}에폭 · {len(picked)}장"
          + (f" · 검증 {a.val_period}에폭마다" if a.val_period > 1 else ""))
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    model = YOLO(a.base)
    if a.val_period > 1:
        #  ultralytics 8.4 에는 val_period 가 없다. trainer.py 가
        #  `if self.args.val or final_epoch ...` 로 검증 여부를 보고, 그 직전에
        #  on_train_epoch_end 콜백이 돌기 때문에 여기서 토글하면 된다.
        #  마지막 에폭은 final_epoch 절 덕분에 항상 검증된다.
        #  ※ 건너뛴 에폭의 results.csv 행에는 직전 지표가 그대로 반복해 적힌다.
        def _periodic_val(trainer, _p=a.val_period):
            trainer.args.val = ((trainer.epoch + 1) % _p == 0)
        model.add_callback("on_train_epoch_end", _periodic_val)
    model.train(data=train_yml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, workers=a.workers, seed=a.seed,
                patience=a.patience if a.patience > 0 else a.epochs + 1,
                project=os.path.join(DATA, "models"), name=S.safe_name(name),
                exist_ok=True, verbose=True, plots=False)
    train_sec = time.perf_counter() - t0

    best = os.path.join(out_dir, "weights", "best.pt")
    if not os.path.isfile(best):
        print(f"[!] 가중치가 생성되지 않았습니다: {best}")
        return 1
    w_sha = sha256(best)   # 채점 전에 떠 둔다. 채점이 가중치를 건드리면 안 된다

    print("\n[채점]")
    r = YOLO(best).val(data=yml, imgsz=a.imgsz, batch=a.batch, device=a.device,
                       workers=0, verbose=False, plots=False,
                       project=out_dir, name="val", exist_ok=True)
    metrics = {"precision": round(float(r.box.mp), 4),
               "recall": round(float(r.box.mr), 4),
               "mAP50": round(float(r.box.map50), 4),
               "mAP50-95": round(float(r.box.map), 4)}
    per_class = {}
    try:
        for i, ci in enumerate(r.box.ap_class_index):
            per_class[project.names[int(ci)]] = {
                "mAP50": round(float(r.box.ap50[i]), 4),
                "mAP50-95": round(float(r.box.ap[i]), 4)}
    except (AttributeError, IndexError, TypeError):
        pass

    # seg 모델은 박스와 마스크 두 벌의 지표를 낸다. 박스만 적으면 정작
    # 이 모델을 만든 이유인 마스크 성능이 기록에서 빠진다.
    mask_metrics, mask_per_class = {}, {}
    seg = getattr(r, "seg", None)
    if seg is not None:
        mask_metrics = {"precision": round(float(seg.mp), 4),
                        "recall": round(float(seg.mr), 4),
                        "mAP50": round(float(seg.map50), 4),
                        "mAP50-95": round(float(seg.map), 4)}
        try:
            for i, ci in enumerate(seg.ap_class_index):
                mask_per_class[project.names[int(ci)]] = {
                    "mAP50": round(float(seg.ap50[i]), 4),
                    "mAP50-95": round(float(seg.ap[i]), 4)}
        except (AttributeError, IndexError, TypeError):
            pass

    # ---- 기록 -------------------------------------------------------------
    meta = {
        "name": name,
        "created": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base": a.base,
        "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
        "seed": a.seed, "device": str(a.device),
        "source_dataset": f"db/{ds_name}",
        "label_source": src.name,
        "label_source_kind": src.kind,
        "label_source_note": (
            "GT 를 사람 주석자 대역으로 사용한 시뮬레이션 (출처 태그 B)"
            if src.kind == "gt" else
            f"작업공간 work/{src.name} 의 사람 편집 결과"),
        "split": a.split, "selection": a.mode, "trained_images": len(picked),
        "train_boxes": st_tr["boxes"],
        "val_split": a.val_split, "val_images": len(vpool),
        "val_boxes": st_va["boxes"], "val_label_source": vsrc.name,
        # 학습 중 검증 설정. 아래 metrics 는 언제나 val_images 전량으로 다시 잰 값이다.
        "val_fast_n": fast_n, "val_period": a.val_period,
        "metrics": metrics, "per_class": per_class,
        "mask_metrics": mask_metrics, "mask_per_class": mask_per_class,
        "mask_metrics_note": (
            "마스크 지표는 SAM 이 만든 라벨을 정답으로 삼아 잰 것이다. "
            "'진짜 분할 성능'이 아니라 'SAM 의 마스크를 얼마나 재현하는가'다."
            if mask_metrics else ""),
        "label_provenance": prov,
        "leak_check": lk,
        "leak_forced": bool(a.allow_leak and lk.get("pairs")),
        "task": base_task, "label_kind": label_kind,
        "task_mismatch_forced": bool(base_task != want),
        "train_seconds": round(train_sec, 1),
        "sha256": w_sha,
        "weights_bytes": os.path.getsize(best),
        # 채점을 돌리는 동안 가중치가 바뀌지 않았는지 확인한다. 예전에 %TEMP% 에서
        # 체크포인트를 옮겨올 때 해시로 검증한 이력이 있어 같은 관례를 잇는다.
        "hash_ok": sha256(best) == w_sha,
        "note": a.note,
        "reminder": ("예측 생성 시 train_image_list 는 반드시 제외할 것 — "
                     "모델이 이미 본 이미지라 자동승인율이 부풀려진다."),
        "train_image_list": picked,
        "val_image_list": vpool,
    }
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # ---- 요약 -------------------------------------------------------------
    hs = (prov or {}).get("human_seconds", 0)
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"  학습 {len(picked)}장 / 박스 {st_tr['boxes']}  ·  "
          f"라벨 출처 {src.name}")
    if hs:
        print(f"  사람 노동 {fmt_hms(hs)}  →  mAP50 {metrics['mAP50']:.4f}")
        print(f"             {hs / 60:.1f}분당 mAP50 "
              f"{metrics['mAP50'] / max(hs / 60, 1e-9):.4f}")
    tag = "박스  " if mask_metrics else "      "
    print(f"  {tag}P {metrics['precision']:.4f} · R {metrics['recall']:.4f} · "
          f"mAP50 {metrics['mAP50']:.4f} · mAP50-95 {metrics['mAP50-95']:.4f}")
    if mask_metrics:
        print(f"  마스크 P {mask_metrics['precision']:.4f} · "
              f"R {mask_metrics['recall']:.4f} · "
              f"mAP50 {mask_metrics['mAP50']:.4f} · "
              f"mAP50-95 {mask_metrics['mAP50-95']:.4f}")
        print(f"         ↑ SAM 이 만든 라벨을 정답으로 삼은 값이다. "
              f"'SAM 재현율'이지 '진짜 분할 성능'이 아니다.")
    for k, v in per_class.items():
        m = mask_per_class.get(k)
        print(f"    {k:<22} 박스 mAP50 {v['mAP50']:.4f}"
              + (f" · 마스크 {m['mAP50']:.4f}" if m else ""))
    print(f"  평가 {len(vpool)}장 ({a.val_split}) · 학습 {fmt_hms(train_sec)}")
    print(f"\n-> {out_dir}")
    print(f"   model.json 에 학습 이미지 목록·출처 분해·중복 검사 기록")
    print(f"   다음: make_preds.py 로 예측 생성 "
          f"(train_image_list {len(picked)}장은 제외할 것)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
