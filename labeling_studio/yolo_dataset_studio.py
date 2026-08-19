#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 YOLO Dataset Studio  --  범용 멀티 프로젝트 YOLO 데이터셋 시각화 / 자동 검증 툴
================================================================================

 의존성 (필수) :  opencv-python, numpy
 의존성 (선택) :  PyYAML   -> 없으면 내장 미니 파서로 data.yaml 파싱
                  Pillow   -> 있으면 캔버스 전송이 2~3배 빨라짐 (없어도 동작)

     pip install opencv-python numpy pyyaml pillow

 실행 :
     python yolo_dataset_studio.py                 # GUI에서 루트 폴더 선택
     python yolo_dataset_studio.py <루트폴더경로>   # 바로 열기


--------------------------------------------------------------------------------
 [ 작업 루트 3구역 구조 ]
--------------------------------------------------------------------------------

 DATA/                          <- [📚 DB 열기] 로 지정하는 작업 루트
 ├── db/                        원본 보관. 이미지 + GT 라벨. 읽기 전용으로 봉인.
 │   └── vest-helmet/
 │       ├── data.yaml
 │       └── train,valid,test/{images,labels}
 │
 ├── work/                      라벨 작업공간. 이미지는 없고 db 를 참조한다.
 │   ├── manual_150/
 │   │   ├── workspace.json     소스 참조 + 설정 + 계보(부모 작업공간/모델)
 │   │   ├── labels/train/*.txt 편집 결과 (순수 YOLO = 그대로 학습 가능)
 │   │   ├── meta/train/*.json  출처 태그 / 편집량 / 소요시간
 │   │   └── report.html        보고서
 │   ├── round1_yolo11s/        같은 이미지에 대한 다른 라벨셋
 │   └── seg_sam/
 │
 └── export/                    학습용으로 조립해 내보낸 결과
     └── 20260815_1430_vest-helmet/
         ├── data.yaml
         ├── export_manifest.json   어느 작업공간에서 무엇이 왔는지 전부 기록
         └── train,valid,test/{images,labels}

 이미지는 db 에만 있다. 작업공간을 몇 개 만들어도 늘어나는 건 라벨 텍스트뿐이고,
 내보낼 때만 이미지가 실제로 복사된다. db 에 쓰는 코드는 이 파일 어디에도 없다.

--------------------------------------------------------------------------------
 [ 인식하는 데이터셋 폴더 형태 ]
--------------------------------------------------------------------------------

 이 툴은 "images/ 와 labels/ 가 형제(sibling)로 존재하는 모든 디렉터리"를
 루트 아래 depth 3까지 재귀 탐색해서 자동으로 스플릿으로 등록한다.
 따라서 아래 형태들이 전부 그대로 열린다.

 (A) Roboflow / Ultralytics 표준  <- 현재 vest-helmet 데이터셋이 이 형태
 ────────────────────────────────────────────────
 1_Helmet/                     <- 이 폴더를 [루트 열기]로 선택
 ├── data.yaml                 <- nc, names 파싱 (클래스명/색상 동적 매핑)
 ├── train/
 │   ├── images/  aaa.jpg  bbb.jpg ...
 │   └── labels/  aaa.txt  bbb.txt ...      (stem 기준 자동 매칭)
 ├── valid/                    <- 'val' 또는 'valid' 둘 다 인식
 │   ├── images/
 │   └── labels/
 └── test/
     ├── images/
     └── labels/

 (B) 스플릿 없는 단일 폴더
 ────────────────────────────────────────────────
 3_Fruit/
 ├── data.yaml   (없어도 됨 -> class_0, class_1 ... 로 자동 명명)
 ├── images/
 └── labels/

 (C) 멀티 프로젝트 워크스페이스 (한 단계 위를 열어도 됨)
 ────────────────────────────────────────────────
 DATA/                         <- 여기를 열면 하위 프로젝트가 전부 스플릿으로 잡힘
 ├── 1_Helmet/{train,valid,test}/{images,labels}
 ├── 2_Gun/{train,valid}/{images,labels}
 └── 3_Fruit/{images,labels}

 (D) 라벨 파일 포맷 (두 가지 모두 렌더링)
 ────────────────────────────────────────────────
   Detection    : <cls> <cx> <cy> <w> <h>                       (정규화 0~1)
   Segmentation : <cls> <x1> <y1> <x2> <y2> ... <xn> <yn>       (정규화 0~1)
   빈 파일(0바이트) = background 이미지로 간주 (에러 아님, INFO 처리)


--------------------------------------------------------------------------------
 [ 단축키 ]
--------------------------------------------------------------------------------
   ← / →   또는 W/S  이전 / 다음 이미지  (키 홀드시 초고속 스킵, 논블로킹)
                     W/S 는 마우스를 잡은 채 왼손만으로 넘기기 위한 것
   PageUp/PageDown  ±10 장
   Home / End       처음 / 마지막
   Ctrl+G           인덱스로 점프
   L                라벨 박스 on/off
   N                클래스 이름 텍스트 on/off
   F                박스 내부 반투명 채우기 on/off
   Space            현재 이미지 플래그(검수 대상) 토글
   R                뷰 리셋(fit)
   마우스 휠         줌 인/아웃 (커서 기준)
   마우스 드래그      팬(이동)

 [ 라벨 편집 (라운드 세션이 열려 있을 때만) ]
   E                편집 모드 on/off
   G                예측 오버레이 on/off
   드래그(빈 곳)      새 박스 그리기
   드래그(박스 안)    이동   /  모서리 핸들 드래그 = 크기 조절
   Ctrl+드래그       편집 모드에서도 강제 팬 (중클릭 드래그도 동일)
   더블클릭(예측)     그 예측을 라벨로 승격
   ~                지우개 도구 on/off — 드래그로 감싼 라벨을 지운다.
                    판정은 '박스 중심이 안에 들어왔는가'. 숫자키를 누르면 그리기로 복귀.
                    도구는 키보드 맨 윗줄 그대로다: ~ 지우개 · 1 첫째 · 2 둘째 …
   1~9, 0           그릴 클래스 지정 — 1=첫째 클래스 … 9=아홉째, 0=열째
                    작업 흐름은 "클래스를 먼저 정하고 → 그 클래스로 계속 그리기".
                    현재 클래스는 상태바에 항상 표시된다.
                    박스를 클릭해 선택한 상태면 그 박스의 클래스도 함께 바뀐다.
                    (새로 그린 박스는 선택되지 않으므로 오염되지 않는다)
   Delete           선택 박스 삭제
   Tab              라벨·예측을 통째로 감췄다 되돌리기 (원본 확인용)
   Shift+Tab / Esc  다음 박스 선택 / 선택 해제
   Ctrl+A           이 이미지의 라벨을 전부 지우고 빈 화면에서 다시 그리기
                    약한 모델의 시드는 고치는 것보다 지우는 편이 쌀 때가 많다
                    (135장 실측: v1 시드 5.1초/박스 vs 빈 화면 3.4초/박스).
                    되돌리기 가능. 비운 채 넘어가면 '객체 없음'으로 저장된다.
   Ctrl+Shift+A     남은 예측을 전부 라벨로 승격 (모델이 셀 때만 쓸 것)
   Ctrl+Z / Ctrl+S  되돌리기 / 저장
   Enter            저장하고 아직 안 한 다음 이미지로

 편집 결과는 db/ 가 아니라 work/<작업공간>/labels/<스플릿>/ 에 저장된다.
 db 의 라벨은 채점 기준이므로 이 툴에는 그것을 덮어쓰는 코드 경로 자체가 없다.
================================================================================
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("[FATAL] opencv-python 이 필요합니다:  pip install opencv-python")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---- 선택적 가속기 --------------------------------------------------------
try:
    import yaml  # PyYAML

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from PIL import Image, ImageTk  # Pillow

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLIT_ORDER = {"train": 0, "valid": 1, "val": 1, "test": 2, "root": 3}


# ==============================================================================
# MODULE 1 : io_utils  --  유니코드 안전 파일 I/O
# ==============================================================================
def imread_u(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread 는 Windows 에서 한글/유니코드 경로를 못 읽는다. 그 우회 버전."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ==============================================================================
# MODULE 2 : project  --  data.yaml 파싱 + 폴더 구조 자동 스캔
# ==============================================================================
def _mini_yaml_parse(text: str) -> dict:
    """PyYAML 없이도 data.yaml 의 nc / names 만큼은 확실히 뽑아내는 최소 파서."""
    out: dict = {}
    names: List[str] = []
    in_names_block = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if in_names_block:
            s = line.strip()
            if line.startswith((" ", "\t")) and s.startswith("- "):
                names.append(s[2:].strip().strip("'\""))
                continue
            if line.startswith((" ", "\t")) and ":" in s:  # {0: a, 1: b} 블록형
                names.append(s.split(":", 1)[1].strip().strip("'\""))
                continue
            in_names_block = False
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key == "names":
            if not val:
                in_names_block = True
            elif val.startswith("["):
                inner = val.strip("[]")
                names = [t.strip().strip("'\"") for t in inner.split(",") if t.strip()]
            elif val.startswith("{"):
                inner = val.strip("{}")
                names = [
                    t.split(":", 1)[1].strip().strip("'\"")
                    for t in inner.split(",")
                    if ":" in t
                ]
        elif key in ("nc", "train", "val", "test", "path"):
            out[key] = val.strip("'\"")
    if names:
        out["names"] = names
    return out


def class_color(idx: int) -> Tuple[int, int, int]:
    """황금비 hue 분산 -> 클래스 수가 몇 개든 서로 잘 구분되는 BGR 색 생성."""
    h = int(((idx * 0.6180339887) % 1.0) * 179)
    s = 235 if idx % 2 == 0 else 200
    hsv = np.uint8([[[h, s, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


@dataclass
class Split:
    """images/ + labels/ 쌍 하나 = 하나의 스플릿."""

    name: str
    images_dir: str
    labels_dir: str
    image_files: List[str] = field(default_factory=list)  # 파일명만 저장(메모리 절약)
    label_stems: Dict[str, str] = field(default_factory=dict)  # stem -> 라벨 파일명

    def image_path(self, i: int) -> str:
        return os.path.join(self.images_dir, self.image_files[i])

    def label_path_for(self, image_name: str) -> Optional[str]:
        fn = self.label_stems.get(stem(image_name))
        return os.path.join(self.labels_dir, fn) if fn else None

    def __len__(self) -> int:
        return len(self.image_files)


class Project:
    """루트 폴더 하나 = 프로젝트 하나. 클래스 메타 + 스플릿 목록을 보유."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root) or self.root
        self.names: List[str] = []
        self.nc: int = 0
        self.yaml_path: Optional[str] = None
        self.splits: List[Split] = []
        self.colors: List[Tuple[int, int, int]] = []
        self._load_yaml()
        self._scan_splits()
        self._finalize_classes()

    # -- data.yaml ---------------------------------------------------------
    def _load_yaml(self) -> None:
        for cand in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
            p = os.path.join(self.root, cand)
            if os.path.isfile(p):
                self.yaml_path = p
                break
        if not self.yaml_path:
            # 한 단계 아래 프로젝트들의 yaml 중 첫 번째라도 참고
            for entry in sorted(os.scandir(self.root), key=lambda e: e.name):
                if entry.is_dir():
                    p = os.path.join(entry.path, "data.yaml")
                    if os.path.isfile(p):
                        self.yaml_path = p
                        break
        if not self.yaml_path:
            return
        try:
            with open(self.yaml_path, "r", encoding="utf-8-sig", errors="replace") as f:
                text = f.read()
            data = yaml.safe_load(text) if _HAS_YAML else _mini_yaml_parse(text)
            if not isinstance(data, dict):
                data = _mini_yaml_parse(text)
            raw_names = data.get("names")
            if isinstance(raw_names, dict):
                self.names = [str(raw_names[k]) for k in sorted(raw_names, key=int)]
            elif isinstance(raw_names, (list, tuple)):
                self.names = [str(n) for n in raw_names]
            try:
                self.nc = int(data.get("nc", len(self.names)))
            except (TypeError, ValueError):
                self.nc = len(self.names)
        except Exception:
            traceback.print_exc()

    # -- 폴더 스캔 ---------------------------------------------------------
    def _scan_splits(self) -> None:
        found: List[Tuple[str, str, str]] = []

        def walk(d: str, depth: int) -> None:
            if depth > 3:
                return
            try:
                entries = [e for e in os.scandir(d) if e.is_dir()]
            except OSError:
                return
            names = {e.name.lower(): e.path for e in entries}
            if "images" in names and "labels" in names:
                rel = os.path.relpath(d, self.root).replace("\\", "/")
                label = "root" if rel == "." else rel
                found.append((label, names["images"], names["labels"]))
                return  # 더 내려가지 않음
            for e in entries:
                walk(e.path, depth + 1)

        walk(self.root, 0)
        found.sort(key=lambda t: (SPLIT_ORDER.get(t[0].split("/")[-1].lower(), 9), t[0]))

        for label, idir, ldir in found:
            sp = Split(label, idir, ldir)
            sp.image_files = sorted(
                e.name
                for e in os.scandir(idir)
                if e.is_file() and e.name.lower().endswith(IMG_EXTS)
            )
            sp.label_stems = {
                stem(e.name): e.name
                for e in os.scandir(ldir)
                if e.is_file() and e.name.lower().endswith(".txt")
            }
            if sp.image_files or sp.label_stems:
                self.splits.append(sp)

    # -- 클래스 확정 -------------------------------------------------------
    def _finalize_classes(self) -> None:
        if not self.names:
            n = max(self.nc, self._probe_max_class() + 1, 1)
            self.names = [f"class_{i}" for i in range(n)]
        self.nc = len(self.names)
        self.colors = [class_color(i) for i in range(self.nc)]

    def _probe_max_class(self) -> int:
        """yaml 이 없을 때: 라벨 파일 200개만 훑어 최대 class id 추정."""
        mx, cnt = -1, 0
        for sp in self.splits:
            for fn in list(sp.label_stems.values())[:200]:
                try:
                    with open(os.path.join(sp.labels_dir, fn), "r") as f:
                        for line in f:
                            t = line.split()
                            if t:
                                mx = max(mx, int(float(t[0])))
                except Exception:
                    pass
                cnt += 1
                if cnt >= 200:
                    return mx
        return mx

    def split_by_name(self, name: str) -> Optional[Split]:
        for sp in self.splits:
            if sp.name == name:
                return sp
        return None

    def class_name(self, cid: int) -> str:
        return self.names[cid] if 0 <= cid < self.nc else f"?{cid}"

    def color_of(self, cid: int) -> Tuple[int, int, int]:
        return self.colors[cid] if 0 <= cid < self.nc else (0, 0, 255)

    @property
    def total_images(self) -> int:
        return sum(len(s) for s in self.splits)


# ==============================================================================
# MODULE 3 : labels  --  YOLO txt 파싱 (detection + segmentation 겸용)
# ==============================================================================
@dataclass
class Shape:
    cls: int
    kind: str  # "box" | "poly"
    # box  : (cx, cy, w, h) 정규화
    # poly : (N,2) 정규화 점들
    box: Optional[Tuple[float, float, float, float]] = None
    poly: Optional[np.ndarray] = None
    raw_line: str = ""
    line_no: int = 0
    # ---- 자동 라벨링 파이프라인용 메타 (MODULE 9 참조) ----------------------
    conf: float = -1.0  # 예측 신뢰도. -1 = 신뢰도 없음(사람 라벨/GT)
    src: str = ""  # 출처 태그 A=사람 B=GT C=학습모델 D=zero-shot (권위 순)
    tier: str = ""  # "auto" | "review" | "reject"  (conf 로부터 파생)

    def clone(self) -> "Shape":
        """언두 스택용 깊은 복사 (poly 배열까지 분리)."""
        return Shape(
            self.cls, self.kind, self.box,
            None if self.poly is None else self.poly.copy(),
            self.raw_line, self.line_no, self.conf, self.src, self.tier,
        )


def parse_label_file(
    path: Optional[str], default_src: str = ""
) -> Tuple[List[Shape], List[str]]:
    """반환: (shapes, 파싱 경고 목록)

    ultralytics 의 `save_txt=True, save_conf=True` 출력처럼 마지막에 신뢰도가 한 개
    더 붙은 줄(box=5토큰, poly=홀수토큰)도 그대로 읽는다. 예측 폴더를 오버레이로
    띄우려면 이 형식을 못 읽으면 안 된다.
    """
    shapes: List[Shape] = []
    warns: List[str] = []
    if not path or not os.path.isfile(path):
        return shapes, warns
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return shapes, [f"읽기 실패: {e}"]

    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        tok = line.split()
        try:
            cid = int(float(tok[0]))
            vals = [float(t) for t in tok[1:]]
        except ValueError:
            warns.append(f"L{i}: 숫자 파싱 불가 -> {line[:60]}")
            continue

        conf = -1.0
        if len(vals) == 5 or (len(vals) >= 7 and len(vals) % 2 == 1):
            conf = vals[-1]  # 꼬리 신뢰도
            vals = vals[:-1]

        if len(vals) == 4:
            shapes.append(Shape(cid, "box", box=tuple(vals), raw_line=line, line_no=i,
                                conf=conf, src=default_src))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
            shapes.append(Shape(cid, "poly", poly=pts, raw_line=line, line_no=i,
                                conf=conf, src=default_src))
        else:
            warns.append(f"L{i}: 토큰 수 이상({len(tok)}개) -> {line[:60]}")
    return shapes, warns


def poly_bbox(pts: np.ndarray) -> Tuple[float, float, float, float]:
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def shape_xyxy(sh: Shape) -> Tuple[float, float, float, float]:
    """도형 종류와 무관하게 정규화 (x0,y0,x1,y1) 을 준다. poly 는 외접 사각형."""
    cx, cy, w, h = sh.box if sh.kind == "box" else poly_bbox(sh.poly)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def xyxy_to_cxcywh(
    x0: float, y0: float, x1: float, y1: float
) -> Tuple[float, float, float, float]:
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def shape_iou(a: Shape, b: Shape) -> float:
    return iou_xyxy(shape_xyxy(a), shape_xyxy(b))


HANDLE_ORDER = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


def handle_points(
    x0: float, y0: float, x1: float, y1: float
) -> Dict[str, Tuple[float, float]]:
    """리사이즈 핸들 8개의 좌표. 렌더러와 히트 테스트가 반드시 이 함수를 공유해야
    '보이는 곳'과 '잡히는 곳'이 어긋나지 않는다."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    return {
        "nw": (x0, y0), "n": (mx, y0), "ne": (x1, y0), "e": (x1, my),
        "se": (x1, y1), "s": (mx, y1), "sw": (x0, y1), "w": (x0, my),
    }


# ==============================================================================
# MODULE 4 : cache  --  LRU 캐시 + 비동기 프리페처 (UI 논블로킹의 핵심)
# ==============================================================================
class LRUCache:
    def __init__(self, capacity: int = 96):
        self.capacity = capacity
        self._d: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key: str, val: np.ndarray) -> None:
        with self._lock:
            self._d[key] = val
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


class AsyncImageLoader:
    """
    cv2.imdecode 는 GIL 을 놓기 때문에 threading 만으로도 실질 병렬 디코딩이 된다.
    - request(): 지금 당장 보여줘야 하는 이미지 (완료시 결과 큐로 통보)
    - prefetch(): 앞뒤 이미지 미리 로딩 (통보 없음, 캐시에만 적재)
    - token: 사용자가 방향키를 연타해 이미 지나가버린 요청은 UI 에서 폐기
    """

    def __init__(self, cache: LRUCache, workers: int = 4):
        self.cache = cache
        self.result_q: "queue.Queue[Tuple[int, str, Optional[np.ndarray]]]" = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="imgload")
        self._inflight: set = set()
        self._lock = threading.Lock()
        self._closed = False

    def _load(self, path: str) -> Optional[np.ndarray]:
        img = self.cache.get(path)
        if img is None:
            img = imread_u(path)
            if img is not None:
                self.cache.put(path, img)
        return img

    def request(self, path: str, token: int) -> Optional[np.ndarray]:
        """캐시 히트면 즉시 반환(동기), 미스면 None 반환 후 백그라운드 로딩."""
        img = self.cache.get(path)
        if img is not None:
            return img
        if self._closed:
            return None

        def job() -> None:
            try:
                out = self._load(path)
            except Exception:
                out = None
            finally:
                with self._lock:
                    self._inflight.discard(path)
            self.result_q.put((token, path, out))

        self._pool.submit(job)
        return None

    def prefetch(self, paths: Iterable[str]) -> None:
        if self._closed:
            return
        for p in paths:
            if self.cache.get(p) is not None:
                continue
            with self._lock:
                if p in self._inflight:
                    continue
                self._inflight.add(p)
            self._pool.submit(self._prefetch_job, p)

    def _prefetch_job(self, path: str) -> None:
        try:
            self._load(path)
        except Exception:
            pass
        finally:
            with self._lock:
                self._inflight.discard(path)

    def shutdown(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False)


# ==============================================================================
# MODULE 5 : render  --  OpenCV 렌더러 (crop -> resize -> draw 순서로 최적화)
# ==============================================================================
@dataclass
class ViewState:
    """scale = 화면픽셀/이미지픽셀, (cx,cy) = 캔버스 중앙에 오는 이미지 좌표."""

    scale: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    fitted: bool = True


@dataclass
class RenderOpts:
    show_shapes: bool = True
    show_names: bool = True
    fill: bool = False
    thickness: int = 2
    hidden_classes: set = field(default_factory=set)
    highlight: int = -1  # 인스펙터에서 선택한 shape 인덱스
    # ---- 편집 모드 ---------------------------------------------------------
    edit_mode: bool = False
    show_ghosts: bool = True  # 예측 오버레이 표시
    show_meta: bool = True  # 라벨 옆 출처/신뢰도 배지


# 티어 색상 (BGR). 신뢰도 3분류 시각화의 유일한 출처.
TIER_COLORS = {
    "auto": (110, 220, 120),  # 초록 - 자동승인
    "review": (70, 190, 255),  # 주황 - 검토 필요
    "reject": (150, 150, 155),  # 회색 - 기각(참고용)
}
HANDLE_R = 4  # 리사이즈 핸들 반경(캔버스 px)
HANDLE_HIT = 7  # 핸들 히트 판정 반경(캔버스 px)


@dataclass
class Overlay:
    """편집 대상 위에 겹쳐 보여주는 읽기 전용 층.

    예측·마스크·다른 작업공간의 라벨을 원하는 만큼 얹을 수 있게 하기 위한 것.
    층마다 색과 스타일이 달라야 무엇이 무엇인지 구분된다.
    """

    shapes: Sequence[Shape] = ()
    color: Optional[Tuple[int, int, int]] = None  # None = 티어색(예측) / 클래스색
    filled: bool = False   # 폴리곤을 반투명하게 채운다 (마스크 확인용)
    dashed: bool = True    # 박스를 점선으로 (확정 라벨과 구분)
    width: int = 1
    label: bool = True     # 클래스명·신뢰도 표시


class Renderer:
    """이미지 전체를 리사이즈하지 않는다. 화면에 보이는 영역만 crop 후 resize.
    -> 8000px 이미지를 400% 줌해도 렌더 비용이 캔버스 크기에 비례해서 일정하다."""

    BG = (32, 32, 34)

    @staticmethod
    def fit(view: ViewState, img_wh: Tuple[int, int], canvas_wh: Tuple[int, int]) -> None:
        w, h = img_wh
        cw, ch = canvas_wh
        view.scale = min(cw / max(w, 1), ch / max(h, 1))
        view.cx, view.cy = w / 2.0, h / 2.0
        view.fitted = True

    # ---- 좌표 변환 (히트 테스트/편집이 렌더러와 같은 식을 쓰도록 공유) --------
    @staticmethod
    def to_canvas(
        view: ViewState, canvas_wh: Tuple[int, int], px: float, py: float
    ) -> Tuple[float, float]:
        """이미지 픽셀 -> 캔버스 픽셀."""
        cw, ch = canvas_wh
        return (px - view.cx) * view.scale + cw / 2, (py - view.cy) * view.scale + ch / 2

    @staticmethod
    def to_image(
        view: ViewState, canvas_wh: Tuple[int, int], X: float, Y: float
    ) -> Tuple[float, float]:
        """캔버스 픽셀 -> 이미지 픽셀."""
        cw, ch = canvas_wh
        s = max(view.scale, 1e-9)
        return view.cx + (X - cw / 2) / s, view.cy + (Y - ch / 2) / s

    @staticmethod
    def _dashed_rect(
        canvas: np.ndarray, p1, p2, color, thick: int, dash: int = 7
    ) -> None:
        """cv2 엔 점선이 없다. 오버레이(예측)를 확정 라벨과 구분하려면 필요."""
        (x0, y0), (x1, y1) = p1, p2
        for a, b, horiz in ((x0, x1, True), (x0, x1, False)):
            y = y0 if horiz else y1
            x = a
            while x < b:
                xe = min(x + dash, b)
                cv2.line(canvas, (int(x), int(y)), (int(xe), int(y)), color, thick)
                x += dash * 2
        for a, b, left in ((y0, y1, True), (y0, y1, False)):
            x = x0 if left else x1
            y = a
            while y < b:
                ye = min(y + dash, b)
                cv2.line(canvas, (int(x), int(y)), (int(x), int(ye)), color, thick)
                y += dash * 2

    @classmethod
    def render(
        cls,
        img: np.ndarray,
        shapes: Sequence[Shape],
        project: Project,
        view: ViewState,
        canvas_wh: Tuple[int, int],
        opts: RenderOpts,
        ghosts: Sequence[Shape] = (),
        layers: Sequence[Overlay] = (),
    ) -> np.ndarray:
        cw, ch = max(canvas_wh[0], 1), max(canvas_wh[1], 1)
        H, W = img.shape[:2]
        s = view.scale

        canvas = np.empty((ch, cw, 3), dtype=np.uint8)
        canvas[:] = cls.BG

        # --- 화면에 보이는 이미지 영역 계산 ---------------------------------
        vx0 = view.cx - cw / (2 * s)
        vy0 = view.cy - ch / (2 * s)
        x0 = max(0, int(math.floor(vx0)))
        y0 = max(0, int(math.floor(vy0)))
        x1 = min(W, int(math.ceil(vx0 + cw / s)))
        y1 = min(H, int(math.ceil(vy0 + ch / s)))

        if x1 > x0 and y1 > y0:
            crop = img[y0:y1, x0:x1]
            dw = max(1, int(round((x1 - x0) * s)))
            dh = max(1, int(round((y1 - y0) * s)))
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_NEAREST
            resized = cv2.resize(crop, (dw, dh), interpolation=interp)

            dx = int(round((x0 - view.cx) * s + cw / 2))
            dy = int(round((y0 - view.cy) * s + ch / 2))
            # 캔버스 경계로 클리핑해서 붙이기
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            tw = min(dw - sx0, cw - tx0)
            th = min(dh - sy0, ch - ty0)
            if tw > 0 and th > 0:
                canvas[ty0 : ty0 + th, tx0 : tx0 + tw] = resized[
                    sy0 : sy0 + th, sx0 : sx0 + tw
                ]

        # ghosts 는 예측 층의 줄임 표기. layers 로 넘기면 여러 층을 쌓을 수 있다.
        layers = list(layers) or ([Overlay(ghosts)] if ghosts else [])
        has_ghosts = any(ov.shapes for ov in layers) and opts.show_ghosts
        if (not opts.show_shapes or not shapes) and not has_ghosts:
            return canvas

        # --- 도형 그리기 (모두 캔버스 좌표계로 변환 후 1-pass) ---------------
        def to_canvas(nx: float, ny: float) -> Tuple[int, int]:
            X = (nx * W - view.cx) * s + cw / 2
            Y = (ny * H - view.cy) * s + ch / 2
            # cv2 는 int32 범위 밖 좌표에서 비정상 동작 -> 넉넉히 클램프
            return int(max(-30000, min(30000, X))), int(max(-30000, min(30000, Y)))

        thick = max(1, opts.thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- 0단계: 오버레이 층들 (아래에서 위로) ----------------------------
        # 층은 원하는 만큼 쌓을 수 있다. 예측·마스크·다른 작업공간의 라벨이
        # 각자 색과 스타일을 갖고 겹쳐 보인다.
        if has_ghosts:
            fill_buf = None
            for ov in layers:
                if not ov.shapes:
                    continue
                for sh in ov.shapes:
                    if sh.cls in opts.hidden_classes:
                        continue
                    col = ov.color or TIER_COLORS.get(sh.tier,
                                                      TIER_COLORS["reject"])
                    if sh.kind == "poly":
                        # 폴리곤을 외접 사각형으로 그리면 마스크 모양이 사라진다.
                        # 세그 라벨을 겹쳐 보는 것이 이 층의 존재 이유다.
                        pts = np.array([to_canvas(px, py) for px, py in sh.poly],
                                       dtype=np.int32)
                        if pts[:, 0].max() < 0 or pts[:, 1].max() < 0 or \
                           pts[:, 0].min() > cw or pts[:, 1].min() > ch:
                            continue
                        if ov.filled:
                            if fill_buf is None:
                                fill_buf = canvas.copy()
                            cv2.fillPoly(fill_buf, [pts], col)
                        cv2.polylines(canvas, [pts], True, col, ov.width,
                                      cv2.LINE_AA)
                        anchor = (int(pts[:, 0].min()), int(pts[:, 1].min()))
                    else:
                        x0n, y0n, x1n, y1n = shape_xyxy(sh)
                        p1, p2 = to_canvas(x0n, y0n), to_canvas(x1n, y1n)
                        if p2[0] < 0 or p2[1] < 0 or p1[0] > cw or p1[1] > ch:
                            continue
                        if ov.dashed:
                            cls._dashed_rect(canvas, p1, p2, col, ov.width)
                        else:
                            cv2.rectangle(canvas, p1, p2, col, ov.width,
                                          cv2.LINE_8)
                        if ov.filled:
                            if fill_buf is None:
                                fill_buf = canvas.copy()
                            cv2.rectangle(fill_buf, p1, p2, col, -1)
                        anchor = p1
                    if opts.show_meta and ov.label:
                        txt = project.class_name(sh.cls)
                        # ASCII 만 쓴다. cv2.putText 는 문자열을 바이트 단위로
                        # 훑어서 non-ASCII 한 글자가 '?' 여러 개로 찍힌다
                        # (가운데점 · 은 UTF-8 2바이트라 '??' 로 나왔다).
                        txt += (f" {sh.conf:.2f}" if sh.conf >= 0
                                else f" [{sh.src}]")
                        cv2.putText(canvas, txt,
                                    (anchor[0] + 3, max(anchor[1] - 4, 10)),
                                    font, 0.4, col, 1, cv2.LINE_AA)
            if fill_buf is not None:
                cv2.addWeighted(fill_buf, 0.35, canvas, 0.65, 0, dst=canvas)

        if not opts.show_shapes or not shapes:
            return canvas

        # 폴리곤은 언제나 칠해서 보여준다.
        #
        # SAM 마스크는 점이 수천 개까지 나온다(실측 최대 4,874). 그걸 윤곽선으로
        # 그리면 조각 사이를 잇는 내부 선이 전부 보여서 낙서처럼 된다 -- 실제로
        # 그걸 데이터 결함으로 오진했다. 칠해 보면 멀쩡하고, 학습도 칠한 것을
        # 쓴다(polygon2mask -> fillPoly). 그러니 칠한 쪽이 진실에 가깝다.
        # 박스만 있는 화면에서는 종전대로 '채우기' 옵션을 따른다.
        has_poly = any(s.kind != "box" for s in shapes)
        overlay = canvas.copy() if (opts.fill or has_poly) else None

        for idx, sh in enumerate(shapes):
            if sh.cls in opts.hidden_classes:
                continue
            color = project.color_of(sh.cls)
            hot = idx == opts.highlight
            t = thick + 2 if hot else thick

            if sh.kind == "box":
                bcx, bcy, bw, bh = sh.box
                p1 = to_canvas(bcx - bw / 2, bcy - bh / 2)
                p2 = to_canvas(bcx + bw / 2, bcy + bh / 2)
                if p2[0] < 0 or p2[1] < 0 or p1[0] > cw or p1[1] > ch:
                    continue
                cv2.rectangle(canvas, p1, p2, color, t, cv2.LINE_8)
                if overlay is not None and opts.fill:
                    cv2.rectangle(overlay, p1, p2, color, -1)
                # 검토 티어는 "확인 안 된 자동 라벨"이므로 한 겹 더 둘러 표시한다.
                if sh.tier == "review":
                    cv2.rectangle(canvas, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3),
                                  TIER_COLORS["review"], 1, cv2.LINE_8)
                anchor = p1
            else:
                pts = np.array([to_canvas(px, py) for px, py in sh.poly], dtype=np.int32)
                cv2.polylines(canvas, [pts], True, color, t, cv2.LINE_AA)
                if overlay is not None:
                    cv2.fillPoly(overlay, [pts], color)
                anchor = (int(pts[:, 0].min()), int(pts[:, 1].min()))

            if opts.show_names:
                txt = project.class_name(sh.cls)
                if opts.show_meta and sh.src:
                    # ASCII 만. cv2.putText 는 바이트 단위라 '·' 가 '??' 로 찍힌다.
                    txt += f" [{sh.src}"
                    txt += f" {sh.conf:.2f}]" if sh.conf >= 0 else "]"
                (tw_, th_), _ = cv2.getTextSize(txt, font, 0.45, 1)
                bx, by = anchor
                by = max(by, th_ + 6)
                cv2.rectangle(
                    canvas, (bx, by - th_ - 6), (bx + tw_ + 8, by), color, -1, cv2.LINE_8
                )
                cv2.putText(
                    canvas, txt, (bx + 4, by - 4), font, 0.45, (16, 16, 16), 1, cv2.LINE_AA
                )

        if overlay is not None:
            cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, dst=canvas)

        # --- 마지막 단계: 선택 박스의 리사이즈 핸들 (채우기 위에 그려야 보인다) --
        if opts.edit_mode and 0 <= opts.highlight < len(shapes):
            sel = shapes[opts.highlight]
            if sel.kind == "box" and sel.cls not in opts.hidden_classes:
                sx0, sy0, sx1, sy1 = shape_xyxy(sel)
                a, b = to_canvas(sx0, sy0), to_canvas(sx1, sy1)
                for hx, hy in handle_points(a[0], a[1], b[0], b[1]).values():
                    hx, hy = int(hx), int(hy)
                    cv2.rectangle(canvas, (hx - HANDLE_R, hy - HANDLE_R),
                                  (hx + HANDLE_R, hy + HANDLE_R), (20, 20, 20), -1)
                    cv2.rectangle(canvas, (hx - HANDLE_R + 1, hy - HANDLE_R + 1),
                                  (hx + HANDLE_R - 1, hy + HANDLE_R - 1),
                                  (255, 255, 255), -1)
        return canvas


class CanvasBlitter:
    """프레임당 Tk 오버헤드 제거용 더블버퍼.

    매 프레임 PhotoImage 를 새로 만들면 (1) Tk 이미지 객체 할당 (2) canvas.itemconfig
    재바인딩 비용이 함께 붙어 렌더 자체보다 비싸진다. Pillow 가 있으면 캔버스 크기의
    PhotoImage 를 한 번만 만들고 paste() 로 픽셀만 덮어써서 두 비용을 모두 없앤다.
    """

    def __init__(self, canvas: tk.Canvas, item_id: int):
        self.canvas = canvas
        self.item_id = item_id
        self._photo: Optional[tk.PhotoImage] = None
        self._wh: Tuple[int, int] = (0, 0)

    def blit(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if _HAS_PIL:
            im = Image.fromarray(rgb)
            if self._photo is None or self._wh != (w, h):
                # Pillow 12 부터 blank 생성(mode=/size=) 시그니처가 제거됨 -> 실제 Image 로 생성
                self._photo = ImageTk.PhotoImage(im)
                self._wh = (w, h)
                self.canvas.itemconfig(self.item_id, image=self._photo)
                return
            self._photo.paste(im)  # 인플레이스 갱신 (재할당/재바인딩 없음)
        else:
            header = f"P6 {w} {h} 255 ".encode("ascii")
            self._photo = tk.PhotoImage(
                width=w, height=h, data=header + rgb.tobytes(), format="PPM"
            )
            self.canvas.itemconfig(self.item_id, image=self._photo)

    def clear(self) -> None:
        self.canvas.itemconfig(self.item_id, image="")
        self._photo = None
        self._wh = (0, 0)


# ==============================================================================
# MODULE 6 : validation  --  룰 기반 자동 검수 엔진
# ==============================================================================
SEVERITIES = ("ERROR", "WARN", "INFO")


@dataclass
class Issue:
    severity: str  # ERROR / WARN / INFO
    code: str
    split: str
    file: str  # 이미지 파일명 (없으면 라벨 파일명)
    detail: str
    index: int = -1  # 스플릿 내 이미지 인덱스 (더블클릭 점프용)


class RuleValidator:
    """외부 모델 없이 즉시 돌아가는 1차 검수. 백그라운드 스레드에서 실행."""

    EPS = 1e-6
    TINY_AREA = 1e-4  # 전체 면적의 0.01% 미만이면 경고

    def __init__(self, project: Project, deep_image_check: bool = False):
        self.project = project
        self.deep = deep_image_check
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def run(self, progress: Callable[[int, int, str], None]) -> List[Issue]:
        issues: List[Issue] = []
        total = sum(len(sp) + len(sp.label_stems) for sp in self.project.splits)
        done = 0

        for sp in self.project.splits:
            img_stems = {}
            for i, fn in enumerate(sp.image_files):
                img_stems[stem(fn)] = i

            # ---- 이미지 기준 순회 -------------------------------------
            for i, fn in enumerate(sp.image_files):
                if self.cancelled:
                    return issues
                done += 1
                if done % 200 == 0:
                    progress(done, total, f"{sp.name} / {fn}")

                lp = sp.label_path_for(fn)
                if lp is None:
                    issues.append(
                        Issue("ERROR", "NO_LABEL", sp.name, fn, "짝이 되는 .txt 라벨 없음", i)
                    )
                    continue

                try:
                    if os.path.getsize(lp) == 0:
                        issues.append(
                            Issue("INFO", "EMPTY_LABEL", sp.name, fn, "빈 라벨 (background)", i)
                        )
                        continue
                except OSError:
                    pass

                shapes, warns = parse_label_file(lp)
                for w in warns:
                    issues.append(Issue("ERROR", "MALFORMED", sp.name, fn, w, i))
                issues.extend(self._check_shapes(shapes, sp.name, fn, i))

                if self.deep:
                    img = imread_u(sp.image_path(i), cv2.IMREAD_REDUCED_COLOR_8)
                    if img is None:
                        issues.append(
                            Issue("ERROR", "CORRUPT_IMAGE", sp.name, fn, "디코딩 실패/손상", i)
                        )

            # ---- 라벨 기준 고아 탐지 -----------------------------------
            for st, lfn in sp.label_stems.items():
                if self.cancelled:
                    return issues
                done += 1
                if st not in img_stems:
                    issues.append(
                        Issue("ERROR", "NO_IMAGE", sp.name, lfn, "짝이 되는 이미지 없음", -1)
                    )

        progress(total, total, "완료")
        sev_rank = {s: i for i, s in enumerate(SEVERITIES)}
        issues.sort(key=lambda x: (sev_rank.get(x.severity, 9), x.split, x.file))
        return issues

    def _check_shapes(
        self, shapes: List[Shape], split: str, fn: str, idx: int
    ) -> List[Issue]:
        out: List[Issue] = []
        seen: set = set()
        nc = self.project.nc

        for sh in shapes:
            tag = f"L{sh.line_no}"

            if not (0 <= sh.cls < nc):
                out.append(
                    Issue(
                        "ERROR",
                        "BAD_CLASS",
                        split,
                        fn,
                        f"{tag}: class id {sh.cls} 가 범위[0,{nc - 1}] 밖",
                        idx,
                    )
                )

            if sh.kind == "box":
                cx, cy, w, h = sh.box
                coords = {"cx": cx, "cy": cy, "w": w, "h": h}
                bad = [k for k, v in coords.items() if v < -self.EPS or v > 1 + self.EPS]
                if bad:
                    out.append(
                        Issue(
                            "ERROR",
                            "COORD_RANGE",
                            split,
                            fn,
                            f"{tag}: 정규화 범위[0,1] 이탈 {', '.join(bad)} "
                            f"({cx:.4f},{cy:.4f},{w:.4f},{h:.4f})",
                            idx,
                        )
                    )
                if w <= self.EPS or h <= self.EPS:
                    out.append(
                        Issue(
                            "ERROR", "DEGENERATE", split, fn,
                            f"{tag}: w/h 가 0 이하 (w={w:.6f}, h={h:.6f})", idx,
                        )
                    )
                else:
                    x0, y0 = cx - w / 2, cy - h / 2
                    x1, y1 = cx + w / 2, cy + h / 2
                    if x0 < -self.EPS or y0 < -self.EPS or x1 > 1 + self.EPS or y1 > 1 + self.EPS:
                        out.append(
                            Issue(
                                "WARN", "OUT_OF_FRAME", split, fn,
                                f"{tag}: 박스가 이미지 밖으로 벗어남 "
                                f"[{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}]", idx,
                            )
                        )
                    if w * h < self.TINY_AREA:
                        out.append(
                            Issue("WARN", "TINY_BOX", split, fn,
                                  f"{tag}: 면적 {w * h * 100:.4f}% 로 과소", idx)
                        )
                key = (sh.cls, round(cx, 5), round(cy, 5), round(w, 5), round(h, 5))
            else:
                p = sh.poly
                if p.shape[0] < 3:
                    out.append(Issue("ERROR", "DEGENERATE", split, fn,
                                     f"{tag}: 폴리곤 점 개수 {p.shape[0]} < 3", idx))
                if float(p.min()) < -self.EPS or float(p.max()) > 1 + self.EPS:
                    out.append(
                        Issue("ERROR", "COORD_RANGE", split, fn,
                              f"{tag}: 폴리곤 좌표 범위 이탈 "
                              f"(min={float(p.min()):.4f}, max={float(p.max()):.4f})", idx)
                    )
                key = (sh.cls, "poly", p.tobytes())

            if key in seen:
                out.append(Issue("WARN", "DUPLICATE", split, fn, f"{tag}: 완전 중복 라벨", idx))
            seen.add(key)

        return out


# ==============================================================================
# MODULE 7 : ai_hooks  --  AI 확장 인터페이스 (지금은 스텁, 나중에 구현체만 끼우면 됨)
# ==============================================================================
@dataclass
class SampleRef:
    """AI 모듈에 넘겨줄 샘플 1건. 무거운 픽셀 데이터는 lazy 로 읽게 경로만 전달."""

    split: str
    index: int
    image_path: str
    label_path: Optional[str]
    shapes: List[Shape]
    project: Project

    def read_image(self) -> Optional[np.ndarray]:
        return imread_u(self.image_path)


class AnomalyDetector:
    """[확장 포인트 1] AI 이상치 탐지.
    구현체는 analyze_batch() 만 채우면 검수 리포트에 그대로 합류한다."""

    name = "base"
    description = ""

    def available(self) -> Tuple[bool, str]:
        """(사용가능여부, 사유). 모델/패키지 미설치면 (False, 안내문)."""
        return False, "미구현 인터페이스"

    def analyze_batch(
        self, samples: Sequence[SampleRef], progress: Callable[[int, int, str], None]
    ) -> List[Issue]:
        raise NotImplementedError


class ClipConsistencyDetector(AnomalyDetector):
    """[스텁] CLIP 임베딩 기반 라벨-이미지 의미 불일치 탐지.

    구현 가이드:
      1) 각 bbox 를 crop -> CLIP image embedding
      2) 클래스명 프롬프트("a photo of a {name}") -> text embedding
      3) cosine similarity 가 클래스별 분포의 하위 p% 면 MISLABEL 후보로 보고
      4) 추가로 클래스별 임베딩 centroid 로부터 mahalanobis 거리 상위 = 이상치
    """

    name = "CLIP 라벨-이미지 정합성"
    description = "bbox crop 과 클래스명의 의미 유사도가 낮은 샘플 색출"

    def available(self) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import open_clip  # noqa: F401
        except ImportError:
            return False, "torch + open_clip 미설치 (pip install torch open_clip_torch)"
        return False, "백엔드는 준비됨. analyze_batch() 구현 필요"


class YoloDisagreementDetector(AnomalyDetector):
    """[스텁] 가학습(pre-trained) YOLO 예측 vs 정답 라벨 불일치 탐지.

    구현 가이드:
      1) ultralytics YOLO 로 배치 추론
      2) 예측 박스 <-> GT 박스 헝가리안/greedy IoU 매칭
      3) 보고 대상:
         - GT 有 / 예측 無 (IoU<0.3)          -> 라벨 오류 또는 극단적 hard sample
         - 예측 高신뢰(>0.8) / GT 無           -> 누락 라벨(missing annotation) 후보
         - IoU 는 높은데 class 불일치          -> 클래스 혼동
    """

    name = "YOLO 예측 불일치"
    description = "학습된 모델 예측과 GT 를 비교해 누락/오라벨 후보 추출"

    def available(self) -> Tuple[bool, str]:
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            return False, "ultralytics 미설치 (pip install ultralytics)"
        return False, "백엔드는 준비됨. analyze_batch() 구현 필요"


class LMStudioVLMDetector(AnomalyDetector):
    """[스텁] LM Studio 로컬 서버(OpenAI 호환)의 비전 모델로 라벨 교차 검증.

    LM Studio 는 http://localhost:1234/v1 에 OpenAI 호환 엔드포인트를 연다.
    이미지를 base64 data URL 로 실어 보내고, JSON 만 뱉도록 강제하는 형태:

        POST /v1/chat/completions
        {
          "model": "<lm-studio-model-id>",
          "messages": [{"role":"user","content":[
              {"type":"text","text":"이 이미지에 안전모를 쓴 사람이 있나? JSON 으로만 답해."},
              {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...."}}
          ]}],
          "temperature": 0,
          "response_format": {"type":"json_schema", "json_schema": {...}}
        }

    ※ 검수(존재 여부 판정)용으로는 쓸 만하지만,
      좌표 생성(라벨링)용으로는 권장하지 않는다. 하단 주석 [설계 노트] 참고.
    """

    name = "LM Studio VLM 교차검증"
    description = "로컬 VLM 에게 '이 클래스가 실제로 있나?' 만 물어 오라벨 색출"

    endpoint = "http://localhost:1234/v1/chat/completions"

    def available(self) -> Tuple[bool, str]:
        return False, f"LM Studio 서버 연동 미구현 ({self.endpoint})"


class AutoLabeler:
    """[확장 포인트 2] 이미지만 있는 폴더에 라벨을 생성하는 자동 라벨러 인터페이스."""

    name = "base"

    def available(self) -> Tuple[bool, str]:
        return False, "미구현 인터페이스"

    def predict(self, image: np.ndarray) -> List[Shape]:
        """정규화 좌표 Shape 리스트를 반환. 그대로 YOLO txt 로 직렬화 가능."""
        raise NotImplementedError


class GroundingDinoAutoLabeler(AutoLabeler):
    """[스텁] 텍스트 프롬프트 -> bbox. 오픈보캐뷸러리 자동 라벨링의 실질적 1순위."""

    name = "GroundingDINO (text -> box)"

    def available(self) -> Tuple[bool, str]:
        return False, "transformers + GroundingDINO 가중치 필요"


class SamRefiner(AutoLabeler):
    """[스텁] box -> mask. 기존 detection 라벨을 segmentation 으로 승격시킨다.

    구현 가이드:
      1) segment_anything / SAM2 로드
      2) predictor.set_image(rgb)
      3) 각 GT 박스를 픽셀좌표로 변환해 box prompt 로 입력
      4) 반환 mask -> cv2.findContours -> approxPolyDP 로 점 수 축소(보통 30~60점)
      5) 정규화 후 "<cls> x1 y1 x2 y2 ..." 로 저장
    이 경로는 좌표를 '생성'하는 게 아니라 '정제'하는 것이라 신뢰도가 훨씬 높다.
    """

    name = "SAM box -> polygon 승격"

    def available(self) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "torch 미설치"
        return False, "SAM 체크포인트 경로 설정 후 predict() 구현 필요"


def shapes_to_yolo_txt(shapes: Sequence[Shape]) -> str:
    """AutoLabeler 결과를 YOLO txt 문자열로 직렬화 (자동 라벨링 파이프라인 공용)."""
    lines = []
    for sh in shapes:
        if sh.kind == "box":
            cx, cy, w, h = sh.box
            lines.append(f"{sh.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:
            coords = " ".join(f"{v:.6f}" for v in sh.poly.reshape(-1))
            lines.append(f"{sh.cls} {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


DETECTOR_REGISTRY: List[AnomalyDetector] = [
    ClipConsistencyDetector(),
    YoloDisagreementDetector(),
    LMStudioVLMDetector(),
]
AUTOLABELER_REGISTRY: List[AutoLabeler] = [GroundingDinoAutoLabeler(), SamRefiner()]


# ==============================================================================
# MODULE 8 : review  --  라운드 세션 / 편집 비용 계측 / 출처 태그
# ==============================================================================
#
#  이 모듈은 "사람이 자동 라벨을 얼마나 고쳐야 하는가"를 실측하기 위한 것이다.
#  자동 라벨러의 품질 지표는 mAP 가 아니라 사람 편집 비용이고, 그 비용을 실제
#  편집 행위에서 뽑아내는 게 여기서 하는 일 전부다.
#
#  [ 3구역 구조 ]
#      <루트>/db/<데이터셋>/     원본. 이미지 + GT 라벨. 읽기 전용으로 봉인.
#      <루트>/work/<작업공간>/   라벨만 보관. 이미지는 db 를 참조한다.
#      <루트>/export/<이름>/     학습용으로 조립해 내보낸 결과.
#
#  db 를 건드리는 쓰기 코드는 이 파일 어디에도 없다. 작업공간은 몇 개든 만들 수
#  있고(사람 수동 / 모델 검수 / 세그멘테이션 …) 서로 독립이므로 실험을 망쳐도
#  원본과 다른 작업공간은 멀쩡하다.
#
#  [ 출처 태그 — 권위 순서대로 내려간다 ]
#      A  사람        사람이 직접 그리거나 고친 것          ← 가장 높음
#      B  GT          데이터셋 원본 주석자 (익명·규약 문서 없음)
#      C  학습된 모델
#      D  zero-shot
#
#  A 와 B 가 부딪히면 A 가 이긴다. GT 를 부정하는 게 아니라 제자리에 놓는 것이다 --
#  db 의 라벨은 문서화된 벤치마크가 아니라 개인이 올린 결과물이고 오류가 있을 수 있다.
#  사람이 아직 보지 않은 이미지에서는 B 가 유일한 기준으로 남는다.
#
#  출처와 검증은 다른 축이다. 모델이 그린 박스(C)를 사람이 보고 승인해도 출처는 C 로
#  남는다. 승격시키면 검수를 마친 이미지가 전부 A 가 되어 자동화율이 0 으로 수렴한다.
#  검증 여부는 박스가 아니라 이미지 단위로 기록된다(meta 기록이 있으면 검증된 것).
#
#  최종 보고는 A+B(사람이 만든 라벨) 대 C+D(기계가 만든 라벨) 로 낸다.
# ==============================================================================
DB_DIR, WORK_DIR, EXPORT_DIR = "db", "work", "export"
MANIFEST = "workspace.json"

# 출처 태그는 권위 순서대로 내려간다. A 가 가장 높고 D 가 가장 낮다.
# 충돌하면 앞선 글자가 이긴다.
PROV_TAGS = {
    "A": "사람 (직접 그리거나 고침)",
    "B": "GT (데이터셋 원본 주석자)",
    "C": "학습된 모델",
    "D": "zero-shot",
}
PROV_RANK = {k: i for i, k in enumerate(PROV_TAGS)}  # 낮을수록 우선
SEED_MODES = {
    "pred": "예측 폴더에서 시드 (C/D)",
    "empty": "빈 상태에서 시작 (전부 A)",
    "gt": "GT 복사 — 시뮬레이션 전용 (B)",
}
WORKSPACE_KINDS = {
    "manual": "사람이 직접 라벨링",
    "auto": "모델 예측 + 사람 검수",
    "seg": "세그멘테이션",
    "import": "외부 라벨 반입",
}


def safe_name(s: str) -> str:
    """스플릿 이름에 '/' 가 들어와도(멀티 프로젝트 루트) 폴더명으로 쓸 수 있게."""
    out = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", s).strip("_")
    return out or "x"


def fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.0f}초"
    if sec < 3600:
        return f"{int(sec // 60)}분 {int(sec % 60)}초"
    return f"{int(sec // 3600)}시간 {int(sec % 3600 // 60)}분"


# ------------------------------------------------------------------ 3구역 배치
class Layout:
    """루트 하나 = db / work / export 삼분할."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    @property
    def db_dir(self) -> str:
        return os.path.join(self.root, DB_DIR)

    @property
    def work_dir(self) -> str:
        return os.path.join(self.root, WORK_DIR)

    @property
    def export_dir(self) -> str:
        return os.path.join(self.root, EXPORT_DIR)

    def ensure(self) -> "Layout":
        for d in (self.db_dir, self.work_dir, self.export_dir):
            os.makedirs(d, exist_ok=True)
        return self

    @staticmethod
    def discover(path: str) -> Optional["Layout"]:
        """아무 경로나 주면 그걸 품고 있는 루트를 찾아 올라간다.

        db/vest-helmet/train 을 열어도, DATA 를 열어도 같은 배치를 찾아내야
        '작업공간을 어디에 만들지' 를 사용자에게 다시 묻지 않는다.
        """
        p = os.path.abspath(path)
        for _ in range(6):
            if os.path.isdir(os.path.join(p, DB_DIR)) and \
               os.path.isdir(os.path.join(p, WORK_DIR)):
                return Layout(p)
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return None

    def datasets(self) -> List[str]:
        """db 아래의 데이터셋 폴더 목록."""
        out = []
        if not os.path.isdir(self.db_dir):
            return out
        for e in sorted(os.scandir(self.db_dir), key=lambda x: x.name):
            if e.is_dir():
                out.append(e.path)
        return out

    def workspace_dirs(self) -> List[str]:
        out = []
        if not os.path.isdir(self.work_dir):
            return out
        for e in sorted(os.scandir(self.work_dir), key=lambda x: x.name):
            if e.is_dir() and os.path.isfile(os.path.join(e.path, MANIFEST)):
                out.append(e.path)
        return out

    def workspaces(self) -> List["Workspace"]:
        out = []
        for d in self.workspace_dirs():
            try:
                out.append(Workspace.load(d))
            except (OSError, ValueError):
                continue
        return out


# ------------------------------------------------------------------- 비용 모델
@dataclass
class CostModel:
    """합의된 사람 편집 비용 모델(초).

    오프라인 시뮬레이션에서 쓴 것과 같은 수치를 쓴다. 그래야 UI 로 실측한 비용과
    시뮬레이션 추정치를 같은 축에서 비교할 수 있다.
    """

    draw: float = 5.0  # 누락 -> 새로 그리기
    delete: float = 1.0  # 오검출 -> 삭제
    adjust: float = 3.0  # IoU 0.5~0.75 -> 위치 조정
    free_iou: float = 0.75  # 이 이상 겹치면 손댈 필요 없음 = 무료
    match_iou: float = 0.50  # 이 미만이면 아예 다른 박스로 취급

    def manual_cost(self, n_boxes: int) -> float:
        """비교 기준선: 아무 자동 라벨 없이 전부 직접 그렸을 때의 비용."""
        return n_boxes * self.draw

    def to_dict(self) -> dict:
        return {"draw": self.draw, "delete": self.delete, "adjust": self.adjust,
                "free_iou": self.free_iou, "match_iou": self.match_iou}

    @staticmethod
    def from_dict(d: dict) -> "CostModel":
        cm = CostModel()
        for k, v in (d or {}).items():
            if hasattr(cm, k):
                setattr(cm, k, float(v))
        return cm


@dataclass
class EditOps:
    """이미지 한 장에 대해 사람이 한 일."""

    drawn: int = 0  # 새로 그린 박스
    deleted: int = 0  # 지운 박스
    adjusted: int = 0  # 위치/클래스를 고친 박스
    kept: int = 0  # 손대지 않고 통과시킨 박스
    seconds: float = 0.0  # 실제 화면에 머문 시간(초)

    def cost(self, cm: CostModel) -> float:
        return self.drawn * cm.draw + self.deleted * cm.delete + self.adjusted * cm.adjust

    @property
    def total_boxes(self) -> int:
        return self.drawn + self.adjusted + self.kept

    def add(self, o: "EditOps") -> None:
        self.drawn += o.drawn
        self.deleted += o.deleted
        self.adjusted += o.adjusted
        self.kept += o.kept
        self.seconds += o.seconds

    def to_dict(self) -> dict:
        return {"drawn": self.drawn, "deleted": self.deleted, "adjusted": self.adjusted,
                "kept": self.kept, "seconds": round(self.seconds, 2)}

    @staticmethod
    def from_dict(d: dict) -> "EditOps":
        d = d or {}
        return EditOps(int(d.get("drawn", 0)), int(d.get("deleted", 0)),
                       int(d.get("adjusted", 0)), int(d.get("kept", 0)),
                       float(d.get("seconds", 0.0)))


def is_flat_dataset(project: "Project") -> bool:
    """스플릿이 없는 '이미지만 있는' 데이터셋인가.

    train/valid/test 가 없으면 작업 흐름이 갈라진다. 그래서 라벨링을 시작하기
    전에 나눠서, 기존 데이터셋과 똑같은 경로를 타게 만든다.
    """
    if not project.splits:
        return True
    names = {canon_split(s.name) for s in project.splits}
    return len(project.splits) == 1 and project.splits[0].name in ("root", ".")


def split_flat_dataset(
    src_root: str,
    dst_root: str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """이미지만 있는 폴더를 train/valid/test 구조로 나눠 새 데이터셋을 만든다.

    원본은 건드리지 않고 복사한다. 라벨(.txt)이 있으면 함께 옮긴다.
    나눈 근거를 split_manifest.json 에 남겨 같은 분할을 다시 만들 수 있게 한다.
    """
    imgs: List[str] = []
    labels_dir = None
    for cand in ("images", "."):
        d = src_root if cand == "." else os.path.join(src_root, cand)
        if not os.path.isdir(d):
            continue
        found = sorted(f for f in os.listdir(d)
                       if f.lower().endswith(IMG_EXTS))
        if found:
            imgs = [os.path.join(d, f) for f in found]
            sib = os.path.join(os.path.dirname(d), "labels")
            labels_dir = sib if os.path.isdir(sib) else None
            break
    if not imgs:
        raise ValueError(f"이미지를 찾지 못했습니다: {src_root}")

    order = list(imgs)
    random.Random(int(seed)).shuffle(order)
    n = len(order)
    tot = sum(ratios) or 1.0
    n_tr = int(round(n * ratios[0] / tot))
    n_va = min(int(round(n * ratios[1] / tot)), n - n_tr)
    buckets = {"train": order[:n_tr],
               "valid": order[n_tr:n_tr + n_va],
               "test": order[n_tr + n_va:]}

    done = 0
    for split, files in buckets.items():
        if not files:
            continue
        oi = os.path.join(dst_root, split, "images")
        ol = os.path.join(dst_root, split, "labels")
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        for p in files:
            shutil.copy2(p, os.path.join(oi, os.path.basename(p)))
            lp = (os.path.join(labels_dir, stem(p) + ".txt")
                  if labels_dir else None)
            dst_lp = os.path.join(ol, stem(p) + ".txt")
            if lp and os.path.isfile(lp):
                shutil.copy2(lp, dst_lp)
            else:
                open(dst_lp, "w", encoding="utf-8").close()
            done += 1
            if progress and done % 50 == 0:
                progress(done, n, os.path.basename(p))

    # data.yaml 이 없으면 최소한의 것을 만든다
    src_yaml = None
    for cand in ("data.yaml", "data.yml", "dataset.yaml"):
        p = os.path.join(src_root, cand)
        if os.path.isfile(p):
            src_yaml = p
            break
    if src_yaml:
        shutil.copy2(src_yaml, os.path.join(dst_root, "data.yaml"))
    else:
        probe = Project(src_root)
        with open(os.path.join(dst_root, "data.yaml"), "w",
                  encoding="utf-8") as f:
            f.write("train: ../train/images\nval: ../valid/images\n"
                    "test: ../test/images\n\n")
            f.write(f"nc: {len(probe.names)}\n")
            f.write("names: [" + ", ".join(f"'{x}'" for x in probe.names) + "]\n")

    mf = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": os.path.abspath(src_root),
        "images": n, "ratios": list(ratios), "seed": seed,
        "counts": {k: len(v) for k, v in buckets.items()},
        "had_labels": bool(labels_dir),
        "note": "이미지만 있던 폴더를 나눈 것. 같은 시드로 같은 분할이 재현된다.",
        "assignment": {os.path.basename(p): sp
                       for sp, fs in buckets.items() for p in fs},
    }
    with open(os.path.join(dst_root, "split_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(mf, f, ensure_ascii=False, indent=1)
    if progress:
        progress(n, n, "완료")
    return mf


def build_worklist(
    files: Sequence[str],
    mode: str = "random",
    n: int = 0,
    stride: int = 1,
    seed: int = 0,
    pred_dir: Optional[str] = None,
    exclude: Sequence[str] = (),
    from_list: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """작업 목록을 만든다. 반환 (목록, 사람이 읽을 단계별 설명).

    이 계산을 UI 밖으로 빼둔 이유: 라운드를 하나 열 때마다 조건이 조금씩 다르고
    (학습에 쓴 이미지 제외, 이전 라운드와 겹침 방지, 등간격 추출 …) 그때마다
    일회용 스크립트를 쓰면 어떤 조건으로 뽑았는지 기록이 남지 않는다.

    mode
      all    : 전부
      random : 무작위 셔플 후 n 장 (n=0 이면 전부)
      stride : stride 간격으로 등간격 추출 -- 봉인 평가셋이 이 방식으로 만들어졌다
    """
    steps: List[str] = []
    pool = list(files)
    steps.append(f"후보 {len(pool)}장")

    if from_list:
        try:
            with open(from_list, "r", encoding="utf-8") as f:
                want = [ln.strip() for ln in f if ln.strip()]
        except OSError as e:
            return [], [f"목록 파일을 읽지 못했습니다: {e}"]
        # 경로가 섞여 있어도 파일명만 본다
        want_names = [os.path.basename(w) for w in want]
        have = set(pool)
        picked = [w for w in want_names if w in have]
        missing = len(want_names) - len(picked)
        steps.append(f"목록 파일에서 {len(picked)}장"
                     + (f" (이 스플릿에 없는 {missing}장 제외)" if missing else ""))
        return picked, steps

    if pred_dir and os.path.isdir(pred_dir):
        have = {stem(f) for f in os.listdir(pred_dir) if f.endswith(".txt")}
        before = len(pool)
        pool = [f for f in pool if stem(f) in have]
        if before != len(pool):
            steps.append(f"예측 없는 {before - len(pool)}장 제외 → {len(pool)}장")

    if exclude:
        ex = set(exclude)
        before = len(pool)
        pool = [f for f in pool if f not in ex]
        if before != len(pool):
            steps.append(f"다른 작업공간과 겹치는 {before - len(pool)}장 제외 "
                         f"→ {len(pool)}장")

    if mode == "stride":
        s = max(1, int(stride))
        pool = pool[::s]
        steps.append(f"{s}장마다 등간격 추출 → {len(pool)}장")
    elif mode == "random":
        random.Random(int(seed)).shuffle(pool)
        if n > 0:
            pool = pool[:n]
        steps.append(f"무작위 셔플(시드 {seed})"
                     + (f" 후 {len(pool)}장" if n > 0 else f" · 전체 {len(pool)}장"))
    else:
        steps.append(f"전체 {len(pool)}장")
    return pool, steps


def tier_of(conf: float, hi: float, lo: float) -> str:
    """신뢰도 3분류. conf 가 없는(사람이 만든) 박스는 자동승인 취급."""
    if conf < 0:
        return "auto"
    if conf >= hi:
        return "auto"
    return "review" if conf >= lo else "reject"


def match_greedy(
    a: Sequence[Shape], b: Sequence[Shape], thr: float, same_class: bool = False
) -> List[Tuple[float, int, int]]:
    """IoU 내림차순 그리디 1:1 매칭. 반환: (iou, a인덱스, b인덱스) 목록."""
    pairs: List[Tuple[float, int, int]] = []
    bb = [shape_xyxy(y) for y in b]
    for i, x in enumerate(a):
        ax = shape_xyxy(x)
        for j, by in enumerate(bb):
            if same_class and a[i].cls != b[j].cls:
                continue
            v = iou_xyxy(ax, by)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(key=lambda t: -t[0])
    out, ua, ub = [], set(), set()
    for v, i, j in pairs:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        out.append((v, i, j))
    return out


def diff_ops(
    baseline: Sequence[Shape], final: Sequence[Shape], cm: CostModel
) -> EditOps:
    """시드 상태 대비 최종 상태를 비교해 편집량을 뽑는다.

    편집 중 이벤트를 세지 않고 전후 상태만 비교하는 이유: 그렸다 지운 박스나
    여러 번 미세조정한 박스가 비용에 중복 계상되면 오프라인 시뮬레이션 수치와
    비교가 안 된다. 매칭 규칙도 시뮬레이션과 동일하게 IoU 그리디로 맞춘다.
    """
    ops = EditOps()
    matched = match_greedy(baseline, final, cm.match_iou)
    for v, i, j in matched:
        if v >= cm.free_iou and baseline[i].cls == final[j].cls:
            ops.kept += 1
        else:
            ops.adjusted += 1
    ops.drawn = len(final) - len(matched)
    ops.deleted = len(baseline) - len(matched)
    return ops


# ------------------------------------------------------------------- 작업공간
class Workspace:
    """라벨셋 하나. 이미지는 갖지 않고 db 의 데이터셋을 참조한다.

    배치:
        work/<이름>/
            workspace.json          소스 참조 + 설정 + 계보
            labels/<split>/*.txt    편집 결과 (순수 YOLO = 그대로 학습 가능)
            meta/<split>/*.json     출처 태그 / 신뢰도 / 편집량 / 소요시간
            report.html             보고서 (요청 시 생성)
    """

    VERSION = 2

    def __init__(
        self,
        path: str,
        source: str,
        name: Optional[str] = None,
        kind: str = "auto",
        note: str = "",
        parent: Optional[str] = None,
        model: Optional[dict] = None,
        seed_mode: str = "pred",
        pred_dir: Optional[str] = None,
        pred_src: str = "C",
        hi: float = 0.60,
        lo: float = 0.25,
        shuffle_seed: int = 0,
        cost: Optional[CostModel] = None,
        created: Optional[str] = None,
        worklist: Optional[Dict[str, List[str]]] = None,
        classes: Optional[List[str]] = None,
        worklist_spec: Optional[dict] = None,
        live: Optional[dict] = None,
    ):
        self.path = os.path.abspath(path)
        self.source = os.path.abspath(source)
        self.name = name or os.path.basename(self.path)
        self.kind = kind if kind in WORKSPACE_KINDS else "auto"
        self.note = note
        self.parent = parent
        self.model = model or {}
        self.seed_mode = seed_mode
        self.pred_dir = pred_dir
        self.pred_src = pred_src if pred_src in PROV_TAGS else "C"
        self.hi, self.lo = float(hi), float(lo)
        self.shuffle_seed = int(shuffle_seed)
        self.cost = cost or CostModel()
        self.created = created or time.strftime("%Y-%m-%d %H:%M:%S")
        self.worklist: Dict[str, List[str]] = dict(worklist or {})
        # 작업목록을 어떤 조건으로 뽑았는지. 없으면 같은 목록을 다시 만들 수 없다.
        self.worklist_spec: dict = dict(worklist_spec or {})
        self.classes: List[str] = list(classes or [])
        # 연속형 라벨링 설정. 비어 있으면 기존 방식(고정 모델 하나)으로 동작한다.
        #   enabled/every  N장 확정할 때마다 백그라운드 학습을 요청한다
        #   model/model_n  지금 시드를 만든 모델과 그 학습분 (낡음 = 확정수 - model_n)
        #   version        다음 모델에 붙일 번호
        # 학습과 예측 생성은 train_worker.py 가 별도 프로세스로 한다.
        self.live: dict = dict(live or {})
        # (split, 파일명) -> meta dict
        self.records: Dict[Tuple[str, str], dict] = {}
        self._load_records()

    # ---- 경로 ------------------------------------------------------------
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.path, MANIFEST)

    @property
    def report_path(self) -> str:
        return os.path.join(self.path, "report.html")

    def labels_dir(self, split: str) -> str:
        return os.path.join(self.path, "labels", safe_name(split))

    def meta_dir(self, split: str) -> str:
        return os.path.join(self.path, "meta", safe_name(split))

    def label_path(self, split: str, image_name: str) -> str:
        return os.path.join(self.labels_dir(split), stem(image_name) + ".txt")

    def meta_path(self, split: str, image_name: str) -> str:
        return os.path.join(self.meta_dir(split), stem(image_name) + ".json")

    def pred_path(self, image_name: str) -> Optional[str]:
        if not self.pred_dir:
            return None
        p = os.path.join(self.pred_dir, stem(image_name) + ".txt")
        return p if os.path.isfile(p) else None

    def has_label(self, split: str, image_name: str) -> bool:
        return os.path.isfile(self.label_path(split, image_name))

    # ---- 영속화 ----------------------------------------------------------
    def to_dict(self) -> dict:
        try:  # 루트째로 옮겨도 살아남도록 상대경로 우선
            src = os.path.relpath(self.source, self.path).replace("\\", "/")
        except ValueError:  # 다른 드라이브
            src = self.source
        return {
            "version": self.VERSION, "name": self.name, "kind": self.kind,
            "source": src, "source_abs": self.source, "note": self.note,
            "parent": self.parent, "model": self.model, "created": self.created,
            "seed_mode": self.seed_mode, "pred_dir": self.pred_dir,
            "pred_src": self.pred_src, "hi": self.hi, "lo": self.lo,
            "shuffle_seed": self.shuffle_seed, "cost": self.cost.to_dict(),
            "classes": self.classes, "worklist": self.worklist,
            "worklist_spec": self.worklist_spec, "live": self.live,
        }

    def save(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "Workspace":
        path = os.path.abspath(path)
        with open(os.path.join(path, MANIFEST), "r", encoding="utf-8") as f:
            d = json.load(f)
        src = d.get("source") or ""
        src_abs = os.path.normpath(os.path.join(path, src)) if src else ""
        if not os.path.isdir(src_abs):
            src_abs = d.get("source_abs") or src_abs
        return cls(
            path=path, source=src_abs, name=d.get("name"), kind=d.get("kind", "auto"),
            note=d.get("note", ""), parent=d.get("parent"), model=d.get("model"),
            seed_mode=d.get("seed_mode", "pred"), pred_dir=d.get("pred_dir"),
            pred_src=d.get("pred_src", "C"), hi=d.get("hi", 0.60), lo=d.get("lo", 0.25),
            shuffle_seed=d.get("shuffle_seed", 0),
            cost=CostModel.from_dict(d.get("cost", {})), created=d.get("created"),
            worklist=d.get("worklist"), classes=d.get("classes"),
            worklist_spec=d.get("worklist_spec"), live=d.get("live"),
        )

    @classmethod
    def create(cls, layout: "Layout", name: str, source: str, **kw) -> "Workspace":
        layout.ensure()
        path = os.path.join(layout.work_dir, safe_name(name))
        ws = cls(path=path, source=source, name=name, **kw)
        os.makedirs(ws.path, exist_ok=True)
        ws.save()
        return ws

    def _load_records(self) -> None:
        base = os.path.join(self.path, "meta")
        if not os.path.isdir(base):
            return
        for sub in os.scandir(base):
            if not sub.is_dir():
                continue
            for e in os.scandir(sub.path):
                if not e.is_file() or not e.name.endswith(".json"):
                    continue
                try:
                    with open(e.path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                except (OSError, ValueError):
                    continue
                key = (d.get("split", sub.name), d.get("image", stem(e.name)))
                self.records[key] = d

    # ---- 시드 / 로드 -----------------------------------------------------
    def ghosts_for(self, image_name: str) -> List[Shape]:
        """예측 오버레이. 티어까지 계산해서 돌려준다."""
        preds, _ = parse_label_file(self.pred_path(image_name), default_src=self.pred_src)
        for p in preds:
            p.tier = tier_of(p.conf, self.hi, self.lo)
        return preds

    def seed_shapes(
        self, image_name: str, gt_shapes: Sequence[Shape] = ()
    ) -> List[Shape]:
        """아직 편집한 적 없는 이미지의 출발 상태를 만든다.

        - pred  : 신뢰도 lo 이상(자동승인+검토)만 라벨로 앉힌다. 기각 티어는
                  고스트로만 남아 클릭 한 번으로 승격시킬 수 있다.
        - gt    : GT 를 사람 주석자 대역으로 쓰는 시뮬레이션. 출처는 GT 이므로 태그 B.
        - empty : 아무것도 안 앉힌다.
        """
        if self.seed_mode == "gt":
            out = []
            for g in gt_shapes:
                s = g.clone()
                s.src, s.conf, s.tier = "B", -1.0, "auto"
                out.append(s)
            return out
        if self.seed_mode == "empty":
            return []
        return [g.clone() for g in self.ghosts_for(image_name) if g.tier != "reject"]

    def load_shapes(
        self, split: str, image_name: str, gt_shapes: Sequence[Shape] = ()
    ) -> Tuple[List[Shape], bool]:
        """반환: (shapes, 이미 저장된 적 있는가).

        저장본이 있으면 그걸 쓰고, 없으면 시드를 만든다. 출처 태그는 순수 YOLO
        txt 에 넣을 수 없으므로 meta json 에서 되살린다.
        """
        lp = self.label_path(split, image_name)
        if os.path.isfile(lp):
            shapes, _ = parse_label_file(lp)
            rec = self.records.get((split, image_name)) or {}
            srcs = rec.get("src") or []
            confs = rec.get("conf") or []
            for i, sh in enumerate(shapes):
                sh.src = srcs[i] if i < len(srcs) else "A"
                sh.conf = float(confs[i]) if i < len(confs) else -1.0
                sh.tier = tier_of(sh.conf, self.hi, self.lo)
            return shapes, True
        return self.seed_shapes(image_name, gt_shapes), False

    def baseline_of(
        self, split: str, image_name: str, seeded: Sequence[Shape]
    ) -> List[Shape]:
        """비용 계산의 기준선. 한 번 저장했으면 meta 에 박제된 시드를 계속 쓴다."""
        rec = self.records.get((split, image_name))
        if rec and rec.get("baseline") is not None:
            return [
                Shape(int(b[0]), "box", box=(b[1], b[2], b[3], b[4]))
                for b in rec["baseline"]
            ]
        return [s.clone() for s in seeded]

    # ---- 저장 ------------------------------------------------------------
    def commit(
        self,
        split: str,
        image_name: str,
        shapes: Sequence[Shape],
        baseline: Sequence[Shape],
        seconds: float,
        promoted: int = 0,
    ) -> EditOps:
        """편집 결과를 작업공간에 쓰고 편집량을 기록한다."""
        os.makedirs(self.labels_dir(split), exist_ok=True)
        os.makedirs(self.meta_dir(split), exist_ok=True)
        with open(self.label_path(split, image_name), "w", encoding="utf-8") as f:
            f.write(shapes_to_yolo_txt(shapes))  # 신뢰도 없는 순수 YOLO = 학습 가능

        ops = diff_ops(baseline, shapes, self.cost)
        prev = self.records.get((split, image_name)) or {}
        ops.seconds = float(prev.get("ops", {}).get("seconds", 0.0)) + max(0.0, seconds)

        # 기준선은 최초 저장 때 한 번만 박제한다. 두 번째 저장부터 현재 상태를
        # 기준선으로 삼으면 편집량이 0 으로 리셋돼 비용이 사라진다.
        base_rec = prev.get("baseline")
        if base_rec is None:
            base_rec = [
                [b.cls, *[round(v, 6) for v in xyxy_to_cxcywh(*shape_xyxy(b))]]
                for b in baseline
            ]

        rec = {
            "image": image_name, "split": split, "workspace": self.name,
            "src": [s.src or "A" for s in shapes],
            "conf": [round(s.conf, 4) for s in shapes],
            "tier": [s.tier or "auto" for s in shapes],
            "cls": [s.cls for s in shapes],
            "ops": ops.to_dict(),
            "cost_sec": round(ops.cost(self.cost), 2),
            "manual_sec": round(self.cost.manual_cost(len(shapes)), 2),
            # 임계값 아래 예측을 클릭 한 번으로 승격시킨 수. 비용 모델에서는 이것도
            # '그리기 5초'로 계상되지만 실제로는 훨씬 싸다. 사후에 다시 값을 매길
            # 수 있도록 개수를 따로 남긴다.
            "promoted": int(prev.get("promoted", 0)) + int(promoted),
            "baseline": base_rec,
            # 이 이미지의 시드를 만든 모델. 연속형에서는 작업 도중에 모델이 바뀌므로
            # 작업공간 단위로는 남길 수 없다. 기준선과 같이 최초 저장 때 박제한다 --
            # 두 번째 저장 때 최신 모델 이름으로 덮으면 누가 시드했는지를 잃는다.
            "pred_model": prev.get("pred_model", self.live.get("model")),
            "pred_model_n": prev.get("pred_model_n", self.live.get("model_n")),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.meta_path(split, image_name), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        self.records[(split, image_name)] = rec
        return ops

    # ---- 연속형 라벨링 ----------------------------------------------------
    #
    #  사람이 라벨링하는 동안 train_worker.py 가 뒤에서 학습하고 예측을 다시 깐다.
    #  스튜디오는 GPU 를 직접 만지지 않는다 -- 요청을 폴더에 떨어뜨리고(live_tick),
    #  끝난 것이 있으면 주워온다(live_adopt). 그래서 UI 가 학습에 붙들리지 않는다.
    #
    #  train_worker 를 지연 import 하는 이유: 그 파일이 없거나 깨져 있어도 스튜디오는
    #  평소대로 돌아야 한다. 연속형은 어디까지나 부가 기능이다.

    @staticmethod
    def _worker():
        try:
            import train_worker
            return train_worker
        except ImportError:
            return None

    def confirmed_count(self) -> int:
        """사람이 검수를 마친 이미지 수. meta 기록이 있으면 확정된 것으로 본다."""
        return len(self.records)

    def live_status(self) -> dict:
        """상태줄에 띄울 값. 낡음 = 지금 확정한 수 - 현재 모델의 학습분."""
        w = self._worker()
        n = self.confirmed_count()
        mn = self.live.get("model_n")
        #  워커가 안 떠 있으면 요청이 큐에 쌓이기만 하고 학습은 영원히 안 된다.
        #  화면에 안 보이면 500장을 라벨하고서야 알게 되는 조용한 실패가 된다.
        root = os.path.dirname(os.path.dirname(self.path))
        hb = w.worker_alive(root) if w else None
        return {
            "enabled": bool(self.live.get("enabled")),
            "confirmed": n,
            "model": self.live.get("model"),
            "model_n": mn,
            "stale": (n - int(mn)) if mn is not None else None,
            "pending": w.pending(self.path) if w else 0,
            "every": int(self.live.get("every", 50)),
            "worker": (hb or {}).get("state") if hb else None,
        }

    def live_tick(self) -> Optional[str]:
        """검수 한 장이 끝날 때마다 부른다. 학습을 걸 때가 됐으면 요청을 넣는다.

        이미 큐에 있거나 돌고 있으면 넣지 않는다. 학습 한 판(3분)보다 사람이 빠른
        구간에서는 요청이 계속 쌓이기만 하기 때문이다. 워커는 어차피 요청 시점까지
        확정된 라벨을 전부 학습한다.
        """
        w = self._worker()
        if not (w and self.live.get("enabled")):
            return None
        n = self.confirmed_count()
        every = max(1, int(self.live.get("every", 50)))
        if n < int(self.live.get("last_request_n", 0)) + every:
            return None
        if w.pending(self.path):
            return None
        ver = int(self.live.get("version", 1))
        job = {
            "workspace": self.name,
            "dataset": self.live.get("dataset"),
            "split": self.live.get("split", "train"),
            "n_confirmed": n,
            "version": ver,
            "base": self.live.get("base", "yolo11s.pt"),
            "epochs": int(self.live.get("epochs", 80)),
            "seed": int(self.live.get("seed", 0)),
            "val_fast_n": int(self.live.get("val_fast_n", 400)),
            "val_period": int(self.live.get("val_period", 5)),
        }
        w.submit(self.path, job)
        self.live["last_request_n"] = n
        self.live["version"] = ver + 1
        self.save()
        return job.get("id")

    def live_adopt(self) -> Optional[dict]:
        """새로 끝난 모델이 있으면 그쪽 예측으로 갈아탄다. 없으면 None.

        확정된 이미지는 건드리지 않는다 -- 예측은 아직 안 연 이미지에만 쓰인다.
        """
        w = self._worker()
        if not (w and self.live.get("enabled")):
            return None
        job = w.latest_model(self.path)
        if not job or job.get("model") == self.live.get("model"):
            return None
        self.live["model"] = job.get("model")
        self.live["model_n"] = job.get("trained_images")
        self.live["metrics"] = job.get("metrics")
        if job.get("pred_dir"):
            self.pred_dir = job["pred_dir"]
        self.model = {"name": job.get("model"),
                      **{k: v for k, v in (job.get("metrics") or {}).items()}}
        self.save()
        return job

    # ---- 진행 ------------------------------------------------------------
    def worklist_pairs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for sp, files in self.worklist.items():
            out.extend((sp, f) for f in files)
        return out

    def progress(self) -> Tuple[int, int]:
        pairs = self.worklist_pairs()
        if not pairs:  # 워크리스트가 없으면 '저장된 것 전부'가 진행분
            return len(self.records), len(self.records)
        return sum(1 for k in pairs if k in self.records), len(pairs)

    def next_pending(
        self, split: str, after: Optional[str] = None
    ) -> Optional[str]:
        wl = self.worklist.get(split) or []
        if not wl:
            return None
        start = (wl.index(after) + 1) if (after in wl) else 0
        for k in range(len(wl)):
            fn = wl[(start + k) % len(wl)]
            if (split, fn) not in self.records:
                return fn
        return None

    def totals(self) -> dict:
        ops = EditOps()
        prov: Dict[str, int] = {k: 0 for k in PROV_TAGS}
        cls_dist: Dict[int, int] = {}
        boxes = promoted = 0
        per_image: List[float] = []
        stamps: List[str] = []
        for rec in self.records.values():
            o = EditOps.from_dict(rec.get("ops"))
            ops.add(o)
            per_image.append(o.seconds)
            promoted += int(rec.get("promoted", 0))
            if rec.get("ts"):
                stamps.append(rec["ts"])
            for t in rec.get("src", []):
                prov[t] = prov.get(t, 0) + 1
                boxes += 1
            for c in rec.get("cls", []):
                cls_dist[int(c)] = cls_dist.get(int(c), 0) + 1
        cost = ops.cost(self.cost)
        manual = self.cost.manual_cost(boxes)
        done, total = self.progress()
        per_image.sort()
        med = per_image[len(per_image) // 2] if per_image else 0.0
        return {
            "ops": ops, "boxes": boxes, "prov": prov, "promoted": promoted,
            "cls_dist": cls_dist, "images": len(self.records),
            "done": done, "total": total,
            "cost_sec": cost, "manual_sec": manual,
            "ratio": (cost / manual) if manual > 0 else 0.0,  # 낮을수록 좋다 (목표 ≤0.35)
            "auto_rate": (ops.kept / ops.total_boxes) if ops.total_boxes else 0.0,
            "median_sec": med,
            "mean_sec": (ops.seconds / len(per_image)) if per_image else 0.0,
            "first_ts": min(stamps) if stamps else "",
            "last_ts": max(stamps) if stamps else "",
        }


# --------------------------------------------------------------- GT 대비 채점
def score_against_gt(
    ws: Workspace, project: "Project", iou_thr: float = 0.5
) -> Optional[dict]:
    """작업공간의 라벨을 db 의 GT 로 채점한다.

    GT 가 없는 도메인(0부터 구축하는 총기·과일 등)에서는 None 을 돌려준다.
    자동승인 티어의 precision 은 별도로 뽑는다 -- 목표(≥0.95)가 걸린 숫자라
    전체 precision 에 섞이면 판단이 안 된다.
    """
    tp = fp = fn = 0
    tier_hit: Dict[str, List[int]] = {t: [0, 0] for t in ("auto", "review", "reject")}
    n_img = 0
    n_gt_img = 0
    for (split, fname), rec in ws.records.items():
        sp = project.split_by_name(split)
        if sp is None:
            continue
        gtp = sp.label_path_for(fname)
        if not gtp:
            continue
        n_gt_img += 1
        gt, _ = parse_label_file(gtp)
        pred, _ = parse_label_file(ws.label_path(split, fname))
        n_img += 1
        matched = match_greedy(pred, gt, iou_thr, same_class=True)
        tp += len(matched)
        fp += len(pred) - len(matched)
        fn += len(gt) - len(matched)
        hit = {i for _v, i, _j in matched}
        tiers = rec.get("tier") or []
        for i in range(len(pred)):
            t = tiers[i] if i < len(tiers) else "auto"
            if t not in tier_hit:
                t = "auto"
            tier_hit[t][1] += 1
            if i in hit:
                tier_hit[t][0] += 1
    if n_gt_img == 0:
        return None
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "images": n_img, "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec_,
        "f1": (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) else 0.0,
        "iou_thr": iou_thr,
        "tier": {t: {"hit": v[0], "total": v[1],
                     "precision": (v[0] / v[1]) if v[1] else 0.0}
                 for t, v in tier_hit.items()},
    }


# ------------------------------------------------------------------- 보고서
def build_report(
    ws: Workspace, project: Optional["Project"] = None,
    siblings: Sequence[Workspace] = ()
) -> dict:
    """작업공간 하나의 상황을 한 덩어리로 모은다."""
    t = ws.totals()
    names = ws.classes or (project.names if project else [])
    score = None
    if project is not None:
        try:
            score = score_against_gt(ws, project)
        except Exception:  # 채점 실패가 보고서 전체를 막지 않게
            traceback.print_exc()
    children = [
        {"name": c.name, "kind": c.kind, "images": len(c.records),
         "auto_rate": c.totals()["auto_rate"], "note": c.note}
        for c in siblings if c.parent == ws.name
    ]
    return {
        "ws": ws, "totals": t, "score": score, "children": children,
        "class_names": names,
        "seed_desc": SEED_MODES.get(ws.seed_mode, ws.seed_mode),
        "kind_desc": WORKSPACE_KINDS.get(ws.kind, ws.kind),
    }


def _bar(pct: float, good: bool = True) -> str:
    pct = max(0.0, min(100.0, pct))
    color = "#7ec97e" if good else "#e0a34a"
    return (f'<div class="bar"><span style="width:{pct:.1f}%;background:{color}">'
            f'</span></div>')


def render_report_html(data: dict) -> str:
    """자체 완결 HTML. 앱 없이 열어봐도 그대로 읽히게."""
    ws: Workspace = data["ws"]
    t = data["totals"]
    ops: EditOps = t["ops"]
    sc = data["score"]
    names = data["class_names"]

    def esc(s) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    rows_prov = "".join(
        f"<tr><td>{k}</td><td>{esc(v)}</td><td class='n'>{t['prov'].get(k, 0)}</td></tr>"
        for k, v in PROV_TAGS.items()
    )
    rows_cls = "".join(
        f"<tr><td>{i}</td><td>{esc(names[i] if i < len(names) else '?')}</td>"
        f"<td class='n'>{n}</td></tr>"
        for i, n in sorted(t["cls_dist"].items())
    ) or "<tr><td colspan=3 class='dim'>라벨 없음</td></tr>"

    human = t["prov"].get("A", 0) + t["prov"].get("B", 0)
    auto = t["prov"].get("C", 0) + t["prov"].get("D", 0)
    saved = (1 - t["ratio"]) * 100 if t["manual_sec"] else 0.0

    if sc:
        tr = sc["tier"]
        score_html = f"""
    <h2>GT 대비 채점 <span class="dim">(IoU ≥ {sc['iou_thr']:.2f}, 클래스 일치)</span></h2>
    <div class="grid">
      <div class="card"><div class="k">Precision</div>
        <div class="v">{sc['precision'] * 100:.1f}%</div>{_bar(sc['precision'] * 100)}</div>
      <div class="card"><div class="k">Recall</div>
        <div class="v">{sc['recall'] * 100:.1f}%</div>{_bar(sc['recall'] * 100)}</div>
      <div class="card"><div class="k">F1</div>
        <div class="v">{sc['f1'] * 100:.1f}%</div>{_bar(sc['f1'] * 100)}</div>
      <div class="card"><div class="k">TP / FP / FN</div>
        <div class="v">{sc['tp']} / {sc['fp']} / {sc['fn']}</div></div>
    </div>
    <table>
      <tr><th>티어</th><th>박스</th><th>정답</th><th>Precision</th><th>목표</th></tr>
      <tr><td>자동승인 (conf ≥ {ws.hi:.2f})</td><td class='n'>{tr['auto']['total']}</td>
          <td class='n'>{tr['auto']['hit']}</td>
          <td class='n'>{tr['auto']['precision'] * 100:.1f}%</td>
          <td class='{"ok" if tr['auto']['precision'] >= 0.95 else "bad"}'>
              ≥95% {"달성" if tr['auto']['precision'] >= 0.95 else "미달"}</td></tr>
      <tr><td>검토 ({ws.lo:.2f} ~ {ws.hi:.2f})</td><td class='n'>{tr['review']['total']}</td>
          <td class='n'>{tr['review']['hit']}</td>
          <td class='n'>{tr['review']['precision'] * 100:.1f}%</td><td class='dim'>—</td></tr>
      <tr><td>기각 승격 (conf &lt; {ws.lo:.2f})</td><td class='n'>{tr['reject']['total']}</td>
          <td class='n'>{tr['reject']['hit']}</td>
          <td class='n'>{tr['reject']['precision'] * 100:.1f}%</td><td class='dim'>—</td></tr>
    </table>"""
    else:
        score_html = ("<h2>GT 대비 채점</h2><p class='dim'>소스에 정답 라벨이 없습니다. "
                      "0부터 구축하는 데이터셋에서는 채점 대신 편집 비용과 진행률로 "
                      "판단하세요.</p>")

    kids = "".join(
        f"<tr><td>{esc(c['name'])}</td><td>{esc(WORKSPACE_KINDS.get(c['kind'], c['kind']))}</td>"
        f"<td class='n'>{c['images']}</td><td class='n'>{c['auto_rate'] * 100:.0f}%</td>"
        f"<td>{esc(c['note'])}</td></tr>"
        for c in data["children"]
    ) or "<tr><td colspan=5 class='dim'>이 작업공간을 기반으로 만든 작업공간이 아직 없습니다.</td></tr>"

    model = ws.model or {}
    model_html = ("<br>".join(f"{esc(k)}: {esc(v)}" for k, v in model.items())
                  if model else "<span class='dim'>—</span>")

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>{esc(ws.name)} 보고서</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ background:#17171a; color:#e6e6e6; margin:0; padding:32px;
        font:14px/1.6 "Segoe UI","Malgun Gothic",system-ui,sans-serif; }}
 .wrap {{ max-width:1000px; margin:0 auto; }}
 h1 {{ font-size:24px; margin:0 0 4px; }}
 h2 {{ font-size:16px; margin:32px 0 10px; color:#9fd3ff;
       border-bottom:1px solid #2e2e34; padding-bottom:6px; }}
 .sub {{ color:#8a8a92; margin-bottom:8px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
          gap:12px; margin:12px 0; }}
 .card {{ background:#212126; border:1px solid #2e2e34; border-radius:8px; padding:12px 14px; }}
 .k {{ color:#8a8a92; font-size:12px; }}
 .v {{ font-size:22px; font-weight:600; margin-top:2px; }}
 .bar {{ height:5px; background:#2e2e34; border-radius:3px; margin-top:8px; overflow:hidden; }}
 .bar span {{ display:block; height:100%; }}
 table {{ border-collapse:collapse; width:100%; margin:8px 0; }}
 th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid #2a2a30; }}
 th {{ color:#8a8a92; font-weight:500; font-size:12px; }}
 td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .dim {{ color:#6f6f78; }}
 .ok {{ color:#7ec97e; }} .bad {{ color:#e0755e; }}
 code {{ background:#212126; padding:1px 5px; border-radius:4px; color:#c9d7a0; }}
</style></head><body><div class="wrap">

<h1>{esc(ws.name)}</h1>
<div class="sub">{esc(data['kind_desc'])} · 소스 <code>{esc(os.path.basename(ws.source))}</code>
 · 생성 {esc(ws.created)}{' · ' + esc(ws.note) if ws.note else ''}</div>

<h2>한눈에</h2>
<div class="grid">
  <div class="card"><div class="k">라벨링한 이미지</div>
    <div class="v">{t['images']:,}장</div>
    <div class="dim">작업목록 {t['done']}/{t['total']}</div></div>
  <div class="card"><div class="k">박스</div><div class="v">{t['boxes']:,}개</div>
    <div class="dim">사람 A+B {human:,} · 자동 C+D {auto:,}</div></div>
  <div class="card"><div class="k">사람 편집 비용</div>
    <div class="v">{saved:.0f}% 절감</div>
    <div class="dim">{fmt_hms(t['cost_sec'])} / 수동환산 {fmt_hms(t['manual_sec'])}</div>
    {_bar(saved, saved >= 65)}</div>
  <div class="card"><div class="k">무편집 통과율</div>
    <div class="v">{t['auto_rate'] * 100:.0f}%</div>
    {_bar(t['auto_rate'] * 100)}</div>
</div>

<h2>사람이 한 일</h2>
<table>
  <tr><th>동작</th><th>횟수</th><th>단가</th><th>비용</th></tr>
  <tr><td>새로 그리기</td><td class='n'>{ops.drawn:,}</td><td class='n'>{ws.cost.draw:.0f}s</td>
      <td class='n'>{fmt_hms(ops.drawn * ws.cost.draw)}</td></tr>
  <tr><td>삭제</td><td class='n'>{ops.deleted:,}</td><td class='n'>{ws.cost.delete:.0f}s</td>
      <td class='n'>{fmt_hms(ops.deleted * ws.cost.delete)}</td></tr>
  <tr><td>위치·클래스 조정</td><td class='n'>{ops.adjusted:,}</td>
      <td class='n'>{ws.cost.adjust:.0f}s</td>
      <td class='n'>{fmt_hms(ops.adjusted * ws.cost.adjust)}</td></tr>
  <tr><td>손대지 않음</td><td class='n'>{ops.kept:,}</td><td class='n'>0s</td>
      <td class='n'>0초</td></tr>
  <tr><td class='dim'>예측 승격(클릭 수락)</td><td class='n dim'>{t['promoted']:,}</td>
      <td colspan=2 class='dim'>비용 모델은 '그리기'로 계상 — 실제로는 더 쌈</td></tr>
</table>

<h2>시간</h2>
<div class="grid">
  <div class="card"><div class="k">실측 총 소요</div>
    <div class="v">{fmt_hms(ops.seconds)}</div></div>
  <div class="card"><div class="k">이미지당 중앙값</div>
    <div class="v">{t['median_sec']:.0f}초</div>
    <div class="dim">평균 {t['mean_sec']:.0f}초</div></div>
  <div class="card"><div class="k">모델 추정 비용</div>
    <div class="v">{fmt_hms(t['cost_sec'])}</div>
    <div class="dim">실측 대비
      {(t['cost_sec'] / ops.seconds * 100) if ops.seconds else 0:.0f}%</div></div>
  <div class="card"><div class="k">작업 기간</div>
    <div class="v" style="font-size:15px">{esc(t['first_ts'][:16]) or '—'}</div>
    <div class="dim">→ {esc(t['last_ts'][:16]) or '—'}</div></div>
</div>

{score_html}

<h2>출처 분해</h2>
<table><tr><th>태그</th><th>뜻</th><th>박스</th></tr>{rows_prov}</table>

<h2>클래스 분포</h2>
<table><tr><th>id</th><th>이름</th><th>박스</th></tr>{rows_cls}</table>

<h2>계보</h2>
<table>
  <tr><th>부모 작업공간</th><td colspan=4>{esc(ws.parent) if ws.parent else "<span class='dim'>없음 (여기서 시작)</span>"}</td></tr>
  <tr><th>시드 방식</th><td colspan=4>{esc(data['seed_desc'])}</td></tr>
  <tr><th>모델</th><td colspan=4>{model_html}</td></tr>
</table>
<table>
  <tr><th>이 작업공간을 기반으로 만든 것</th><th>종류</th><th>이미지</th><th>무편집률</th><th>메모</th></tr>
  {kids}
</table>

<p class="dim" style="margin-top:32px">생성 {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}
 · YOLO Dataset Studio</p>
</div></body></html>"""


def write_report(
    ws: Workspace, project: Optional["Project"] = None,
    siblings: Sequence[Workspace] = ()
) -> str:
    data = build_report(ws, project, siblings)
    html = render_report_html(data)
    with open(ws.report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return ws.report_path


# ------------------------------------------------------------------- 내보내기
SPLIT_ALIASES = (("train", "train"), ("valid", "valid"), ("val", "valid"),
                 ("test", "test"))


def canon_split(name: str) -> str:
    """'vest-helmet/valid' 같은 이름을 train/valid/test 로 정규화."""
    low = name.lower()
    for needle, canon in SPLIT_ALIASES:
        if needle in low:
            return canon
    return "train"


@dataclass
class ExportPlan:
    out_dir: str
    project: "Project"
    sources: List[Workspace]  # 우선순위 순 — 앞쪽이 이깁니다
    split_mode: str = "keep"  # keep = 원본 스플릿 유지, ratio = 재분할
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    include_unlabeled: bool = False  # 라벨 없는 이미지를 배경(빈 txt)으로 포함
    class_names: Optional[List[str]] = None

    def collect(self) -> List[dict]:
        """내보낼 항목 목록. 각 항목 = 이미지 하나 + 그 라벨의 출처."""
        items: List[dict] = []
        for sp in self.project.splits:
            for fname in sp.image_files:
                src_ws = None
                for ws in self.sources:
                    if ws.has_label(sp.name, fname):
                        src_ws = ws
                        break
                if src_ws is None and not self.include_unlabeled:
                    continue
                items.append({
                    "split": sp.name,
                    "image": fname,
                    "image_path": os.path.join(sp.images_dir, fname),
                    "label_path": src_ws.label_path(sp.name, fname) if src_ws else None,
                    "workspace": src_ws.name if src_ws else None,
                })
        return items

    def assign_splits(self, items: List[dict]) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {"train": [], "valid": [], "test": []}
        if self.split_mode == "keep":
            for it in items:
                out[canon_split(it["split"])].append(it)
            return out
        order = list(items)
        random.Random(self.seed).shuffle(order)
        n = len(order)
        r = self.ratios
        tot = sum(r) or 1.0
        n_tr = int(round(n * r[0] / tot))
        n_va = int(round(n * r[1] / tot))
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        out["train"] = order[:n_tr]
        out["valid"] = order[n_tr:n_tr + n_va]
        out["test"] = order[n_tr + n_va:]
        return out

    def run(self, progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
        items = self.collect()
        buckets = self.assign_splits(items)
        names = self.class_names or self.project.names

        os.makedirs(self.out_dir, exist_ok=True)
        total = sum(len(v) for v in buckets.values())
        done = 0
        collisions = 0
        seen: set = set()
        for split, rows in buckets.items():
            if not rows:
                continue
            idir = os.path.join(self.out_dir, split, "images")
            ldir = os.path.join(self.out_dir, split, "labels")
            os.makedirs(idir, exist_ok=True)
            os.makedirs(ldir, exist_ok=True)
            for it in rows:
                # 스플릿이 다른데 파일명이 같을 수 있다(재분할 시 한 폴더로 합쳐짐).
                base = it["image"]
                key = (split, base.lower())
                if key in seen:
                    collisions += 1
                    root_, ext = os.path.splitext(base)
                    base = f"{root_}__{safe_name(it['split'])}{ext}"
                seen.add((split, base.lower()))

                shutil.copy2(it["image_path"], os.path.join(idir, base))
                lp = os.path.join(ldir, stem(base) + ".txt")
                if it["label_path"] and os.path.isfile(it["label_path"]):
                    shutil.copy2(it["label_path"], lp)
                else:
                    open(lp, "w", encoding="utf-8").close()  # 배경 이미지
                it["out_name"] = base
                it["out_split"] = split
                done += 1
                if progress and done % 25 == 0:
                    progress(done, total, base)

        with open(os.path.join(self.out_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("# YOLO Dataset Studio 내보내기\n")
            f.write("train: ../train/images\n")
            f.write("val: ../valid/images\n")
            f.write("test: ../test/images\n\n")
            f.write(f"nc: {len(names)}\n")
            f.write("names: [" + ", ".join(f"'{n}'" for n in names) + "]\n")

        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.project.root,
            "workspaces": [{"name": w.name, "path": w.path, "kind": w.kind}
                           for w in self.sources],
            "split_mode": self.split_mode,
            "ratios": list(self.ratios), "seed": self.seed,
            "include_unlabeled": self.include_unlabeled,
            "counts": {k: len(v) for k, v in buckets.items()},
            "from_workspace": {},
            "renamed": collisions,
            "items": [{"out_split": it.get("out_split"), "out": it.get("out_name"),
                       "src_split": it["split"], "src": it["image"],
                       "ws": it["workspace"]} for it in items],
        }
        for it in items:
            k = it["workspace"] or "(라벨 없음)"
            manifest["from_workspace"][k] = manifest["from_workspace"].get(k, 0) + 1
        with open(os.path.join(self.out_dir, "export_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)

        if progress:
            progress(total, total, "완료")
        return manifest

# ==============================================================================
# MODULE 9 : gui  --  Tkinter 애플리케이션
# ==============================================================================
class App(tk.Tk):
    PREFETCH_AHEAD = 12
    PREFETCH_BEHIND = 4
    CACHE_CAPACITY = 128
    UNDO_DEPTH = 60
    # 한 장에 이보다 오래 머물면 자리를 비운 것으로 보고 소요시간 계측에서 잘라낸다.
    # 안 자르면 점심 먹고 온 한 장이 라운드 전체 평균을 삼켜버린다.
    MAX_IMAGE_SECONDS = 180.0
    MIN_BOX_PX = 3  # 이보다 작게 그린 박스는 오클릭으로 간주

    def __init__(self, initial_root: Optional[str] = None):
        super().__init__()
        self.title("YOLO Dataset Studio")
        self.geometry("1560x920")
        self.minsize(1100, 680)
        self.configure(bg="#1e1e20")

        self.project: Optional[Project] = None
        self.split: Optional[Split] = None
        self.index: int = 0
        self.filtered: List[int] = []  # 리스트박스 행 -> 스플릿 내 실제 인덱스
        self.shapes: List[Shape] = []
        self.flags: set = set()  # (split, filename)
        self.issues: List[Issue] = []

        self.cache = LRUCache(self.CACHE_CAPACITY)
        self.loader = AsyncImageLoader(self.cache, workers=max(4, (os.cpu_count() or 4) // 2))
        self.view = ViewState()
        self.opts = RenderOpts()

        self._token = 0
        self._cur_img: Optional[np.ndarray] = None
        self._blitter: Optional[CanvasBlitter] = None
        self._pending_render = False
        self._status_t0 = 0.0
        self._status_job = None
        self._resize_job = None
        self._drag: Optional[Tuple[int, int, float, float]] = None
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0
        self._fps = 0.0
        self._validator: Optional[RuleValidator] = None

        # ---- 리뷰/편집 상태 (MODULE 8) -------------------------------------
        self.layout: Optional[Layout] = None  # db / work / export 3구역
        self.ws: Optional[Workspace] = None  # 열려 있는 작업공간(라벨셋)
        self.ghosts: List[Shape] = []  # 예측 오버레이 (읽기 전용)
        self.gt_shapes: List[Shape] = []  # 원본 labels/ 의 내용 (읽기 전용)
        self.baseline: List[Shape] = []  # 편집 비용 기준선
        self.sel: int = -1
        self.cur_cls: int = 0
        self.dirty: bool = False
        self.pred_dir: Optional[str] = None
        # 연속형 학습 워커 (별도 프로세스). 스튜디오가 띄우고 닫을 때 거둔다.
        self._hover: Optional[Tuple[int, int]] = None
        self._worker_proc: Optional[subprocess.Popen] = None
        self._worker_spawn_t: float = 0.0
        self._worker_tries: int = 0
        self.pred_src: str = "C"
        self.hi: float = 0.60
        self.lo: float = 0.25
        self._undo: List[List[Shape]] = []
        self._edit: Optional[dict] = None  # 진행 중인 편집 드래그
        self._draft: Optional[Shape] = None  # 그리는 중인 박스
        # 도구 선택은 키보드 한 줄로 끝난다: ~ 지우개 · 1 첫째 클래스 · 2 둘째 …
        # 지우개는 클래스가 아니라 '도구'이므로 cur_cls 와 별도 상태로 둔다.
        self.erase_mode: bool = False
        # Tab 으로 원본만 볼 때 끄기 전 표시 상태를 담아둔다 (None = 평소)
        self._peek = None
        self._cur_fn: Optional[str] = None  # 자동 저장 대상 판별용
        self._t_open: float = 0.0
        self._promoted: int = 0  # 기각 티어 고스트를 클릭으로 승격시킨 수
        self._worklist_only: bool = False
        # 참조 레이어 — 원하는 만큼 얹는다. 각 항목:
        #   {name, path, per_split, color, filled, dashed, visible, shapes}
        # 예측 폴더(고스트)와 달리 시드/티어에 관여하지 않는 순수 표시용이다.
        self.ref_layers: List[dict] = []
        self._start_menu_shown = False

        self._build_style()
        self._build_ui()
        self._bind_keys()
        self.after(15, self._pump_results)
        # 연속형 라벨링: 백그라운드 워커가 새 모델을 냈는지 주기적으로 본다.
        # 폴더 하나 읽는 일이라 3초 간격이면 화면에 영향이 없다.
        self.after(3000, self._live_poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if initial_root:
            self.after(80, lambda: self.load_project(initial_root))
        # 기능이 늘어 '무엇부터 눌러야 하는지'가 불분명해졌다. 할 일을 먼저 고르게 한다.
        self.after(400, self._maybe_start_menu)

    # ---------------------------------------------------------------- style
    def _build_style(self) -> None:
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        BG, FG, AC = "#1e1e20", "#e6e6e6", "#2d2d31"
        SEL = "#4a6da7"        # 선택 강조. 윈도우 기본 파랑 대신 이걸로 통일
        DIM = "#8f8f98"        # 비활성 글자. 어두워도 형태는 읽혀야 한다
        st.configure(".", background=BG, foreground=FG, fieldbackground=AC, borderwidth=0)
        # configure 는 '보통 상태'만 정한다. clam 은 상태별 기본값을 따로 갖고
        # 있고 그게 전부 밝은 색이다 -- disabled #dcdad5, active #eeebe7.
        # 둘 다 베이지다. 이걸 안 덮으면 어두운 테마 위에서 마우스만 올려도
        # 베이지 바탕에 흰 글씨가 되어 글자가 통째로 사라진다.
        # 실제로 콤보박스(대상·스플릿·클래스)가 그래서 안 보였다.
        # 뿌리에서 한 번 막아두면 모든 위젯이 덮인다.
        st.map(".",
               background=[("disabled", BG), ("active", "#3c3c42"),
                           ("pressed", "#3c3c42")],
               foreground=[("disabled", DIM)],
               fieldbackground=[("disabled", BG), ("readonly", AC)])
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("TButton", background=AC, foreground=FG, padding=(9, 4))
        st.map("TButton", background=[("active", "#3c3c42")])
        st.configure("TCheckbutton", background=BG, foreground=FG)

        # ---- 콤보박스 ------------------------------------------------------
        # configure 만으로는 안 된다. 이 앱의 콤보박스는 전부 state="readonly"
        # 인데, clam 테마는 readonly 일 때 fieldbackground 를 무시하고 상태맵의
        # 기본값(밝은 회색)을 그린다. 그 결과가 '베이지 배경에 흰 글씨' 였다 --
        # 대상·스플릿·클래스가 화면에서 통째로 사라져 있었다.
        #
        # 그래서 상태를 하나도 빠뜨리지 않고 전부 못박는다. 한 상태라도 비워두면
        # 그 상태에서만 다시 안 보이게 되고, 그건 발견하기 더 어렵다.
        st.configure("TCombobox", fieldbackground=AC, background=AC,
                     foreground=FG, arrowcolor=FG, selectbackground=AC,
                     selectforeground=FG, padding=(4, 3))
        st.map(
            "TCombobox",
            fieldbackground=[("readonly", AC), ("disabled", BG),
                             ("focus", AC), ("!focus", AC)],
            background=[("readonly", AC), ("disabled", BG), ("active", "#3c3c42"),
                        ("pressed", "#3c3c42")],
            foreground=[("readonly", FG), ("disabled", DIM), ("focus", FG)],
            arrowcolor=[("disabled", DIM), ("active", "#ffffff")],
            # readonly 콤보는 값을 '선택된 텍스트'로 그린다. 이 두 줄이 없으면
            # 포커스가 갔을 때만 파란 바탕에 묻힌다.
            selectbackground=[("readonly", AC), ("focus", AC), ("!focus", AC)],
            selectforeground=[("readonly", FG), ("focus", FG), ("!focus", FG)],
        )
        # 펼쳤을 때 나오는 목록은 ttk 가 아니라 순수 tk Listbox 라 스타일이
        # 닿지 않는다. 옵션 데이터베이스로 따로 칠해야 한다.
        for opt, val in (("Listbox.background", AC), ("Listbox.foreground", FG),
                         ("Listbox.selectBackground", SEL),
                         ("Listbox.selectForeground", "#ffffff")):
            self.option_add(f"*TCombobox*{opt}", val)

        # ---- tk(비 ttk) 위젯 기본값 ----------------------------------------
        # Listbox·Entry·Text 는 ttk 스타일이 닿지 않아 만들 때마다 색을 일일이
        # 넘겨야 하는데, 다이얼로그를 하나 추가할 때마다 빠뜨리기 쉽다.
        # 실제로 클래스 필터와 레이어 목록은 선택색을 안 넘겨서 윈도우 기본
        # 파란색(SystemHighlight)이 그대로 나오고 있었다 -- 어두운 배경과
        # 어울리지도 않고 대비도 낮았다.
        # 여기서 기본값을 못박으면 앞으로 만드는 위젯까지 전부 덮인다.
        # _build_style() 이 _build_ui() 보다 먼저 돌므로 순서도 맞다.
        for cls in ("Listbox", "Entry", "Text", "Spinbox"):
            self.option_add(f"*{cls}.selectBackground", SEL)
            self.option_add(f"*{cls}.selectForeground", "#ffffff")
        self.option_add("*Listbox.background", "#252528")
        self.option_add("*Listbox.foreground", "#dcdcdc")
        st.configure("Treeview", background=AC, fieldbackground=AC, foreground=FG,
                     rowheight=21, borderwidth=0)
        st.configure("Treeview.Heading", background="#3a3a40", foreground=FG)
        st.map("Treeview", background=[("selected", SEL)])
        st.configure("Status.TLabel", background="#141416", foreground="#9fd3ff")
        # 보조 설명용 회색. 원래 #8a8a92 였는데 4.9:1 로 빠듯해서 한 단 올렸다.
        # 본문보다 눌러 보이되 읽히기는 해야 한다.
        st.configure("Head.TLabel", background=BG, foreground="#a4a4ae")
        st.configure("Session.TLabel", background="#26262b", foreground="#c9d7a0")
        st.configure("Card.TLabel", background=BG, foreground="#e6e6e6",
                     font=("Segoe UI", 12, "bold"))

    # ------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        # ---- 툴바 ---------------------------------------------------------
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="🏁 무엇을 할까요",
                   command=self.on_start_menu).pack(side=tk.LEFT)
        ttk.Button(bar, text="📚 DB 열기", command=self.on_open_db).pack(side=tk.LEFT)
        ttk.Button(bar, text="📂 폴더 직접 열기",
                   command=self.on_open).pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text="  스플릿").pack(side=tk.LEFT)
        self.cb_split = ttk.Combobox(bar, width=22, state="readonly")
        self.cb_split.pack(side=tk.LEFT, padx=(4, 12))
        self.cb_split.bind("<<ComboboxSelected>>", lambda e: self.on_split_change())

        self.v_shapes = tk.BooleanVar(value=True)
        self.v_names = tk.BooleanVar(value=True)
        self.v_fill = tk.BooleanVar(value=False)
        for txt, var in (("라벨(L)", self.v_shapes), ("이름(N)", self.v_names),
                         ("채우기(F)", self.v_fill)):
            ttk.Checkbutton(bar, text=txt, variable=var,
                            command=self._sync_opts).pack(side=tk.LEFT, padx=3)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(bar, text="🔍 자동 검수 실행", command=self.on_validate).pack(side=tk.LEFT)
        ttk.Button(bar, text="🤖 AI 검수", command=self.on_ai_menu).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="🚩 플래그 내보내기",
                   command=self.on_export_flags).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="↺ 뷰 리셋(R)", command=self.reset_view).pack(side=tk.RIGHT)

        # ---- 리뷰 툴바 (P3 편집 파이프라인) ---------------------------------
        bar2 = ttk.Frame(self, padding=(8, 0, 8, 6))
        bar2.pack(side=tk.TOP, fill=tk.X)

        self.v_edit = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar2, text="✏ 편집 모드(E)", variable=self.v_edit,
                        command=self.toggle_edit).pack(side=tk.LEFT)
        ttk.Label(bar2, text="  클래스").pack(side=tk.LEFT)
        self.cb_cls = ttk.Combobox(bar2, width=16, state="readonly")
        self.cb_cls.pack(side=tk.LEFT, padx=(4, 10))
        self.cb_cls.bind("<<ComboboxSelected>>",
                         lambda e: setattr(self, "cur_cls", self.cb_cls.current()))

        ttk.Separator(bar2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar2, text="🆕 작업공간 만들기",
                   command=self.on_ws_new).pack(side=tk.LEFT)
        ttk.Button(bar2, text="🗂 작업공간 열기",
                   command=self.on_ws_open).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar2, text="📊 보고서",
                   command=self.on_ws_report).pack(side=tk.LEFT)
        ttk.Button(bar2, text="📤 내보내기",
                   command=self.on_export).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar2, text="🔮 예측 폴더",
                   command=self.on_pick_pred_dir).pack(side=tk.LEFT)
        self.v_ghost = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar2, text="예측 표시(G)", variable=self.v_ghost,
                        command=self._sync_opts).pack(side=tk.LEFT, padx=4)
        self.v_gtref = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar2, text="GT 참조(채점용)", variable=self.v_gtref,
                        command=self._on_gtref_toggle).pack(side=tk.LEFT, padx=4)
        self.v_wlonly = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar2, text="작업목록만", variable=self.v_wlonly,
                        command=self._on_wlonly_toggle).pack(side=tk.LEFT, padx=4)

        ttk.Button(bar2, text="💾 저장(Ctrl+S)",
                   command=lambda: self.save_current(force=True)).pack(side=tk.RIGHT)

        # ---- 본문 3분할 ---------------------------------------------------
        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        pane = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # 좌: 파일 목록
        left = ttk.Frame(pane, padding=4)
        pane.add(left, weight=0)
        ttk.Label(left, text="파일 검색", style="Head.TLabel").pack(anchor="w")
        self.e_filter = ttk.Entry(left, width=34)
        self.e_filter.pack(fill=tk.X, pady=(2, 6))
        self.e_filter.bind("<KeyRelease>", lambda e: self._apply_filter())
        lf = ttk.Frame(left)
        lf.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self.lb = tk.Listbox(
            lf, width=38, activestyle="none", exportselection=False,
            bg="#252528", fg="#dcdcdc", selectbackground="#4a6da7",
            highlightthickness=0, borderwidth=0, yscrollcommand=sb.set,
        )
        sb.config(command=self.lb.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lb.bind("<<ListboxSelect>>", self._on_list_select)

        # 중: 캔버스
        center = ttk.Frame(pane)
        pane.add(center, weight=1)
        self.canvas = tk.Canvas(center, bg="#202024", highlightthickness=0, cursor="tcross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._img_id = self.canvas.create_image(0, 0, anchor="nw")
        self._blitter = CanvasBlitter(self.canvas, self._img_id)

        # 커서 태그 — 지금 무슨 클래스가 장전돼 있는지 포인터가 데리고 다닌다.
        # 없으면 박스를 다 그리고 색을 보고서야 알게 된다(순서가 거꾸로다).
        # 캔버스 아이템이라 이미지 재렌더 없이 coords 만 옮기면 되고,
        # 블리터는 이미지 아이템 하나만 재사용하므로 이 둘은 항상 그 위에 남는다.
        self._tag_bg = self.canvas.create_rectangle(
            0, 0, 0, 0, state="hidden", outline="", width=0)
        self._tag_tx = self.canvas.create_text(
            0, 0, anchor="nw", state="hidden", font=("Segoe UI", 9, "bold"))
        # 지우개 드래그 사각형. 이것도 캔버스 아이템이라 이미지 재렌더가 없다.
        self._erase_id = self.canvas.create_rectangle(
            0, 0, 0, 0, state="hidden", outline="#ff5566", width=2, dash=(5, 3))
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", lambda e: self._hide_cursor_tag())

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel(e, 120))
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel(e, -120))
        self.canvas.bind("<ButtonPress-1>", self._on_b1_press)
        self.canvas.bind("<B1-Motion>", self._on_b1_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_b1_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        # 중클릭은 편집 모드에서도 항상 팬. 편집 중에 시야를 옮기려고 모드를
        # 껐다 켜는 동작이 제일 성가시다.
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, "_drag", None))

        # 우: 인스펙터
        right = ttk.Frame(pane, padding=4)
        pane.add(right, weight=0)
        self.lbl_session = ttk.Label(right, text="라운드 없음 · 편집 잠김",
                                     style="Session.TLabel", anchor="w",
                                     padding=(6, 4), justify="left", wraplength=300)
        self.lbl_session.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(right, text="현재 이미지 라벨", style="Head.TLabel").pack(anchor="w")
        self.tv = ttk.Treeview(right, columns=("cls", "geo", "src"),
                               show="headings", height=14)
        self.tv.heading("cls", text="클래스")
        self.tv.heading("geo", text="좌표 (정규화)")
        self.tv.heading("src", text="출처")
        self.tv.column("cls", width=110, anchor="w")
        self.tv.column("geo", width=210, anchor="w")
        self.tv.column("src", width=64, anchor="w")
        self.tv.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        self.tv.bind("<<TreeviewSelect>>", self._on_tv_select)

        ttk.Label(right, text="클래스 필터 (더블클릭=숨김 토글)",
                  style="Head.TLabel").pack(anchor="w")
        self.lb_cls = tk.Listbox(right, height=9, bg="#252528", fg="#dcdcdc",
                                 highlightthickness=0, borderwidth=0, exportselection=False)
        self.lb_cls.pack(fill=tk.X, pady=(2, 6))
        self.lb_cls.bind("<Double-Button-1>", self._on_class_toggle)

        lh = ttk.Frame(right)
        lh.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(lh, text="겹쳐 보기 레이어 (더블클릭=on/off)",
                  style="Head.TLabel").pack(side=tk.LEFT)
        ttk.Button(lh, text="−", width=2,
                   command=self.on_remove_layer).pack(side=tk.RIGHT)
        ttk.Button(lh, text="+", width=2,
                   command=self.on_add_layer).pack(side=tk.RIGHT, padx=2)
        self.lb_layers = tk.Listbox(right, height=5, bg="#252528", fg="#dcdcdc",
                                    highlightthickness=0, borderwidth=0,
                                    exportselection=False)
        self.lb_layers.pack(fill=tk.X, pady=(2, 6))
        self.lb_layers.bind("<Double-Button-1>", self.on_toggle_layer)

        self.lbl_meta = ttk.Label(right, text="", style="Head.TLabel",
                                  justify="left", wraplength=300)
        self.lbl_meta.pack(anchor="w", fill=tk.X)

        # ---- 상태바 -------------------------------------------------------
        self.status = ttk.Label(self, text="루트 폴더를 열어주세요.",
                                style="Status.TLabel", anchor="w", padding=(8, 4))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.pb = ttk.Progressbar(self, mode="determinate")

    TEXT_WIDGETS = (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, tk.Spinbox)

    def _hotkeys_live(self, event=None) -> bool:
        """텍스트 입력 위젯에서 친 글자면 단축키를 삼킨다.

        bind_all 은 Entry 안에서 친 글자까지 전부 잡아간다. 파일 검색창에 'r' 을
        치면 뷰가 리셋되던 버그가 여기서 나왔고, 편집 단축키를 늘리면 파일명에
        숫자만 쳐도 박스 클래스가 바뀌는 사고로 커진다.

        판정은 focus_get() 이 아니라 event.widget 으로 한다. 키 이벤트는 언제나
        포커스를 가진 위젯에서 올라오고, focus_get() 은 창이 아직 매핑되지 않았을
        때 None 을 돌려줘 가드가 통째로 뚫린다.
        """
        w = getattr(event, "widget", None)
        if w is None:
            try:
                w = self.focus_get()
            except (KeyError, tk.TclError):
                return True
        return not isinstance(w, self.TEXT_WIDGETS)

    def _bind_keys(self) -> None:
        def b(seq: str, fn: Callable[[], object], guard: bool = True) -> None:
            def handler(e, fn=fn, guard=guard):
                if guard and not self._hotkeys_live(e):
                    return None
                fn()
                return "break"

            self.bind_all(seq, handler)

        b("<Left>", lambda: self.step(-1))
        b("<Right>", lambda: self.step(1))
        # 왼손 전용 이동. 오른손은 마우스를 잡은 채로 계속 그릴 수 있어야 한다.
        # 목록이 세로로 서 있으므로 w=위=이전, s=아래=다음.
        for k in ("w", "W"):
            b(f"<Key-{k}>", lambda: self.step(-1))
        for k in ("s", "S"):
            b(f"<Key-{k}>", lambda: self.step(1))
        b("<Prior>", lambda: self.step(-10))
        b("<Next>", lambda: self.step(10))
        b("<Home>", lambda: self.goto(0))
        b("<End>", lambda: self.goto(10**9))
        b("<Control-g>", self.on_goto_dialog, guard=False)
        b("<Control-s>", lambda: self.save_current(force=True), guard=False)
        b("<Control-z>", self.undo, guard=False)
        b("<space>", self.toggle_flag)
        b("<Key-r>", self.reset_view)
        b("<Key-R>", self.reset_view)
        for key, var in (("l", self.v_shapes), ("n", self.v_names), ("f", self.v_fill),
                         ("g", self.v_ghost)):
            for k in (key, key.upper()):
                b(f"<Key-{k}>", lambda v=var: (v.set(not v.get()), self._sync_opts()))

        # ---- 편집 단축키 ---------------------------------------------------
        for k in ("e", "E"):
            b(f"<Key-{k}>", lambda: (self.v_edit.set(not self.v_edit.get()),
                                     self.toggle_edit()))
        b("<Delete>", self.delete_selected)
        b("<BackSpace>", self.delete_selected)
        b("<Escape>", lambda: self.select_shape(-1))
        # Tab 은 원본 훔쳐보기. 선 두께가 작은 객체를 덮어 확대해야 보이는 일이
        # 잦은데, 껐다 켜는 편이 확대·축소보다 빠르다. 왼손으로 닿는 자리이기도 하다.
        # 다음 박스 선택은 Shift+Tab 으로 옮겼다.
        #
        # ※ Tab 만은 bind_all 로 못 잡는다. Tk 의 바인딩 순서는
        #   widget -> class -> toplevel -> all 인데, Tab 은 class 단계의 포커스
        #   이동에 먹혀서 'all'(=bind_all)까지 오지 않는다. 실측했더니 어느
        #   위젯에 포커스가 있든 핸들러가 한 번도 안 불렸다. 위젯 단계에 직접
        #   걸면 class 보다 먼저 돌고 "break" 로 포커스 이동까지 막을 수 있다.
        def bw(seq: str, fn: Callable[[], object]) -> None:
            def handler(e, fn=fn):
                if not self._hotkeys_live(e):
                    return None
                fn()
                return "break"
            for w in (self.canvas, self.lb, self.tv, self.lb_cls):
                w.bind(seq, handler)

        bw("<Tab>", self.toggle_peek)
        bw("<Shift-Tab>", lambda: self.cycle_selection(1))
        bw("<ISO_Left_Tab>", lambda: self.cycle_selection(1))
        # 둘 다 Ctrl 을 물린다. 맨 'a' 는 w/s 바로 옆이라 오타 한 번에 사고가 난다.
        #   Ctrl+A        전부 지우기  — Ctrl+Z 와 같은 손모양이라 되돌리기와 짝이다
        #   Ctrl+Shift+A  전부 승격    — 원래 Ctrl+A 였으나 자리를 내줬다
        # 승격을 안쪽으로 미룬 이유: 약한 모델이 장당 3.29개(내포 27)를 뱉는 구간에서는
        # 전부 승격이 전부 지우기보다 훨씬 위험하다. 지운 것은 화면이 비어 바로 보이지만
        # 쏟아진 것은 겹쳐 있어 안 보인다.
        b("<Control-a>", self.clear_all, guard=False)
        b("<Control-Shift-A>", self.accept_ghosts, guard=False)
        b("<Return>", self.next_pending_image)
        # 도구는 키보드 맨 윗줄 하나로 끝난다 — 물리적 배열이 곧 도구 순서다.
        #   ~  지우개    1  첫째 클래스    2  둘째 클래스  …
        # 숫자키는 1부터. 키보드가 1 로 시작하고 0 은 멀다. (1~9 -> 0~8, 0 -> 9)
        for k in ("asciitilde", "grave"):   # Shift 유무에 상관없이 같은 키
            b(f"<Key-{k}>", lambda: self.set_erase(not self.erase_mode))
        for d in range(1, 10):
            b(f"<Key-{d}>", lambda d=d: self.set_class(d - 1))
        b("<Key-0>", lambda: self.set_class(9))

    # ------------------------------------------------------------- 프로젝트
    def on_open(self) -> None:
        d = filedialog.askdirectory(title="데이터셋 루트 폴더 선택")
        if d:
            self.load_project(d)

    def load_project(self, root: str) -> None:
        self.status.config(text=f"스캔 중… {root}")
        self.update_idletasks()
        try:
            proj = Project(root)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("로드 실패", f"{e}")
            return
        if not proj.splits:
            messagebox.showwarning(
                "구조 불일치",
                "images/ 와 labels/ 가 나란히 있는 폴더를 찾지 못했습니다.\n"
                "스크립트 상단의 [지원하는 폴더 트리 구조] 주석을 확인하세요.",
            )
            return

        self.project = proj
        self.cache.clear()
        self.flags.clear()
        self.issues.clear()
        self._close_workspace()
        self.layout = Layout.discover(proj.root)

        self.cb_split["values"] = [f"{s.name}  ({len(s)}장)" for s in proj.splits]
        self.cb_split.current(0)

        self.lb_cls.delete(0, tk.END)
        self.opts.hidden_classes.clear()
        for i, n in enumerate(proj.names):
            b, g, r = proj.color_of(i)
            # 1~9 -> 클래스 0~8, 0 -> 클래스 9 이므로 표시도 그 규칙을 따른다
            key = str(i + 1) if i < 9 else ("0" if i == 9 else None)
            self.lb_cls.insert(tk.END,
                               f"  {i}: {n}   (키 {key})" if key else f"  {i}: {n}")
            self.lb_cls.itemconfig(i, foreground=f"#{r:02x}{g:02x}{b:02x}")
        self.cb_cls["values"] = [f"{i}: {n}" for i, n in enumerate(proj.names)]
        self.cur_cls = 0
        if proj.names:
            self.cb_cls.current(0)

        self.title(f"YOLO Dataset Studio  —  {proj.name}")
        self.on_split_change()

    def on_split_change(self) -> None:
        if not self.project:
            return
        self.save_current()
        self.split = self.project.splits[self.cb_split.current()]
        self.index = 0
        self._cur_fn = None
        # 작업공간은 스플릿을 가리지 않는다. 라벨은 labels/<split>/ 로 나뉘어
        # 저장되므로 train 을 보다 valid 로 넘어가도 그대로 이어서 작업하면 된다.
        self._apply_filter()

    def _apply_filter(self) -> None:
        if not self.split:
            return
        q = self.e_filter.get().strip().lower()
        wl = self.ws.worklist.get(self.split.name) if self.ws else None
        if self._worklist_only and wl:
            # 작업 목록 순서(무작위 셔플)를 그대로 유지해야 라운드 프로토콜이 산다.
            pos = {f: i for i, f in enumerate(self.split.image_files)}
            self.filtered = [pos[f] for f in wl
                             if f in pos and (not q or q in f.lower())]
        else:
            if self._worklist_only and self.ws:
                # 이 스플릿엔 작업 목록이 없다. 빈 화면을 보여주는 대신 전체를
                # 띄우고 왜 그런지 알린다.
                self.status.config(
                    text=f"'{self.split.name}' 은 작업공간 '{self.ws.name}' 의 "
                         f"작업 목록에 없습니다. 전체 목록을 표시합니다.")
            self.filtered = [i for i, f in enumerate(self.split.image_files)
                             if not q or q in f.lower()]
        self._fill_listbox()
        self.index = self.filtered[0] if self.filtered else 0
        self.show_current(sync_list=True)

    def _fill_listbox(self) -> None:
        self.lb.delete(0, tk.END)
        if not self.filtered:
            return
        done = self.ws.records if self.ws else {}
        sp = self.split.name
        self.lb.insert(tk.END, *(
            ("✓ " if (sp, self.split.image_files[i]) in done else "   ")
            + self.split.image_files[i]
            for i in self.filtered
        ))

    def _refresh_row_marks(self, fn: Optional[str] = None) -> None:
        """저장 직후 목록의 완료 표시만 갱신 (전체 재구축 없이).

        갱신할 파일을 인자로 받는 이유: 자동 저장은 show_current 안에서 일어나는데
        그 시점에는 self.index 가 이미 '다음 장'으로 옮겨져 있다. self.index 를 보면
        방금 끝낸 장이 아니라 이제 막 연 장에 ✓ 가 찍힌다 -- 저장은 제대로 됐는데
        표시만 진행 방향으로 한 칸 밀리므로, 위로 갔다 내려올 때만 맞아 보인다.
        """
        if not (self.ws and self.filtered and self.split):
            return
        if fn is None:
            if not (0 <= self.index < len(self.split.image_files)):
                return
            fn = self.split.image_files[self.index]
        try:
            row = self._row_of(self.split.image_files.index(fn))
        except ValueError:
            return
        if row < 0:
            return
        mark = "✓ " if (self.split.name, fn) in self.ws.records else "   "
        if self.lb.get(row) != mark + fn:
            selected = row in self.lb.curselection()
            self.lb.delete(row)
            self.lb.insert(row, mark + fn)
            if selected:            # 지웠다 넣으면 선택이 풀린다. 원래 상태만 복원.
                self.lb.selection_set(row)

    # ----------------------------------------------------------------- 탐색
    def _row_of(self, index: int) -> int:
        try:
            return self.filtered.index(index)
        except ValueError:
            return -1

    def step(self, delta: int) -> None:
        if not self.filtered:
            return
        row = self._row_of(self.index)
        if row < 0:
            row = 0
        row = max(0, min(len(self.filtered) - 1, row + delta))
        self.index = self.filtered[row]
        self.show_current(sync_list=True)

    def goto(self, row: int) -> None:
        if not self.filtered:
            return
        row = max(0, min(len(self.filtered) - 1, row))
        self.index = self.filtered[row]
        self.show_current(sync_list=True)

    def on_goto_dialog(self) -> None:
        top = tk.Toplevel(self)
        top.title("이동")
        top.configure(bg="#1e1e20")
        top.transient(self)
        self._center(top)
        ttk.Label(top, text=f"1 ~ {len(self.filtered)} 사이 번호").pack(padx=12, pady=(12, 4))
        e = ttk.Entry(top, width=14)
        e.pack(padx=12)
        e.focus_set()

        def ok(_=None):
            try:
                self.goto(int(e.get()) - 1)
            except ValueError:
                pass
            top.destroy()

        e.bind("<Return>", ok)
        ttk.Button(top, text="이동", command=ok).pack(pady=10)

    def _on_list_select(self, _=None) -> None:
        sel = self.lb.curselection()
        if sel and self.filtered:
            self.index = self.filtered[sel[0]]
            self.show_current(sync_list=False)

    # ------------------------------------------------------------- 이미지 표시
    def show_current(self, sync_list: bool = False) -> None:
        # 화면을 바꾸기 전에 편집 중이던 것부터 확정한다. 필터를 쳐서 목록이 비는
        # 경우까지 포함해 여기가 유일한 이탈 지점이라 여기서 한 번만 처리한다.
        if not self.split or not self.filtered:
            self.save_current()
            self._cur_fn = None
            self._cur_img = None
            self.shapes = []
            self.ghosts = []
            self.baseline = []
            self._blitter.clear()
            self.status.config(text="표시할 이미지가 없습니다.")
            return

        self._token += 1
        token = self._token
        fn = self.split.image_files[self.index]
        path = self.split.image_path(self.index)

        if self._cur_fn is not None and self._cur_fn != fn:
            # 편집 중이면 고친 게 없어도 확정한다. 다음 장으로 넘어간다는 건
            # 이 장을 끝냈다는 뜻이고, 예측이 전부 맞아서 손댈 게 없었던 경우도
            # '검수했다'로 남아야 한다. 안 그러면 진행률과 시간이 비어버린다.
            self.save_current(force=self._committing())
        if self._cur_fn != fn:
            self._reset_edit_state(fn)

        if sync_list:
            row = self._row_of(self.index)
            if row >= 0:
                self.lb.selection_clear(0, tk.END)
                self.lb.selection_set(row)
                self.lb.see(row)

        # 라벨은 항상 즉시 파싱 (수 KB 이므로 블로킹 무시 가능)
        # db 의 원본 라벨은 데이터셋 원본 주석자가 만든 것이므로 출처 B 로 표시한다.
        # 비워두면 화면에 '-' 로 떠서 "누가 만든 건지 모르는 것"처럼 보인다.
        self.gt_shapes, warns = parse_label_file(
            self.split.label_path_for(fn), default_src="B")
        if self.ws:
            # 작업공간이 열려 있으면 편집 대상은 그쪽 사본이다. db 의 labels/ 는
            # 이 시점부터 읽기 전용 참조로만 남는다.
            self.shapes, _saved = self.ws.load_shapes(
                self.split.name, fn, self.gt_shapes)
            self.baseline = self.ws.baseline_of(self.split.name, fn, self.shapes)
        else:
            self.shapes = self.gt_shapes
            self.baseline = []
        self.ghosts = self._ghosts_for(fn)
        self._load_ref_layers(fn)
        self._refresh_layer_list()
        self._fill_inspector(warns)
        self._update_session_label()

        img = self.loader.request(path, token)
        self._queue_prefetch()

        if img is None:  # 캐시 미스 -> 논블로킹 대기. 방향키는 계속 먹는다.
            self.status.config(text=f"[{self._row_of(self.index) + 1}/{len(self.filtered)}] "
                                    f"{fn}  ·  로딩…")
            return
        self._cur_img = img
        if self.view.fitted:
            Renderer.fit(self.view, (img.shape[1], img.shape[0]), self._canvas_wh())
        self.request_render()

    def _queue_prefetch(self) -> None:
        if not self.split or not self.filtered:
            return
        row = self._row_of(self.index)
        if row < 0:
            return
        lo = max(0, row - self.PREFETCH_BEHIND)
        hi = min(len(self.filtered), row + self.PREFETCH_AHEAD + 1)
        self.loader.prefetch(
            self.split.image_path(self.filtered[r]) for r in range(lo, hi) if r != row
        )

    def _pump_results(self) -> None:
        """워커 스레드 결과를 메인 루프로 끌어온다. 만료 토큰은 조용히 폐기."""
        try:
            while True:
                token, path, img = self.loader.result_q.get_nowait()
                if token != self._token or not self.split:
                    continue
                if img is None:
                    self.status.config(text=f"⚠ 디코딩 실패: {os.path.basename(path)}")
                    continue
                self._cur_img = img
                if self.view.fitted:
                    Renderer.fit(self.view, (img.shape[1], img.shape[0]), self._canvas_wh())
                self.request_render()
        except queue.Empty:
            pass
        self.after(15, self._pump_results)

    # ----------------------------------------------------------------- 렌더
    def _canvas_wh(self) -> Tuple[int, int]:
        return max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)

    def request_render(self) -> None:
        """연타 시 프레임을 합쳐(coalesce) 실제 렌더 횟수를 줄인다."""
        if self._pending_render:
            return
        self._pending_render = True
        self.after_idle(self._do_render)

    def _do_render(self) -> None:
        self._pending_render = False
        if self._cur_img is None or not self.project:
            return
        shapes = self.shapes if self._draft is None else [*self.shapes, self._draft]
        frame = Renderer.render(self._cur_img, shapes, self.project,
                                self.view, self._canvas_wh(), self.opts,
                                layers=self._overlays())
        self._blitter.blit(frame)
        self._tick_fps()
        # 상태바는 Tk 위젯 갱신이라 프레임마다 때리면 낭비 -> 100ms 스로틀.
        # 스킵된 갱신은 trailing 타이머로 보정해 최종 상태는 항상 정확하게 남는다.
        now = time.perf_counter()
        if now - self._status_t0 >= 0.1:
            self._status_t0 = now
            self._update_status()
        else:
            if self._status_job:
                self.after_cancel(self._status_job)
            self._status_job = self.after(120, self._flush_status)

    def _flush_status(self) -> None:
        self._status_job = None
        self._status_t0 = time.perf_counter()
        self._update_status()

    def _tick_fps(self) -> None:
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 0.5:
            self._fps = self._fps_n / (now - self._fps_t0)
            self._fps_n, self._fps_t0 = 0, now

    # ---- 커서 태그 --------------------------------------------------------
    CURSOR_TAG_DY = 16      # 포인터 아래로 이만큼. 십자 커서와 안 겹치는 최소값.
    CURSOR_TAG_DX = 12

    def _hide_cursor_tag(self) -> None:
        for i in (self._tag_bg, self._tag_tx):
            self.canvas.itemconfigure(i, state="hidden")

    def _on_hover(self, event) -> None:
        self._hover = (event.x, event.y)
        self._place_cursor_tag()

    def _place_cursor_tag(self) -> None:
        """장전된 클래스를 포인터 '아래'에 띄운다.

        위나 옆이면 지금 그리려는 영역을 가린다. 아래로 두면 드래그를 시작하는
        지점(좌상단)에서 시야를 안 막는다. 화면 가장자리에서는 반대편으로 접는다.
        """
        if not (self.opts.edit_mode and self.project and self._hover):
            self._hide_cursor_tag()
            return
        x, y = self._hover
        if self.erase_mode:
            label, fill = "[~]  지우개", "#ff5566"
        else:
            key = self.cur_cls + 1 if self.cur_cls < 9 else 0
            label = f"[{key}]  {self.project.class_name(self.cur_cls)}"
            b, g, r = class_color(self.cur_cls)
            fill = "#%02x%02x%02x" % (r, g, b)
        self.canvas.itemconfigure(self._tag_tx, text=label, fill=fill, state="normal")
        self.canvas.coords(self._tag_tx, x + self.CURSOR_TAG_DX,
                           y + self.CURSOR_TAG_DY)
        bb = self.canvas.bbox(self._tag_tx)
        if not bb:
            return
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        cw, ch = self._canvas_wh()
        tx = x + self.CURSOR_TAG_DX
        ty = y + self.CURSOR_TAG_DY
        if tx + w + 6 > cw:                     # 오른쪽 끝 -> 왼쪽으로 접는다
            tx = x - self.CURSOR_TAG_DX - w
        if ty + h + 6 > ch:                     # 아래쪽 끝 -> 위로 접는다
            ty = y - self.CURSOR_TAG_DY - h
        self.canvas.coords(self._tag_tx, tx, ty)
        self.canvas.coords(self._tag_bg, tx - 5, ty - 3, tx + w + 5, ty + h + 3)
        self.canvas.itemconfigure(self._tag_bg, fill="#16161a", state="normal")
        self.canvas.tag_raise(self._tag_bg)
        self.canvas.tag_raise(self._tag_tx)

    def _update_status(self) -> None:
        if not self.split or not self.filtered:
            return
        fn = self.split.image_files[self.index]
        row = self._row_of(self.index) + 1
        h, w = self._cur_img.shape[:2] if self._cur_img is not None else (0, 0)
        flag = " 🚩" if (self.split.name, fn) in self.flags else ""
        # 편집 중에는 '지금 무슨 클래스로 그리는가'가 제일 중요한 정보다.
        # 그리기 전에 클래스를 정하는 흐름이라 화면 어딘가에 항상 떠 있어야 한다.
        draw_as = ""
        if self.opts.edit_mode and self.project:
            if self.erase_mode:
                draw_as = "   ·   🩹 지우개 [~] — 드래그로 감싸 지우기"
            else:
                key = self.cur_cls + 1 if self.cur_cls < 9 else 0
                draw_as = (f"   ·   ✏ 그리기: [{key}] "
                           f"{self.project.class_name(self.cur_cls)}")
        self.status.config(
            text=(f"[{row}/{len(self.filtered)}] {fn}{flag}   ·   {w}×{h}px"
                  f"   ·   박스 {len(self.shapes)}개   ·   줌 {self.view.scale * 100:.0f}%"
                  f"{draw_as}"
                  f"   ·   캐시 {len(self.cache)}/{self.CACHE_CAPACITY}"
                  f"   ·   {self._fps:.0f} fps")
        )

    def _sync_opts(self) -> None:
        self.opts.show_shapes = self.v_shapes.get()
        self.opts.show_names = self.v_names.get()
        self.opts.fill = self.v_fill.get()
        self.opts.show_ghosts = self.v_ghost.get()
        self.opts.edit_mode = self.v_edit.get()
        self._place_cursor_tag()   # 편집 모드를 끄면 커서 태그도 사라진다
        self.request_render()

    def reset_view(self) -> None:
        if self._cur_img is not None:
            Renderer.fit(self.view, (self._cur_img.shape[1], self._cur_img.shape[0]),
                         self._canvas_wh())
            self.request_render()

    # --------------------------------------------------------- 마우스 인터랙션
    def _on_canvas_resize(self, _=None) -> None:
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(70, self._after_resize)

    def _after_resize(self) -> None:
        self._resize_job = None
        if self._cur_img is not None and self.view.fitted:
            Renderer.fit(self.view, (self._cur_img.shape[1], self._cur_img.shape[0]),
                         self._canvas_wh())
        self.request_render()

    def _on_wheel(self, event, forced_delta: Optional[int] = None) -> None:
        if self._cur_img is None:
            return
        delta = forced_delta if forced_delta is not None else event.delta
        factor = 1.18 if delta > 0 else 1 / 1.18
        cw, ch = self._canvas_wh()
        # 커서 아래 이미지 좌표를 고정점으로 유지
        ix = self.view.cx + (event.x - cw / 2) / self.view.scale
        iy = self.view.cy + (event.y - ch / 2) / self.view.scale
        self.view.scale = max(0.02, min(60.0, self.view.scale * factor))
        self.view.cx = ix - (event.x - cw / 2) / self.view.scale
        self.view.cy = iy - (event.y - ch / 2) / self.view.scale
        self.view.fitted = False
        self.request_render()

    def _on_drag_start(self, event) -> None:
        self._drag = (event.x, event.y, self.view.cx, self.view.cy)

    def _on_drag_move(self, event) -> None:
        if not self._drag:
            return
        x0, y0, cx0, cy0 = self._drag
        self.view.cx = cx0 - (event.x - x0) / self.view.scale
        self.view.cy = cy0 - (event.y - y0) / self.view.scale
        self.view.fitted = False
        self.request_render()

    # ==========================================================================
    #  편집 (P3)
    # ==========================================================================
    # ---- 레이어 -----------------------------------------------------------
    def _ghosts_for(self, fn: str) -> List[Shape]:
        """예측 오버레이. 세션이 있으면 세션의 임계값/출처 설정을 따른다."""
        if self.ws:
            return self.ws.ghosts_for(fn)
        if not self.pred_dir:
            return []
        p = os.path.join(self.pred_dir, stem(fn) + ".txt")
        preds, _ = parse_label_file(p, default_src=self.pred_src)
        for s in preds:
            s.tier = tier_of(s.conf, self.hi, self.lo)
        return preds

    def _visible_ghosts(self) -> List[Shape]:
        out = list(self.ghosts)
        if self.v_gtref.get() and self.ws:
            # GT 참조는 기본 꺼짐. 켜면 사람이 정답을 보고 베끼게 되므로 이때의
            # 결과는 시뮬레이션(출처 B)으로만 의미가 있다.
            for g in self.gt_shapes:
                s = g.clone()
                s.src, s.tier, s.conf = "B", "reject", -1.0
                out.append(s)
        return out

    # ---- 참조 레이어 -------------------------------------------------------
    LAYER_COLORS = [(90, 200, 255), (255, 160, 90), (170, 120, 255),
                    (120, 230, 160), (255, 120, 200), (90, 240, 240)]

    def _overlays(self) -> List[Overlay]:
        """렌더러에 넘길 오버레이 목록. 예측 층이 맨 아래, 참조 층이 그 위."""
        out = [Overlay(self._visible_ghosts())]
        for i, L in enumerate(self.ref_layers):
            if not L.get("visible", True) or not L.get("shapes"):
                continue
            out.append(Overlay(
                L["shapes"],
                color=L.get("color") or self.LAYER_COLORS[i % len(self.LAYER_COLORS)],
                filled=L.get("filled", True),
                dashed=L.get("dashed", False),
                width=2,
                label=L.get("label", False),
            ))
        return out

    def _load_ref_layers(self, fn: str) -> None:
        """현재 이미지에 대한 각 참조 레이어의 라벨을 읽어둔다."""
        if not self.split:
            return
        for L in self.ref_layers:
            base = L["path"]
            p = (os.path.join(base, safe_name(self.split.name), stem(fn) + ".txt")
                 if L.get("per_split") else
                 os.path.join(base, stem(fn) + ".txt"))
            shapes, _ = parse_label_file(p, default_src=L.get("src", "?"))
            L["shapes"] = shapes

    def on_add_layer(self) -> None:
        """라벨 폴더를 참조 레이어로 얹는다."""
        d = filedialog.askdirectory(title="겹쳐 볼 라벨 폴더 선택")
        if not d:
            return
        # 스플릿별 하위 폴더 구조인지 자동 판별
        per_split = False
        if self.split:
            cand = os.path.join(d, safe_name(self.split.name))
            per_split = os.path.isdir(cand)
        n_txt = 0
        probe = os.path.join(d, safe_name(self.split.name)) if per_split else d
        try:
            n_txt = sum(1 for e in os.scandir(probe)
                        if e.is_file() and e.name.endswith(".txt"))
        except OSError:
            pass
        if n_txt == 0:
            messagebox.showwarning(
                "레이어", f"txt 라벨을 찾지 못했습니다:\n{probe}")
            return
        self.ref_layers.append({
            "name": os.path.basename(os.path.normpath(d)) or d,
            "path": d, "per_split": per_split, "visible": True,
            "filled": True, "dashed": False, "label": False,
            "src": "?", "color": None, "shapes": [],
        })
        self._refresh_layer_list()
        if self.filtered:
            self._load_ref_layers(self.split.image_files[self.index])
        self.status.config(text=f"레이어 추가: {os.path.basename(d)} "
                                f"(txt {n_txt}개{' · 스플릿별' if per_split else ''})")
        self.request_render()

    def on_remove_layer(self) -> None:
        sel = self.lb_layers.curselection()
        if not sel:
            return
        L = self.ref_layers.pop(sel[0])
        self._refresh_layer_list()
        self.status.config(text=f"레이어 제거: {L['name']}")
        self.request_render()

    def on_toggle_layer(self, _=None) -> None:
        sel = self.lb_layers.curselection()
        if not sel:
            return
        L = self.ref_layers[sel[0]]
        L["visible"] = not L.get("visible", True)
        self._refresh_layer_list()
        self.request_render()

    def _refresh_layer_list(self) -> None:
        self.lb_layers.delete(0, tk.END)
        for i, L in enumerate(self.ref_layers):
            mark = "☑" if L.get("visible", True) else "☐"
            n = len(L.get("shapes") or [])
            self.lb_layers.insert(tk.END, f" {mark} {L['name']}  ({n})")
            b, g, r = L.get("color") or self.LAYER_COLORS[i % len(self.LAYER_COLORS)]
            self.lb_layers.itemconfig(
                i, foreground=f"#{r:02x}{g:02x}{b:02x}" if L.get("visible", True)
                else "#565660")

    def _can_edit(self) -> bool:
        """작업공간 없이는 편집 불가. db 를 실수로 고치는 경로 자체를 없앤다."""
        if self.ws is None:
            self.status.config(
                text="⚠ 편집하려면 먼저 [🆕 작업공간 만들기] 로 라벨셋을 만드세요. "
                     "db/ 의 원본 라벨은 채점용이라 이 툴에서 절대 수정되지 않습니다."
            )
            return False
        return self._cur_img is not None

    def toggle_edit(self) -> None:
        if self.v_edit.get() and self.ws is None:
            self.v_edit.set(False)
            self._can_edit()
        self.canvas.config(cursor="crosshair" if self.v_edit.get() else "tcross")
        self._sync_opts()

    def _reset_edit_state(self, fn: str) -> None:
        self._cur_fn = fn
        self._t_open = time.perf_counter()
        self._undo.clear()
        self._draft = None
        self._edit = None
        self._promoted = 0
        self.dirty = False
        self.select_shape(-1, render=False)

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _committing(self) -> bool:
        """이동할 때 자동으로 확정할 상황인가.

        작업공간을 열고 편집 모드인 동안만. 그냥 둘러보는 중에는 건드리지 않는다.
        """
        return self.ws is not None and self.opts.edit_mode

    # ---- 좌표 / 히트 테스트 ------------------------------------------------
    def _canvas_to_norm(self, X: float, Y: float) -> Tuple[float, float]:
        H, W = self._cur_img.shape[:2]
        px, py = Renderer.to_image(self.view, self._canvas_wh(), X, Y)
        return px / max(W, 1), py / max(H, 1)

    def _hit(self, X: float, Y: float) -> Tuple[str, int, str]:
        """반환 (종류, shape 인덱스, 핸들이름). 종류 = handle | body | none."""
        if self._cur_img is None:
            return ("none", -1, "")
        cwh = self._canvas_wh()
        H, W = self._cur_img.shape[:2]

        # 1) 선택 박스의 핸들이 최우선 — 겹친 박스 위에서도 크기 조절이 되어야 한다.
        if 0 <= self.sel < len(self.shapes) and self.shapes[self.sel].kind == "box":
            x0, y0, x1, y1 = shape_xyxy(self.shapes[self.sel])
            a = Renderer.to_canvas(self.view, cwh, x0 * W, y0 * H)
            b = Renderer.to_canvas(self.view, cwh, x1 * W, y1 * H)
            for name, (hx, hy) in handle_points(a[0], a[1], b[0], b[1]).items():
                if abs(X - hx) <= HANDLE_HIT and abs(Y - hy) <= HANDLE_HIT:
                    return ("handle", self.sel, name)

        # 2) 포함하는 박스 중 가장 작은 것 — 큰 박스 안의 작은 박스가 잡히도록.
        nx, ny = self._canvas_to_norm(X, Y)
        best, best_area = -1, float("inf")
        for i, sh in enumerate(self.shapes):
            if sh.cls in self.opts.hidden_classes:
                continue
            x0, y0, x1, y1 = shape_xyxy(sh)
            if x0 <= nx <= x1 and y0 <= ny <= y1:
                area = (x1 - x0) * (y1 - y0)
                if area <= best_area:
                    best, best_area = i, area
        return ("body", best, "") if best >= 0 else ("none", -1, "")

    def _hit_ghost(self, X: float, Y: float) -> int:
        nx, ny = self._canvas_to_norm(X, Y)
        best, best_area = -1, float("inf")
        for i, sh in enumerate(self.ghosts):
            if sh.cls in self.opts.hidden_classes:
                continue
            x0, y0, x1, y1 = shape_xyxy(sh)
            if x0 <= nx <= x1 and y0 <= ny <= y1:
                area = (x1 - x0) * (y1 - y0)
                if area <= best_area:
                    best, best_area = i, area
        return best

    # ---- 마우스 -----------------------------------------------------------
    def _on_b1_press(self, event) -> None:
        self._drag = None
        self._edit = None
        if self._cur_img is None:
            return
        # 편집 모드가 꺼져 있거나 Ctrl 을 누른 채면 예전처럼 팬.
        if not self.opts.edit_mode or (event.state & 0x0004):
            self._on_drag_start(event)
            return
        if not self._can_edit():
            self._on_drag_start(event)
            return

        if self.erase_mode:
            # 지우개는 히트 테스트를 거치지 않는다. 박스 위에서 시작해도 이동/리사이즈가
            # 아니라 지우기 드래그여야 한다 -- 지우려는 것 위에서 시작하는 게 자연스럽다.
            nx, ny = self._canvas_to_norm(event.x, event.y)
            self._edit = {"op": "erase", "anchor": (nx, ny),
                          "c0": (event.x, event.y), "pushed": False}
            self.canvas.coords(self._erase_id, event.x, event.y, event.x, event.y)
            self.canvas.itemconfigure(self._erase_id, state="normal")
            self.canvas.tag_raise(self._erase_id)
            return

        kind, idx, hname = self._hit(event.x, event.y)
        if kind == "handle":
            self._edit = {"op": "resize", "idx": idx, "h": hname,
                          "box": shape_xyxy(self.shapes[idx]), "pushed": False}
        elif kind == "body":
            self.select_shape(idx, render=False)
            if self.shapes[idx].kind == "box":
                nx, ny = self._canvas_to_norm(event.x, event.y)
                self._edit = {"op": "move", "idx": idx, "grab": (nx, ny),
                              "box": shape_xyxy(self.shapes[idx]), "pushed": False}
        else:
            self.select_shape(-1, render=False)
            nx, ny = self._canvas_to_norm(event.x, event.y)
            self._edit = {"op": "draw", "anchor": (nx, ny), "pushed": False}
        self.request_render()

    def _on_b1_move(self, event) -> None:
        # 드래그 중에는 <Motion> 이 안 온다. 박스를 그리는 동안이야말로 무슨
        # 클래스인지 보여야 하므로 여기서도 태그를 따라 옮긴다.
        self._hover = (event.x, event.y)
        self._place_cursor_tag()
        if self._drag:
            self._on_drag_move(event)
            return
        if not self._edit or self._cur_img is None:
            return
        e = self._edit
        nx, ny = self._canvas_to_norm(event.x, event.y)

        if e["op"] == "erase":
            # 캔버스 아이템만 옮긴다. 언두는 실제로 지우는 시점(release)에 쌓는다 --
            # 드래그만 하다 말았을 때 언두 스택을 더럽히지 않기 위해서다.
            cx0, cy0 = e["c0"]
            self.canvas.coords(self._erase_id, cx0, cy0, event.x, event.y)
            return

        if not e["pushed"]:
            self._push_undo()
            e["pushed"] = True

        if e["op"] == "draw":
            ax, ay = e["anchor"]
            cx, cy, w, h = xyxy_to_cxcywh(ax, ay, nx, ny)
            self._draft = Shape(self.cur_cls, "box", box=(cx, cy, w, h),
                                src="A", tier="auto")
        elif e["op"] == "move":
            gx, gy = e["grab"]
            x0, y0, x1, y1 = e["box"]
            bw, bh = x1 - x0, y1 - y0
            # 이미지 밖으로 밀려나면 크기가 잘리므로 원점을 클램프한다.
            nx0 = min(max(0.0, x0 + nx - gx), max(0.0, 1.0 - bw))
            ny0 = min(max(0.0, y0 + ny - gy), max(0.0, 1.0 - bh))
            self._set_box(e["idx"], nx0, ny0, nx0 + bw, ny0 + bh)
        else:  # resize
            x0, y0, x1, y1 = e["box"]
            h = e["h"]
            if "w" in h:
                x0 = nx
            if "e" in h:
                x1 = nx
            if "n" in h:
                y0 = ny
            if "s" in h:
                y1 = ny
            self._set_box(e["idx"], x0, y0, x1, y1)
        self.request_render()

    def _on_b1_release(self, event) -> None:
        self._drag = None
        e, self._edit = self._edit, None
        draft, self._draft = self._draft, None
        if not e:
            return
        if e["op"] == "erase":
            self.canvas.itemconfigure(self._erase_id, state="hidden")
            ax, ay = e["anchor"]
            nx, ny = self._canvas_to_norm(event.x, event.y)
            self._push_undo()
            gone = self.erase_in_rect(ax, ay, nx, ny)
            if gone:
                self.status.config(
                    text=f"🩹 라벨 {gone}개를 지웠습니다 (되돌리기 Ctrl+Z)")
            else:
                self._pop_undo_silently()   # 아무것도 안 지웠으면 언두도 남기지 않는다
                self.status.config(text="🩹 감싼 영역에 라벨이 없습니다.")
            self.request_render()
            return
        if e["op"] == "draw":
            if draft is None:
                return
            H, W = self._cur_img.shape[:2]
            _, _, w, h = draft.box
            if w * W < self.MIN_BOX_PX or h * H < self.MIN_BOX_PX:
                self._pop_undo_silently()  # 오클릭 -> 언두 스택도 더럽히지 않는다
                self.request_render()
                return
            self.shapes.append(draft)
            # 일부러 선택하지 않는다. 그린 박스를 선택 상태로 두면 다음 박스용으로
            # 클래스 키를 눌렀을 때 방금 그린 박스가 바뀌어버린다.
            # 작업 흐름은 "클래스를 정하고 → 그 클래스로 계속 그린다" 이다.
            self.select_shape(-1, render=False)
            self._mark_dirty()
        elif e["pushed"]:
            self._mark_dirty()
        self._fill_inspector([])
        self.request_render()

    def _on_double_click(self, event) -> None:
        """예측 고스트 더블클릭 = 그 박스를 라벨로 승격."""
        if not self.opts.edit_mode or not self._can_edit():
            return
        gi = self._hit_ghost(event.x, event.y)
        if gi < 0:
            return
        self._promote_ghost(self.ghosts[gi])
        self._fill_inspector([])
        self.request_render()

    def _promote_ghost(self, g: Shape) -> None:
        s = g.clone()
        s.tier = "auto"  # 사람이 확인했으므로 더 이상 검토 대상이 아니다
        self._push_undo()
        self.shapes.append(s)
        self.select_shape(-1, render=False)  # 그리기와 같은 이유로 선택하지 않는다
        self._promoted += 1
        self._mark_dirty()

    # ---- 편집 연산 ---------------------------------------------------------
    def _set_box(self, idx: int, x0: float, y0: float, x1: float, y1: float) -> None:
        sh = self.shapes[idx]
        if sh.kind != "box":
            return
        sh.box = xyxy_to_cxcywh(x0, y0, x1, y1)
        sh.tier = "auto"  # 손댔으면 검토 완료
        self._mark_dirty()

    def _push_undo(self) -> None:
        self._undo.append([s.clone() for s in self.shapes])
        if len(self._undo) > self.UNDO_DEPTH:
            self._undo.pop(0)

    def _pop_undo_silently(self) -> None:
        if self._undo:
            self.shapes = self._undo.pop()

    def undo(self) -> None:
        if not self._undo:
            self.status.config(text="되돌릴 편집이 없습니다.")
            return
        self.shapes = self._undo.pop()
        self.select_shape(min(self.sel, len(self.shapes) - 1), render=False)
        self._mark_dirty()
        self._fill_inspector([])
        self.request_render()

    def select_shape(self, idx: int, render: bool = True) -> None:
        self.sel = idx if 0 <= idx < len(self.shapes) else -1
        self.opts.highlight = self.sel
        cur = self.tv.selection()
        want = (str(self.sel),) if self.sel >= 0 and self.tv.exists(str(self.sel)) else ()
        if cur != want:  # 불필요한 selection 이벤트로 렌더를 되풀이하지 않도록
            if cur:
                self.tv.selection_remove(*cur)
            if want:
                self.tv.selection_set(*want)
        if render:
            self.request_render()

    def cycle_selection(self, step: int) -> None:
        if not self.shapes:
            return
        self.select_shape((self.sel + step) % len(self.shapes))

    def toggle_peek(self) -> None:
        """Tab — 라벨과 예측을 통째로 감췄다 되돌린다. 원본을 잠깐 보는 용도다.

        선 두께가 작은 객체를 통째로 덮어버려서 확대하지 않으면 안 보일 때가 있다.
        확대·축소보다 한 번 껐다 켜는 편이 빠르다.

        L(라벨)·G(예측)을 각각 끄는 것과 달리 **둘 다** 끄고, 끄기 전 상태를
        기억했다가 그대로 되돌린다 -- 원본을 보려는 것이지 표시 설정을 바꾸려는
        것이 아니기 때문이다.
        """
        if self._peek is None:
            self._peek = (self.v_shapes.get(), self.v_ghost.get())
            self.v_shapes.set(False)
            self.v_ghost.set(False)
            self.status.config(text="👁 원본만 보기 — Tab 을 다시 누르면 돌아옵니다")
        else:
            shapes, ghosts = self._peek
            self._peek = None
            self.v_shapes.set(shapes)
            self.v_ghost.set(ghosts)
            self.status.config(text="👁 라벨을 다시 표시합니다")
        self._sync_opts()

    def set_class(self, cid: int) -> None:
        """그릴 클래스를 정한다. 박스가 선택돼 있으면 그 박스도 함께 바꾼다.

        선택은 '사용자가 박스를 클릭해서' 생긴 것만이다 -- 새로 그리거나 승격시킨
        박스는 일부러 선택하지 않으므로, 다음 박스를 위해 클래스를 바꿔도 방금
        만든 박스가 오염되지 않는다.
        """
        if not self.project or cid >= self.project.nc:
            return
        self.cur_cls = cid
        self.erase_mode = False    # 숫자키를 누르면 지우개에서 빠져나온다
        self.cb_cls.current(cid)
        self._update_status()
        self._place_cursor_tag()   # 마우스를 안 움직여도 즉시 반영
        if self.opts.edit_mode and 0 <= self.sel < len(self.shapes) and self._can_edit():
            if self.shapes[self.sel].cls != cid:
                self._push_undo()
                self.shapes[self.sel].cls = cid
                self.shapes[self.sel].tier = "auto"
                self._mark_dirty()
                self._fill_inspector([])
        self.request_render()

    def delete_selected(self) -> None:
        if not self.opts.edit_mode or not self._can_edit():
            return
        if not (0 <= self.sel < len(self.shapes)):
            return
        self._push_undo()
        self.shapes.pop(self.sel)
        self.select_shape(-1, render=False)
        self._mark_dirty()
        self._fill_inspector([])
        self.request_render()

    def set_erase(self, on: bool = True) -> None:
        """지우개 도구 on/off. 클래스 선택과 같은 줄(~ 1 2 …)의 도구 하나다."""
        # 조용히 무시하면 '키가 안 먹는다'로만 보인다. 왜 안 되는지 말해준다.
        if not self.opts.edit_mode:
            self.status.config(text="🩹 지우개는 편집 모드에서만 씁니다 — [E] 로 켜세요")
            return
        if not self._can_edit():
            return
        self.erase_mode = bool(on)
        self.canvas.config(cursor="X_cursor" if self.erase_mode else "crosshair")
        if not self.erase_mode:
            self.canvas.itemconfigure(self._erase_id, state="hidden")
        self.status.config(
            text="🩹 지우개 — 드래그로 감싼 라벨을 지웁니다 (숫자키로 그리기 복귀)"
            if self.erase_mode else "✏ 그리기 모드로 돌아왔습니다")
        self._update_status()
        self._place_cursor_tag()

    def erase_in_rect(self, x0: float, y0: float, x1: float, y1: float) -> int:
        """드래그 사각형 안의 라벨을 지운다. 반환: 지운 개수.

        판정은 **박스 중심이 안에 들어왔는가**로 한다. '완전히 포함'을 요구하면
        크게 잘못 그려진 박스가 영역 밖으로 삐져나와 안 지워지는데, 지우고 싶은 것이
        보통 그런 박스다. 반대로 '조금이라도 겹치면'은 옆 박스까지 말려든다.
        """
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        keep, gone = [], 0
        for s in self.shapes:
            a, b, c, d = shape_xyxy(s)
            cx, cy = (a + c) / 2, (b + d) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                gone += 1
            else:
                keep.append(s)
        if gone:
            self.shapes = keep
            self.select_shape(-1, render=False)
            self._mark_dirty()
            self._fill_inspector([])
        return gone

    def clear_all(self) -> None:
        """이 이미지의 라벨을 전부 지우고 빈 화면에서 다시 시작한다.

        약한 모델이 내놓은 시드는 고치는 것보다 지우고 새로 그리는 편이 쌀 때가
        많다 -- 135장 실측에서 v1 시드가 박스당 5.1초로 빈 화면 3.4초보다 비쌌다.
        그때 박스를 하나씩 지우는 대신 한 번에 비운다.

        되돌리기(Ctrl+Z)로 복구된다. 다만 비운 채로 다음 장으로 넘어가면
        '봤는데 객체가 없었다'는 유효한 판정으로 저장되므로 상태줄에 크게 알린다.
        """
        if not self.opts.edit_mode or not self._can_edit():
            return
        if not self.shapes:
            self.status.config(text="지울 라벨이 없습니다.")
            return
        self._push_undo()
        n = len(self.shapes)
        self.shapes = []
        self.select_shape(-1, render=False)
        self._mark_dirty()
        self._fill_inspector([])
        self.request_render()
        self.status.config(
            text=f"🧹 라벨 {n}개를 지웠습니다 — 빈 화면에서 다시 그리세요 "
                 f"(되돌리기 Ctrl+Z · 이대로 넘어가면 '객체 없음'으로 저장됩니다)")

    def accept_ghosts(self) -> None:
        """남은 예측 고스트를 전부 라벨로 승격 (이미 있는 것과 겹치면 건너뜀)."""
        if not self.opts.edit_mode or not self._can_edit():
            return
        added = 0
        for g in self.ghosts:
            if any(shape_iou(g, s) >= 0.5 and g.cls == s.cls for s in self.shapes):
                continue
            self._promote_ghost(g)
            added += 1
        self.status.config(text=f"예측 {added}개를 라벨로 승격했습니다.")
        self._fill_inspector([])
        self.request_render()

    # ---- 저장 / 진행 -------------------------------------------------------
    def save_current(self, force: bool = False) -> None:
        if not self.ws or not self._cur_fn or not self.split:
            if force:
                self._can_edit()
            return
        if not self.dirty and not force:
            return
        elapsed = min(time.perf_counter() - self._t_open, self.MAX_IMAGE_SECONDS)
        # 자동 저장은 이동 도중(show_current 안)에 불린다. 그때 self.index 는 이미
        # 다음 장이므로, 방금 확정한 파일 이름을 붙들어 두었다가 표시 갱신에 쓴다.
        saved_fn = self._cur_fn
        try:
            ops = self.ws.commit(self.split.name, saved_fn, self.shapes,
                                 self.baseline, elapsed, promoted=self._promoted)
        except OSError as e:
            messagebox.showerror("저장 실패", f"{e}")
            return
        self.dirty = False
        self._t_open = time.perf_counter()
        msg = (f"💾 {saved_fn} 저장 · 그리기 {ops.drawn} 삭제 {ops.deleted} "
               f"조정 {ops.adjusted} 유지 {ops.kept} · "
               f"비용 {ops.cost(self.ws.cost):.0f}s")
        # 연속형: 학습을 걸 때가 됐으면 요청만 떨어뜨린다(파일 하나 쓰기).
        # 실제 학습은 train_worker.py 가 별도 프로세스로 하므로 화면은 안 멈춘다.
        try:
            if self.ws.live_tick():
                msg += "  ·  🧠 학습 요청"
        except OSError:
            pass
        self.status.config(text=msg)
        self._update_session_label()
        self._refresh_row_marks(saved_fn)

    def next_pending_image(self) -> None:
        """저장하고 아직 손대지 않은 다음 이미지로. 라운드 진행의 기본 동작."""
        if not self.ws or not self.split or not self.ws.worklist.get(self.split.name):
            self.save_current(force=bool(self.ws))
            self.step(1)
            return
        self.save_current(force=True)
        nxt = self.ws.next_pending(self.split.name, self._cur_fn)
        if nxt is None:
            messagebox.showinfo("작업공간", "이 스플릿의 작업 목록을 모두 마쳤습니다. "
                                            "[📊 보고서] 에서 결과를 확인하세요.")
            return
        try:
            self.index = self.split.image_files.index(nxt)
        except ValueError:
            return
        self.show_current(sync_list=True)

    # -------------------------------------------------------------- 인스펙터
    def _fill_inspector(self, warns: Sequence[str]) -> None:
        self.tv.delete(*self.tv.get_children())
        for i, sh in enumerate(self.shapes):
            name = self.project.class_name(sh.cls) if self.project else str(sh.cls)
            if sh.kind == "box":
                cx, cy, w, h = sh.box
                geo = f"box  c({cx:.3f},{cy:.3f})  {w:.3f}×{h:.3f}"
            else:
                geo = f"poly  {sh.poly.shape[0]}pts  bbox{tuple(round(v, 3) for v in poly_bbox(sh.poly))}"
            tag = sh.src or "-"
            if sh.conf >= 0:
                tag += f" {sh.conf:.2f}"
            self.tv.insert("", tk.END, iid=str(i),
                           values=(f"{sh.cls}: {name}", geo, tag))
        self.select_shape(self.sel, render=False)

        if self.split and self.filtered:
            fn = self.split.image_files[self.index]
            lp = self.split.label_path_for(fn)
            msg = f"원본 라벨:\n{os.path.basename(lp) if lp else '❌ 없음 (고아 이미지)'}"
            if self.ws:
                lp2 = self.ws.label_path(self.split.name, fn)
                msg += f"\n작업본:\n{self.ws.name}/{os.path.relpath(lp2, self.ws.path)}"
            if self.ghosts:
                n_auto = sum(1 for g in self.ghosts if g.tier == "auto")
                n_rev = sum(1 for g in self.ghosts if g.tier == "review")
                n_rej = sum(1 for g in self.ghosts if g.tier == "reject")
                msg += f"\n예측: 자동승인 {n_auto} / 검토 {n_rev} / 기각 {n_rej}"
            if warns:
                msg += "\n\n⚠ 파싱 경고:\n" + "\n".join(warns[:5])
            self.lbl_meta.config(text=msg)

    def _on_tv_select(self, _=None) -> None:
        sel = self.tv.selection()
        self.sel = int(sel[0]) if sel else -1
        self.opts.highlight = self.sel
        self.request_render()

    def _on_class_toggle(self, _=None) -> None:
        sel = self.lb_cls.curselection()
        if not sel or not self.project:
            return
        cid = sel[0]
        if cid in self.opts.hidden_classes:
            self.opts.hidden_classes.discard(cid)
            b, g, r = self.project.color_of(cid)
            self.lb_cls.itemconfig(cid, foreground=f"#{r:02x}{g:02x}{b:02x}")
        else:
            self.opts.hidden_classes.add(cid)
            self.lb_cls.itemconfig(cid, foreground="#565660")
        self.request_render()

    # ----------------------------------------------------------------- 플래그
    def toggle_flag(self) -> None:
        if not self.split or not self.filtered:
            return
        key = (self.split.name, self.split.image_files[self.index])
        self.flags.symmetric_difference_update({key})
        self._update_status()

    def on_export_flags(self) -> None:
        if not self.flags:
            messagebox.showinfo("플래그", "플래그된 이미지가 없습니다. (Space 로 토글)")
            return
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         initialfile="flagged.txt",
                                         filetypes=[("Text", "*.txt")])
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            for sp, fn in sorted(self.flags):
                f.write(f"{sp}\t{fn}\n")
        messagebox.showinfo("플래그", f"{len(self.flags)}건 저장:\n{p}")

    # ----------------------------------------------------------------- 검수
    def on_validate(self) -> None:
        if not self.project:
            messagebox.showwarning("검수", "먼저 루트 폴더를 열어주세요.")
            return
        deep = messagebox.askyesno(
            "자동 검수",
            "이미지 파일 무결성(디코딩)까지 검사할까요?\n"
            "· 예 : 손상 파일까지 탐지 (느림)\n"
            "· 아니오 : 라벨/짝맞춤/좌표만 검사 (빠름, 권장)",
        )
        self.pb.pack(side=tk.BOTTOM, fill=tk.X)
        self.pb["value"] = 0
        self._validator = RuleValidator(self.project, deep_image_check=deep)
        prog_q: "queue.Queue[Tuple[int, int, str]]" = queue.Queue()
        result: dict = {}

        def worker():
            try:
                result["issues"] = self._validator.run(
                    lambda d, t, m: prog_q.put((d, t, m))
                )
            except Exception as e:
                traceback.print_exc()
                result["error"] = str(e)
            result["done"] = True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            last = None
            try:
                while True:
                    last = prog_q.get_nowait()
            except queue.Empty:
                pass
            if last:
                d, t, m = last
                self.pb["maximum"] = max(t, 1)
                self.pb["value"] = d
                self.status.config(text=f"검수 중… {d}/{t}  {m}")
            if result.get("done"):
                self.pb.pack_forget()
                if "error" in result:
                    messagebox.showerror("검수 실패", result["error"])
                    return
                self.issues = result["issues"]
                self.status.config(text=f"검수 완료 · 이슈 {len(self.issues)}건")
                self.show_report(self.issues, "룰 기반 자동 검수 결과")
            else:
                self.after(80, poll)

        self.after(80, poll)

    def on_ai_menu(self) -> None:
        """AI 확장 모듈 상태 표시 + 사용 가능한 것만 실행."""
        top = tk.Toplevel(self)
        top.title("AI 확장 모듈")
        top.configure(bg="#1e1e20")
        top.transient(self)
        self._center(top, 620, 400)

        ttk.Label(top, text="이상치 탐지 (AnomalyDetector)",
                  style="Head.TLabel").pack(anchor="w", padx=12, pady=(12, 2))
        for det in DETECTOR_REGISTRY:
            ok, why = det.available()
            row = ttk.Frame(top, padding=(12, 2))
            row.pack(fill=tk.X)
            ttk.Label(row, text=("🟢" if ok else "⚪") + f"  {det.name}").pack(anchor="w")
            ttk.Label(row, text=f"      {det.description}\n      상태: {why}",
                      style="Head.TLabel", justify="left").pack(anchor="w")

        ttk.Separator(top, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=8)
        ttk.Label(top, text="자동 라벨링 (AutoLabeler)",
                  style="Head.TLabel").pack(anchor="w", padx=12)
        for al in AUTOLABELER_REGISTRY:
            ok, why = al.available()
            ttk.Label(top, text=("🟢" if ok else "⚪") + f"  {al.name}   —   {why}",
                      justify="left").pack(anchor="w", padx=12, pady=1)

        ttk.Label(
            top,
            text="\n구현 방법은 소스의 MODULE 7 (ai_hooks) 클래스 docstring 참고.\n"
                 "analyze_batch() / predict() 만 채우면 이 UI 에 자동 연결됩니다.",
            style="Head.TLabel", justify="left",
        ).pack(anchor="w", padx=12, pady=(6, 0))

    def show_report(self, issues: List[Issue], title: str) -> None:
        top = tk.Toplevel(self)
        top.title(title)
        self._center(top, 1120, 620)
        top.configure(bg="#1e1e20")

        head = ttk.Frame(top, padding=8)
        head.pack(fill=tk.X)
        counts = {s: sum(1 for i in issues if i.severity == s) for s in SEVERITIES}
        ttk.Label(
            head,
            text=f"총 {len(issues)}건    ❌ ERROR {counts['ERROR']}    "
                 f"⚠ WARN {counts['WARN']}    ℹ INFO {counts['INFO']}",
        ).pack(side=tk.LEFT)

        f_err = tk.BooleanVar(value=True)
        f_warn = tk.BooleanVar(value=True)
        f_info = tk.BooleanVar(value=False)

        tvf = ttk.Frame(top)
        tvf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        cols = ("sev", "code", "split", "file", "detail")
        tree = ttk.Treeview(tvf, columns=cols, show="headings")
        for c, txt, w in (("sev", "등급", 62), ("code", "코드", 130),
                          ("split", "스플릿", 90), ("file", "파일", 300),
                          ("detail", "상세", 520)):
            tree.heading(c, text=txt)
            tree.column(c, width=w, anchor="w")
        vs = ttk.Scrollbar(tvf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree.tag_configure("ERROR", foreground="#ff7b72")
        tree.tag_configure("WARN", foreground="#ffcc66")
        tree.tag_configure("INFO", foreground="#8ab4f8")

        def refill():
            tree.delete(*tree.get_children())
            keep = {"ERROR": f_err.get(), "WARN": f_warn.get(), "INFO": f_info.get()}
            for n, it in enumerate(issues):
                if not keep.get(it.severity, True):
                    continue
                tree.insert("", tk.END, iid=str(n), tags=(it.severity,),
                            values=(it.severity, it.code, it.split, it.file, it.detail))

        for txt, var in (("ERROR", f_err), ("WARN", f_warn), ("INFO", f_info)):
            ttk.Checkbutton(head, text=txt, variable=var,
                            command=refill).pack(side=tk.LEFT, padx=6)

        def jump(_=None):
            sel = tree.selection()
            if not sel or not self.project:
                return
            it = issues[int(sel[0])]
            for si, sp in enumerate(self.project.splits):
                if sp.name == it.split:
                    if self.cb_split.current() != si:
                        self.cb_split.current(si)
                        self.on_split_change()
                    if it.index >= 0:
                        self.index = it.index
                        self.e_filter.delete(0, tk.END)
                        self._apply_filter()
                        self.index = it.index
                        self.show_current(sync_list=True)
                    break

        tree.bind("<Double-Button-1>", jump)

        def export():
            p = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile="validation_report.csv",
                filetypes=[("CSV", "*.csv")])
            if not p:
                return
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["severity", "code", "split", "file", "detail"])
                for it in issues:
                    w.writerow([it.severity, it.code, it.split, it.file, it.detail])
            messagebox.showinfo("리포트", f"저장 완료:\n{p}", parent=top)

        ttk.Button(head, text="CSV 내보내기", command=export).pack(side=tk.RIGHT)
        ttk.Label(head, text="(행 더블클릭 = 해당 이미지로 점프)  ",
                  style="Head.TLabel").pack(side=tk.RIGHT)
        refill()

    # ==========================================================================
    #  시작 화면 — 무엇을 할지 먼저 고른다
    # ==========================================================================
    def _open_dataset(self, path: str) -> bool:
        """데이터셋을 연다. 스플릿이 없으면 먼저 나눈다.

        이 한 단계 덕분에 이후 흐름이 갈라지지 않는다. '스플릿 있는 경우'와
        '이미지만 있는 경우'를 따로 다루면 작업목록·내보내기·채점이 전부
        두 벌이 된다. 들어오는 문에서 모양을 맞춰 보내는 편이 낫다.
        """
        self.load_project(path)
        if not self.project:
            return False
        if not is_flat_dataset(self.project):
            return True

        n_img = sum(len(s) for s in self.project.splits)
        if not messagebox.askyesno(
            "스플릿 없음",
            f"{os.path.basename(path)} 에는 train/valid/test 가 없습니다.\n"
            f"이미지 {n_img}장.\n\n"
            f"작업을 시작하기 전에 나눠야 이후 과정(작업목록·채점·내보내기)이\n"
            f"기존 데이터셋과 똑같이 흘러갑니다.\n\n지금 나눌까요?"):
            return False
        return self._do_split_dataset(path)

    def _do_split_dataset(self, path: str) -> bool:
        """비율·시드를 받아 새 데이터셋으로 나눈다. 원본은 그대로 둔다."""
        lay = self._require_layout()
        if not lay:
            return False

        top = tk.Toplevel(self)
        top.title("train / valid / test 로 나누기")
        top.configure(bg="#1e1e20")
        top.transient(self)
        top.grab_set()
        self._center(top)
        frm = ttk.Frame(top, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        base = os.path.basename(os.path.normpath(path))
        v_name = tk.StringVar(value=f"{base}_split")
        v_tr, v_va, v_te = (tk.StringVar(value="0.8"), tk.StringVar(value="0.1"),
                            tk.StringVar(value="0.1"))
        v_seed = tk.StringVar(value="0")
        state = {"ok": False, "root": None}

        ttk.Label(frm, text=f"원본: {path}", style="Head.TLabel",
                  wraplength=460).grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(frm, text="새 이름").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(frm, textvariable=v_name, width=28).grid(
            row=1, column=1, columnspan=5, sticky="we", padx=(8, 0))
        ttk.Label(frm, text="비율").grid(row=2, column=0, sticky="w")
        for i, (lb, v) in enumerate((("train", v_tr), ("valid", v_va),
                                     ("test", v_te))):
            ttk.Label(frm, text=lb).grid(row=2, column=1 + i * 2, sticky="e")
            ttk.Entry(frm, textvariable=v, width=6).grid(row=2, column=2 + i * 2,
                                                         padx=(2, 8))
        ttk.Label(frm, text="시드").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Entry(frm, textvariable=v_seed, width=6).grid(row=3, column=1,
                                                          sticky="w", padx=(8, 0))
        ttk.Label(frm, text="같은 시드면 같은 분할이 재현됩니다. 원본은 복사만 하고 "
                           "건드리지 않습니다.", style="Head.TLabel",
                  wraplength=460, justify="left").grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(4, 0))

        def go():
            try:
                ratios = (float(v_tr.get()), float(v_va.get()), float(v_te.get()))
                sd = int(v_seed.get())
            except ValueError:
                messagebox.showerror("나누기", "비율·시드를 확인하세요.", parent=top)
                return
            nm = safe_name(v_name.get().strip() or f"{base}_split")
            dst = os.path.join(lay.db_dir, nm)
            if os.path.isdir(dst) and os.listdir(dst):
                if not messagebox.askyesno(
                    "나누기", f"db/{nm} 이(가) 이미 있습니다. 덮어쓸까요?",
                    parent=top):
                    return
            top.destroy()
            self.pb.pack(side=tk.BOTTOM, fill=tk.X)
            self.pb["value"] = 0
            q: "queue.Queue[Tuple[int, int, str]]" = queue.Queue()
            res: dict = {}

            def worker():
                try:
                    res["mf"] = split_flat_dataset(
                        path, dst, ratios, sd, lambda d, t, m: q.put((d, t, m)))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    res["error"] = f"{type(e).__name__}: {e}"
                res["done"] = True

            threading.Thread(target=worker, daemon=True).start()

            def poll():
                last = None
                try:
                    while True:
                        last = q.get_nowait()
                except queue.Empty:
                    pass
                if last:
                    d, t, m = last
                    self.pb["maximum"] = max(t, 1)
                    self.pb["value"] = d
                    self.status.config(text=f"나누는 중… {d}/{t}  {m}")
                if not res.get("done"):
                    self.after(80, poll)
                    return
                self.pb.pack_forget()
                if "error" in res:
                    messagebox.showerror("나누기 실패", res["error"])
                    return
                c = res["mf"]["counts"]
                state["ok"] = True
                state["root"] = dst
                self.load_project(dst)
                self.status.config(
                    text=f"나누기 완료 · train {c['train']} / valid {c['valid']} "
                         f"/ test {c['test']} → db/{nm}")
                messagebox.showinfo(
                    "나누기 완료",
                    f"train {c['train']} · valid {c['valid']} · test {c['test']}\n"
                    f"db/{nm}\n\n이제 기존 데이터셋과 같은 흐름으로 작업합니다.")

            self.after(80, poll)

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=6, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="취소", command=top.destroy).pack(side=tk.RIGHT,
                                                              padx=4)
        ttk.Button(btns, text="나누기", command=go).pack(side=tk.RIGHT)
        self.wait_window(top)
        return bool(state["ok"])

    def _maybe_start_menu(self) -> None:
        """실행 직후 한 번만. 이미 작업 중이면 방해하지 않는다."""
        if self._start_menu_shown or self.ws is not None:
            return
        self._start_menu_shown = True
        try:
            self.on_start_menu()
        except Exception:  # noqa: BLE001  (시작 메뉴 때문에 앱이 죽으면 안 된다)
            traceback.print_exc()

    def on_start_menu(self) -> None:
        """할 일을 골라 그에 맞는 준비를 한 번에 끝낸다.

        기능이 늘면서 '무엇부터 눌러야 하는지'가 불분명해졌다. 작업공간을 만들고
        예측 폴더를 잡고 시드 모드를 고르는 일을 매번 손으로 하는 대신, 하려는
        일을 고르면 나머지는 알아서 맞춘다.
        """
        lay = self.layout or Layout.discover(
            os.path.dirname(os.path.abspath(__file__)))
        datasets = lay.datasets() if lay else []
        if not datasets:
            messagebox.showwarning(
                "시작", "db/ 에서 데이터셋을 찾지 못했습니다.\n"
                        "[📚 DB 열기] 로 작업 루트를 먼저 지정해주세요.")
            return
        self.layout = lay
        names = [os.path.basename(p) for p in datasets]

        top = tk.Toplevel(self)
        top.title("무엇을 할까요")
        top.configure(bg="#1e1e20")
        top.transient(self)
        top.grab_set()
        self._center(top)
        wrap = ttk.Frame(top, padding=16)
        wrap.pack(fill=tk.BOTH, expand=True)

        def card(title, desc):
            f = ttk.Frame(wrap, padding=(14, 12))
            f.pack(fill=tk.X, pady=6)
            ttk.Label(f, text=title, style="Card.TLabel").pack(anchor="w")
            ttk.Label(f, text=desc, style="Head.TLabel", justify="left",
                      wraplength=560).pack(anchor="w", pady=(2, 8))
            return f

        # ---------- ① 연속 라벨링 -------------------------------------------
        #  기본 데이터셋은 목록의 첫 번째(=이름순 첫 항목)가 아니라 가장 많이
        #  정리된 파생본이어야 한다. 원본에는 반사 패딩과 스플릿 간 중복이 남아
        #  있어서, 모르고 원본을 고르면 그걸 그대로 라벨하게 된다.
        best = next((n for n in reversed(names) if not n.endswith("_seg")),
                    names[0])

        c0 = card("①  연속 라벨링  (사람 + GPU 동시)",
                  "라벨링하는 동안 뒤에서 학습이 돌고, 새 모델이 나오면 아직 안 연 "
                  "이미지에 예측을 깔아줍니다. 처음 몇 장은 빈 화면이고, 그 뒤로는 "
                  "고치기만 하면 됩니다. 학습 워커는 자동으로 켜집니다.")
        r0 = ttk.Frame(c0)
        r0.pack(fill=tk.X)
        v0_ds = tk.StringVar(value=best)
        ttk.Combobox(r0, textvariable=v0_ds, values=names, state="readonly",
                     width=22).pack(side=tk.LEFT)
        v0_split = tk.StringVar(value="train")
        ttk.Label(r0, text="  대상").pack(side=tk.LEFT)
        ttk.Combobox(r0, textvariable=v0_split, state="readonly", width=8,
                     values=["train", "valid", "test"]).pack(side=tk.LEFT, padx=2)
        ttk.Label(r0, text="  N장마다 학습").pack(side=tk.LEFT)
        v0_every = tk.StringVar(value="50")
        ttk.Entry(r0, textvariable=v0_every, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(r0, text="  이름").pack(side=tk.LEFT)
        v0_name = tk.StringVar(value="live_" + time.strftime("%m%d"))
        ttk.Entry(r0, textvariable=v0_name, width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(r0, text="시작",
                   command=lambda: start_live()).pack(side=tk.RIGHT)

        # ---------- ② 수동 bbox 제작 ----------------------------------------
        c1 = card("②  수동 bbox 제작",
                  "train 에서 뽑아 빈 화면에서 직접 그립니다. 자동화 없이 순수 사람 "
                  "비용을 재는 기준선이자, 가장 깨끗한 라벨이 됩니다.")
        r1 = ttk.Frame(c1)
        r1.pack(fill=tk.X)
        v1_ds = tk.StringVar(value=names[0])
        ttk.Combobox(r1, textvariable=v1_ds, values=names, state="readonly",
                     width=22).pack(side=tk.LEFT)
        v1_split = tk.StringVar(value="train")
        ttk.Label(r1, text="  대상").pack(side=tk.LEFT)
        ttk.Combobox(r1, textvariable=v1_split, state="readonly", width=8,
                     values=["train", "valid", "test"]).pack(side=tk.LEFT, padx=2)
        v1_n = tk.StringVar(value="50")
        ttk.Entry(r1, textvariable=v1_n, width=6).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(r1, text="장   시드").pack(side=tk.LEFT)
        v1_seed = tk.StringVar(value="0")
        ttk.Entry(r1, textvariable=v1_seed, width=5).pack(side=tk.LEFT, padx=2)
        v1_name = tk.StringVar(value="manual_bbox")
        ttk.Label(r1, text="  이름").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=v1_name, width=14).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="시작",
                   command=lambda: start_manual()).pack(side=tk.RIGHT)

        # ---------- ③ seg 업그레이드 ----------------------------------------
        c2 = card("③  seg 업그레이드",
                  "이미 만든 bbox 를 SAM 에 먹여 마스크로 올립니다. 사람 노동이 "
                  "추가로 들지 않고, 결과는 겹쳐 보며 검수할 수 있습니다.")
        r2 = ttk.Frame(c2)
        r2.pack(fill=tk.X)
        v2_ds = tk.StringVar(value=names[0])
        ttk.Combobox(r2, textvariable=v2_ds, values=names, state="readonly",
                     width=22).pack(side=tk.LEFT)
        ws_names = [w.name for w in (lay.workspaces() if lay else [])]
        v2_src = tk.StringVar(value="db 라벨")
        ttk.Label(r2, text="  박스 출처").pack(side=tk.LEFT)
        ttk.Combobox(r2, textvariable=v2_src,
                     values=["db 라벨"] + ws_names, state="readonly",
                     width=18).pack(side=tk.LEFT, padx=2)
        v2_sam = tk.StringVar(value="sam2.1_b.pt")
        ttk.Label(r2, text="  SAM").pack(side=tk.LEFT)
        ttk.Combobox(r2, textvariable=v2_sam, state="readonly", width=14,
                     values=["mobile_sam.pt", "sam2.1_b.pt", "sam2.1_l.pt"]
                     ).pack(side=tk.LEFT, padx=2)
        ttk.Button(r2, text="시작",
                   command=lambda: start_seg()).pack(side=tk.RIGHT)

        # ---------- ④ 겹쳐 보기 · 검증 --------------------------------------
        c3 = card("④  겹쳐 보기 · 검증",
                  "라벨 파일을 원하는 만큼 얹어 한 화면에서 비교합니다. "
                  "박스와 마스크, 서로 다른 모델의 결과를 겹쳐 볼 수 있습니다.")
        r3 = ttk.Frame(c3)
        r3.pack(fill=tk.X)
        v3_ds = tk.StringVar(value=names[0])
        ttk.Combobox(r3, textvariable=v3_ds, values=names, state="readonly",
                     width=22).pack(side=tk.LEFT)
        ttk.Label(r3, text="  (연 뒤 우측 패널에서 + 로 레이어 추가)",
                  style="Head.TLabel").pack(side=tk.LEFT, padx=6)
        ttk.Button(r3, text="시작",
                   command=lambda: start_view()).pack(side=tk.RIGHT)

        ttk.Button(wrap, text="건너뛰기",
                   command=top.destroy).pack(side=tk.RIGHT, pady=(10, 0))

        # ---------- 동작 -----------------------------------------------------
        def ds_path(name: str) -> str:
            return datasets[names.index(name)]

        def find_split(want: str):
            for s in self.project.splits:
                if canon_split(s.name) == want:
                    return s
            messagebox.showwarning("시작", f"'{want}' 스플릿을 찾지 못했습니다.")
            return None

        def start_live():
            try:
                every = max(1, int(v0_every.get() or 50))
            except ValueError:
                messagebox.showerror("시작", "학습 주기를 확인하세요.", parent=top)
                return
            nm = v0_name.get().strip() or "live"
            ds_name, want = v0_ds.get(), v0_split.get()
            top.destroy()
            if not self._open_dataset(ds_path(ds_name)):
                return
            sp = find_split(want)
            if sp is None:
                return
            # 순서는 무작위 셔플. 액티브 러닝은 무작위를 이기지 못했으므로
            # 우선순위 큐를 둘 이유가 없다(P2b).
            files, steps = build_worklist(sp.image_files, mode="random", n=0, seed=0)
            ws = Workspace.create(
                self.layout, nm, self.project.root, kind="auto",
                note=f"연속 라벨링 — {every}장마다 학습 · 데이터셋 {ds_name}",
                # pred 로 두되 pred_dir 은 아직 없다. 첫 모델이 나오기 전까지는
                # 빈 화면이고, live_adopt 가 pred_dir 을 채우면 자동으로 시드된다.
                seed_mode="pred", pred_dir=None, hi=0.80, lo=0.25,
                worklist={sp.name: files}, classes=list(self.project.names),
                worklist_spec={"preset": "live", "split": sp.name,
                               "pick": "random", "n": 0, "seed": 0,
                               "steps": steps, "result": len(files),
                               "built": time.strftime("%Y-%m-%d %H:%M:%S")},
                live={"enabled": True, "every": every, "dataset": ds_name,
                      "split": sp.name, "base": "yolo11s.pt", "epochs": 80,
                      "seed": 0, "val_fast_n": 400, "val_period": 5,
                      "version": 1})
            self._open_workspace(ws)
            self._ensure_worker()   # 학습 워커는 스튜디오가 알아서 띄운다
            self.status.config(
                text=f"연속 라벨링 시작 · {sp.name} {len(files)}장 · "
                     f"{every}장마다 학습 · 처음 {every}장은 빈 화면입니다")

        def start_manual():
            try:
                n = int(v1_n.get() or 0)
                sd = int(v1_seed.get() or 0)
            except ValueError:
                messagebox.showerror("시작", "장수·시드를 확인하세요.", parent=top)
                return
            nm = v1_name.get().strip() or "manual_bbox"
            want = v1_split.get()
            top.destroy()
            if not self._open_dataset(ds_path(v1_ds.get())):
                return
            sp = None
            for s in self.project.splits:
                if canon_split(s.name) == want:
                    sp = s
                    break
            if sp is None:
                messagebox.showwarning(
                    "시작", f"'{want}' 스플릿을 찾지 못했습니다.")
                return
            files, steps = build_worklist(sp.image_files, mode="random",
                                          n=n, seed=sd)
            if not files:
                messagebox.showwarning("시작", "작업 목록이 비었습니다.")
                return
            ws = Workspace.create(
                self.layout, nm, self.project.root, kind="manual",
                note="빈 화면에서 직접 그린 bbox — 사람 비용 기준선",
                seed_mode="empty", worklist={sp.name: files},
                classes=list(self.project.names),
                worklist_spec={"preset": "manual_bbox", "split": sp.name,
                               "pick": "random", "n": n, "seed": sd,
                               "steps": steps, "result": len(files),
                               "built": time.strftime("%Y-%m-%d %H:%M:%S")})
            self._open_workspace(ws)
            self.status.config(
                text=f"수동 bbox 제작 시작 · {sp.name} {len(files)}장 · "
                     f"클래스를 정하고(숫자키) 빈 곳을 드래그해 그리세요")

        def start_seg():
            ds = v2_ds.get()
            src = v2_src.get()
            sam = v2_sam.get()
            top.destroy()
            self._run_seg_upgrade(ds_path(ds), ds, src, sam)

        def start_view():
            top.destroy()
            if not self._open_dataset(ds_path(v3_ds.get())):
                return
            self.v_edit.set(False)
            self.toggle_edit()
            self.status.config(
                text="겹쳐 보기 · 우측 패널 [+] 로 라벨 폴더를 얹으세요 "
                     "(더블클릭 = on/off)")

    def _run_seg_upgrade(self, ds_path: str, ds_name: str,
                         box_src: str, sam: str) -> None:
        """SAM 으로 마스크를 만들고, 끝나면 그걸 겹쳐 볼 수 있게 연다."""
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "make_masks.py")
        if not os.path.isfile(script):
            messagebox.showerror("seg 업그레이드", f"make_masks.py 가 없습니다:\n{script}")
            return
        out_name = f"sam_{ds_name}" + ("" if box_src == "db 라벨"
                                       else f"_{safe_name(box_src)}")
        cmd = [sys.executable, script, ds_name, "--sam", sam, "--out", out_name]
        if box_src != "db 라벨":
            cmd += ["--from-workspace", box_src]

        self.pb.pack(side=tk.BOTTOM, fill=tk.X)
        self.pb.config(mode="indeterminate")
        self.pb.start(12)
        self.status.config(text=f"SAM 마스크 생성 중… ({sam}) — 수 분 걸립니다")
        result: dict = {}

        def worker():
            try:
                import subprocess
                env = dict(os.environ, YOLO_AUTOINSTALL="false")
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", env=env)
                result["rc"] = p.returncode
                result["out"] = (p.stdout or "")[-2000:]
                result["err"] = (p.stderr or "")[-1500:]
            except Exception as e:  # noqa: BLE001
                result["rc"] = -1
                result["err"] = f"{type(e).__name__}: {e}"
            result["done"] = True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if not result.get("done"):
                self.after(200, poll)
                return
            self.pb.stop()
            self.pb.pack_forget()
            self.pb.config(mode="determinate")
            if result.get("rc") != 0:
                messagebox.showerror(
                    "seg 업그레이드 실패",
                    (result.get("err") or result.get("out") or "").strip()[-900:])
                self.status.config(text="SAM 마스크 생성 실패")
                return
            preds = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", out_name, "preds")
            self.load_project(ds_path)
            self.ref_layers.append({
                "name": out_name, "path": preds, "per_split": True,
                "visible": True, "filled": True, "dashed": False,
                "label": False, "src": "D", "color": None, "shapes": [],
            })
            if self.filtered:
                self._load_ref_layers(self.split.image_files[self.index])
            self._refresh_layer_list()
            self.request_render()
            tail = [ln for ln in (result.get("out") or "").splitlines()
                    if "마스크" in ln or "이미지" in ln]
            messagebox.showinfo(
                "seg 업그레이드 완료",
                "\n".join(tail[-3:]) + f"\n\nmodels/{out_name}/preds/\n\n"
                "박스 위에 마스크가 겹쳐 표시됩니다.")
            self.status.config(text=f"seg 업그레이드 완료 · 레이어 '{out_name}' 추가됨")

        self.after(200, poll)

    # ==========================================================================
    #  작업공간 / 보고서 / 내보내기 (MODULE 8 과 짝)
    # ==========================================================================
    def _close_workspace(self) -> None:
        self.save_current()
        self.ws = None
        self._worklist_only = False
        if hasattr(self, "v_wlonly"):
            self.v_wlonly.set(False)
            self.v_edit.set(False)
            self.toggle_edit()
        self._update_session_label()

    def _update_session_label(self) -> None:
        if not self.ws:
            self.lbl_session.config(
                text="작업공간 없음 · 편집 잠김\ndb/ 의 원본 라벨은 읽기 전용입니다.")
            return
        w = self.ws
        t = w.totals()
        prov = t["prov"]
        human = prov.get("A", 0) + prov.get("B", 0)
        auto = prov.get("C", 0) + prov.get("D", 0)
        saved = (1 - t["ratio"]) * 100 if t["manual_sec"] else 0.0
        self.lbl_session.config(
            text=(f"🗂 {w.name} · {WORKSPACE_KINDS.get(w.kind, w.kind)}\n"
                  f"라벨 {t['images']}장 / 박스 {t['boxes']}개"
                  + (f"   작업목록 {t['done']}/{t['total']}" if t["total"] else "") + "\n"
                  f"편집비용 {fmt_hms(t['cost_sec'])} vs 수동 {fmt_hms(t['manual_sec'])}"
                  f"  →  {saved:.0f}% 절감 (목표 65%)\n"
                  f"무편집 통과 {t['auto_rate'] * 100:.0f}%   "
                  f"출처 사람 A+B {human} · 기계 C+D {auto}"
                  + (f"   승격 {t['promoted']}" if t["promoted"] else "")
                  + self._live_line())
        )

    def _live_line(self) -> str:
        """연속형 상태 한 줄. 꺼져 있으면 빈 문자열이라 기존 화면과 같다.

        낡음(staleness)이 이 시스템의 진짜 비용이다 -- 예측은 176 img/s 라 큐가 빌
        일이 없고, 사람은 기다리지 않는다. 대신 지금 보는 예측을 만든 모델이
        몇 장 뒤처져 있는지가 품질을 좌우한다.
        """
        if not (self.ws and self.ws.live.get("enabled")):
            return ""
        s = self.ws.live_status()
        if s["worker"] is None:
            # 워커가 없으면 나머지 숫자는 의미가 없다. 그것부터 말한다.
            # 보통은 _ensure_worker 가 곧 띄우므로 잠깐 스쳐가는 상태다.
            if self._worker_tries < self.WORKER_MAX_TRIES:
                return "\n🤖 학습 워커 시작 중…"
            return ("\n⚠ 학습 워커를 띄우지 못했습니다 — work/worker.log 를 보세요"
                    + (f"  (요청 {s['pending']}건 대기 중)" if s["pending"] else ""))
        model = s["model"] or "아직 없음 (빈 화면으로 시작)"
        bits = [f"\n🤖 {model}"]
        if s["stale"] is not None:
            bits.append(f"낡음 {s['stale']}장")
        if s["pending"]:
            bits.append("학습 중")
        else:
            left = max(0, int(self.ws.live.get("last_request_n", 0))
                       + s["every"] - s["confirmed"])
            bits.append(f"다음 학습까지 {left}장")
        return " · ".join(bits)

    WORKER_RETRY_SEC = 60.0
    WORKER_MAX_TRIES = 3

    def _ensure_worker(self) -> bool:
        """학습 워커가 안 떠 있으면 띄운다. 반환: 지금 살아 있는가.

        사람이 bat 파일을 따로 열어야 한다는 것 자체가 사고 원인이다 -- 안 열면
        요청이 큐에 쌓이기만 하고 학습은 영원히 안 되는데 화면에는 아무 표시가 없다.

        반드시 **별도 프로세스**여야 한다(스레드면 ultralytics 가 GIL 을 잡아 UI 가
        얼어붙는다). 창은 띄우지 않고 출력은 work/worker.log 로 보낸다.
        """
        w = Workspace._worker()
        if not (w and self.layout):
            return False
        root = self.layout.root
        if w.worker_alive(root):
            self._worker_tries = 0
            return True
        # 계속 실패하는 환경(venv 깨짐 등)에서 무한히 다시 띄우지 않는다
        now = time.time()
        if (now - self._worker_spawn_t) < self.WORKER_RETRY_SEC:
            return False
        if self._worker_tries >= self.WORKER_MAX_TRIES:
            return False
        self._worker_spawn_t = now
        self._worker_tries += 1
        here = os.path.dirname(os.path.abspath(__file__))
        py = os.path.join(here, ".venv", "Scripts", "python.exe")
        if not os.path.isfile(py):
            py = sys.executable
        script = os.path.join(here, "train_worker.py")
        if not os.path.isfile(script):
            return False
        try:
            os.makedirs(os.path.join(root, "work"), exist_ok=True)
            log = open(os.path.join(root, "work", "worker.log"), "a",
                       encoding="utf-8", errors="replace")
            log.write("\n===== %s 스튜디오가 워커를 띄움 =====\n"
                      % time.strftime("%Y-%m-%d %H:%M:%S"))
            log.flush()
            self._worker_proc = subprocess.Popen(
                [py, script, "--root", root], cwd=here,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=dict(os.environ, YOLO_AUTOINSTALL="false",
                         PYTHONIOENCODING="utf-8"),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            return False
        return False  # 방금 띄웠다 — 심장박동은 다음 폴링에서 잡힌다

    def _live_poll(self) -> None:
        """워커가 새 모델을 냈는지 3초마다 확인하고, 있으면 그 예측으로 갈아탄다.

        이미 확정한 이미지는 건드리지 않는다. 예측은 아직 안 연 이미지에만 쓰인다.
        """
        try:
            if self.ws and self.ws.live.get("enabled"):
                self._ensure_worker()
                job = self.ws.live_adopt()
                if job:
                    self.pred_dir = self.ws.pred_dir or self.pred_dir
                    m = (job.get("metrics") or {}).get("mAP50")
                    self.status.config(
                        text=f"🤖 새 모델 {job.get('model')} 적용 · "
                             f"학습 {job.get('trained_images')}장 · mAP50 {m}")
                    self._update_session_label()
        except (OSError, ValueError, AttributeError):
            pass  # 워커 쪽 문제로 라벨링이 멈추면 안 된다
        finally:
            self.after(3000, self._live_poll)

    def _on_gtref_toggle(self) -> None:
        if self.v_gtref.get():
            messagebox.showwarning(
                "GT 참조",
                "db/ 의 정답 라벨을 화면에 띄웁니다.\n\n"
                "이 상태에서 만든 라벨은 사람이 정답을 보고 베낀 것이므로\n"
                "출처가 GT(태그 B)인 시뮬레이션으로만 유효합니다.\n"
                "실제 사람 편집 비용(B)을 측정하는 중이라면 꺼두세요.",
            )
        self.request_render()

    def _on_wlonly_toggle(self) -> None:
        if self.v_wlonly.get() and not self.ws:
            self.v_wlonly.set(False)
            self.status.config(text="작업 목록이 없습니다. 먼저 작업공간을 여세요.")
            return
        self._worklist_only = self.v_wlonly.get()
        self._apply_filter()

    def on_pick_pred_dir(self) -> None:
        d = filedialog.askdirectory(title="예측 라벨(.txt) 폴더 선택")
        if not d:
            return
        self.pred_dir = d
        if self.ws:
            self.ws.pred_dir = d
            self.ws.save()
        n = sum(1 for e in os.scandir(d) if e.is_file() and e.name.endswith(".txt"))
        self.status.config(text=f"예측 폴더 연결: {d}  (txt {n}개)")
        if self.split and self.filtered:
            self.ghosts = self._ghosts_for(self.split.image_files[self.index])
            self._fill_inspector([])
            self.request_render()

    # ---- DB ---------------------------------------------------------------
    def on_open_db(self) -> None:
        """db/ 아래 데이터셋을 골라 연다. 없으면 루트부터 잡는다."""
        lay = self.layout
        if lay is None:
            d = filedialog.askdirectory(
                title="작업 루트 선택 (db / work / export 를 담을 폴더)")
            if not d:
                return
            lay = self._layout_for(d)
            if lay is None:
                return
            self.layout = lay
        ds = lay.datasets()
        if not ds:
            messagebox.showinfo(
                "DB", f"{lay.db_dir} 가 비어 있습니다.\n"
                      "데이터셋 폴더를 그 안에 넣은 뒤 다시 열어주세요.")
            return
        if len(ds) == 1:
            self.load_project(ds[0])
            return

        top = tk.Toplevel(self)
        top.title("DB 데이터셋 선택")
        top.configure(bg="#1e1e20")
        top.transient(self)
        top.grab_set()
        self._center(top)
        lb = tk.Listbox(top, width=60, height=min(14, len(ds)), bg="#252528",
                        fg="#dcdcdc", highlightthickness=0, borderwidth=0)
        lb.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        for p in ds:
            lb.insert(tk.END, "  " + os.path.basename(p))
        lb.selection_set(0)

        def ok(_=None):
            sel = lb.curselection()
            if not sel:
                return
            p = ds[sel[0]]
            top.destroy()
            self.load_project(p)

        lb.bind("<Double-Button-1>", ok)
        ttk.Button(top, text="열기", command=ok).pack(pady=(0, 12))

    def _center(self, top: tk.Toplevel, w: int = 0, h: int = 0) -> None:
        """대화창을 본 창 기준 가운데에 띄운다.

        Tk 는 위치를 지정하지 않으면 화면 구석이나 직전 창 옆에 쌓아 올린다.
        작업 중인 창에서 시선이 멀어지므로 매번 찾아야 한다.
        세로는 정확한 중앙보다 살짝 위가 안정적으로 보인다(1/3 지점).
        """
        def place() -> None:
            if not top.winfo_exists():
                return
            top.update_idletasks()
            ww = w or top.winfo_reqwidth()
            hh = h or top.winfo_reqheight()

            px, py = self.winfo_rootx(), self.winfo_rooty()
            pw, ph = self.winfo_width(), self.winfo_height()
            if pw <= 1 or ph <= 1:      # 본 창이 아직 안 떴으면 화면 기준
                px = py = 0
                pw, ph = top.winfo_screenwidth(), top.winfo_screenheight()

            x = px + (pw - ww) // 2
            y = py + (ph - hh) // 3
            sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
            x = max(0, min(x, sw - ww))
            y = max(0, min(y, sh - hh))
            top.geometry(f"{ww}x{hh}+{x}+{y}")

        # 위젯을 다 만들기 전에 크기를 재면 빈 창 기준으로 잡혀 어긋난다.
        # 호출부가 내용을 채우고 이벤트 루프로 돌아온 뒤에 배치한다.
        top.after_idle(place)

    def _layout_for(self, d: str) -> Optional[Layout]:
        """고른 폴더로부터 작업 루트를 정한다. 없으면 만들지 물어본다.

        여기가 함정이 되기 쉽다. 버튼 이름이 'DB 열기'라 사용자가 자연스럽게
        db 폴더를 고르는데, 그걸 그대로 루트로 삼으면 db 안에 db/work/export 를
        또 만들어버린다. 실제로 그런 일이 있었다.
        """
        # 1) 이미 배치 안이면 그 루트를 쓴다 (db 폴더나 데이터셋을 골라도 됨)
        found = Layout.discover(d)
        if found:
            return found
        # 2) db/work/export 중 하나를 고른 경우 -> 부모가 루트다
        base = os.path.basename(os.path.normpath(d)).lower()
        if base in (DB_DIR, WORK_DIR, EXPORT_DIR):
            parent = os.path.dirname(os.path.normpath(d))
            if os.path.isdir(parent):
                return Layout(parent).ensure()
        # 3) 정말 새로 만드는 경우에만 확인을 받는다
        if not messagebox.askyesno(
            "작업 루트 만들기",
            f"{d}\n\n여기에 db / work / export 폴더를 새로 만듭니다.\n"
            f"데이터셋은 db/ 안에 넣어야 합니다.\n\n계속할까요?"):
            return None
        return Layout(d).ensure()

    def _require_layout(self) -> Optional[Layout]:
        if self.layout:
            return self.layout
        messagebox.showwarning(
            "작업 루트 없음",
            "db / work / export 구조를 찾지 못했습니다.\n"
            "[📚 DB 열기] 로 작업 루트를 먼저 지정해주세요.")
        return None

    # ---- 작업공간 만들기 ---------------------------------------------------
    def on_ws_new(self) -> None:
        if not self.project or not self.split:
            messagebox.showwarning("작업공간", "먼저 데이터셋을 열어주세요.")
            return
        lay = self._require_layout()
        if not lay:
            return
        existing = lay.workspaces()

        top = tk.Toplevel(self)
        top.title("작업공간 만들기")
        top.configure(bg="#1e1e20")
        top.transient(self)
        top.grab_set()
        self._center(top)
        frm = ttk.Frame(top, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)
        frm.columnconfigure(1, weight=1)

        v_name = tk.StringVar(value=f"{self.project.name}_{len(existing) + 1:02d}")
        v_kind = tk.StringVar(value="auto")
        v_note = tk.StringVar(value="")
        v_parent = tk.StringVar(value="")
        v_model = tk.StringVar(value="")
        v_mode = tk.StringVar(value="pred")
        v_pred = tk.StringVar(value=self.pred_dir or "")
        v_src = tk.StringVar(value=self.pred_src)
        v_hi = tk.StringVar(value=f"{self.hi:.2f}")
        v_lo = tk.StringVar(value=f"{self.lo:.2f}")
        v_n = tk.StringVar(value="150")
        v_seed = tk.StringVar(value="0")

        r = 0

        def row(label, widget, hint=""):
            nonlocal r
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="we", pady=3, padx=(8, 0))
            if hint:
                ttk.Label(frm, text=hint, style="Head.TLabel").grid(
                    row=r, column=2, sticky="w", padx=(8, 0))
            r += 1

        row("이름", ttk.Entry(frm, textvariable=v_name),
            f"work/ 아래 폴더명이 됩니다")
        row("종류", ttk.Combobox(frm, textvariable=v_kind, state="readonly", width=14,
                               values=list(WORKSPACE_KINDS)),
            " · ".join(f"{k}={v}" for k, v in WORKSPACE_KINDS.items()))
        row("메모", ttk.Entry(frm, textvariable=v_note), "보고서 머리에 표시됩니다")
        row("부모 작업공간",
            ttk.Combobox(frm, textvariable=v_parent, state="readonly", width=24,
                         values=[""] + [w.name for w in existing]),
            "이걸로 학습한 모델을 쓴다면 지정 — 계보가 보고서에 남습니다")
        row("모델 메모", ttk.Entry(frm, textvariable=v_model),
            "예: yolo11s, 150장 학습, epochs=80")

        ttk.Label(frm, text="시드 소스").grid(row=r, column=0, sticky="nw", pady=3)
        modes = ttk.Frame(frm)
        modes.grid(row=r, column=1, columnspan=2, sticky="w", pady=3, padx=(8, 0))
        for key, txt in SEED_MODES.items():
            ttk.Radiobutton(modes, text=txt, value=key, variable=v_mode).pack(anchor="w")
        r += 1

        predrow = ttk.Frame(frm)
        ttk.Entry(predrow, textvariable=v_pred).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(predrow, text="…", width=3,
                   command=lambda: v_pred.set(
                       filedialog.askdirectory(parent=top) or v_pred.get())
                   ).pack(side=tk.LEFT, padx=4)
        row("예측 폴더", predrow, "ultralytics save_txt save_conf 출력")
        row("예측 출처 태그",
            ttk.Combobox(frm, textvariable=v_src, width=8, state="readonly",
                         values=["C", "D"]),
            "C=학습된 모델 · D=zero-shot")
        row("자동승인 임계 hi", ttk.Entry(frm, textvariable=v_hi, width=10),
            "이 이상 = 자동승인 (Precision ≥0.95 목표)")
        row("검토 하한 lo", ttk.Entry(frm, textvariable=v_lo, width=10),
            "이 미만 = 기각(고스트로만 표시)")
        # ---- 작업목록 구획 ------------------------------------------------
        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=3, sticky="we", pady=(12, 8))
        r += 1
        ttk.Label(frm, text="작업목록", style="Head.TLabel").grid(
            row=r, column=0, sticky="w")
        r += 1

        v_pick = tk.StringVar(value="random")
        v_stride = tk.StringVar(value="7")
        v_onlypred = tk.BooleanVar(value=True)
        v_nooverlap = tk.BooleanVar(value=True)
        v_listfile = tk.StringVar(value="")

        pick = ttk.Frame(frm)
        pick.grid(row=r, column=0, columnspan=3, sticky="we", padx=(0, 0))
        r += 1
        ttk.Radiobutton(pick, text="전체", value="all", variable=v_pick
                        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(pick, text="무작위", value="random", variable=v_pick
                        ).grid(row=1, column=0, sticky="w")
        ttk.Entry(pick, textvariable=v_n, width=7).grid(row=1, column=1, padx=(6, 2))
        ttk.Label(pick, text="장   시드").grid(row=1, column=2)
        ttk.Entry(pick, textvariable=v_seed, width=6).grid(row=1, column=3, padx=2)
        ttk.Label(pick, text="  (0 = 전체 · 액티브 러닝은 기각됨)",
                  style="Head.TLabel").grid(row=1, column=4, sticky="w")
        ttk.Radiobutton(pick, text="등간격", value="stride", variable=v_pick
                        ).grid(row=2, column=0, sticky="w")
        ttk.Entry(pick, textvariable=v_stride, width=7).grid(row=2, column=1, padx=(6, 2))
        ttk.Label(pick, text="장마다").grid(row=2, column=2, sticky="w")
        ttk.Label(pick, text="  (봉인 평가셋 204장이 stride 7 로 만들어졌음)",
                  style="Head.TLabel").grid(row=2, column=4, sticky="w")

        ttk.Checkbutton(frm, text="예측이 있는 이미지만 (모델이 학습에 쓴 것은 자동 제외)",
                        variable=v_onlypred).grid(row=r, column=0, columnspan=3,
                                                  sticky="w", pady=1)
        r += 1
        ttk.Checkbutton(frm, text="다른 작업공간에 이미 있는 이미지 제외 (겹침 방지)",
                        variable=v_nooverlap).grid(row=r, column=0, columnspan=3,
                                                   sticky="w", pady=1)
        r += 1

        lrow = ttk.Frame(frm)
        ttk.Entry(lrow, textvariable=v_listfile).pack(side=tk.LEFT, fill=tk.X,
                                                      expand=True)
        ttk.Button(lrow, text="…", width=3,
                   command=lambda: v_listfile.set(
                       filedialog.askopenfilename(
                           parent=top, filetypes=[("텍스트", "*.txt"), ("전체", "*.*")])
                       or v_listfile.get())).pack(side=tk.LEFT, padx=4)
        ttk.Button(lrow, text="지우기", width=6,
                   command=lambda: v_listfile.set("")).pack(side=tk.LEFT)
        row("파일 목록", lrow, "한 줄에 파일명 하나 · 지정하면 위 설정을 무시합니다")

        lbl_preview = ttk.Label(frm, text="", style="Head.TLabel", justify="left",
                                wraplength=560)
        lbl_preview.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 0))
        r += 1

        def others_names():
            """다른 작업공간이 이 스플릿에서 이미 잡고 있는 파일명."""
            out = set()
            for w in existing:
                out |= set(w.worklist.get(self.split.name) or [])
                out |= {fn for (sp, fn) in w.records if sp == self.split.name}
            return out

        def compute():
            try:
                n_ = int(v_n.get() or 0)
                sd_ = int(v_seed.get() or 0)
                st_ = int(v_stride.get() or 1)
            except ValueError:
                return [], ["숫자 항목을 확인하세요."]
            pdir = v_pred.get().strip() or None
            return build_worklist(
                self.split.image_files, mode=v_pick.get(), n=n_, stride=st_, seed=sd_,
                pred_dir=pdir if (v_onlypred.get() and pdir) else None,
                exclude=others_names() if v_nooverlap.get() else (),
                from_list=v_listfile.get().strip() or None,
            )

        def refresh(*_a):
            files_, steps_ = compute()
            lbl_preview.config(
                text="  →  ".join(steps_) + f"\n최종 {len(files_)}장"
                + (f"   예: {files_[0][:44]}…" if files_ else "   ⚠ 비어 있음"))

        for v in (v_pick, v_n, v_seed, v_stride, v_onlypred, v_nooverlap,
                  v_listfile, v_pred):
            v.trace_add("write", refresh)
        refresh()

        def start():
            name = v_name.get().strip()
            if not name:
                messagebox.showerror("작업공간", "이름을 입력하세요.", parent=top)
                return
            try:
                hi, lo = float(v_hi.get()), float(v_lo.get())
                n, sd = int(v_n.get()), int(v_seed.get())
            except ValueError:
                messagebox.showerror("작업공간", "숫자 항목을 확인하세요.", parent=top)
                return
            if lo > hi:
                messagebox.showerror("작업공간", "lo 는 hi 보다 클 수 없습니다.", parent=top)
                return
            mode = v_mode.get()
            pdir = v_pred.get().strip() or None
            if mode == "pred" and not (pdir and os.path.isdir(pdir)):
                messagebox.showerror("작업공간", "예측 폴더를 지정하세요.", parent=top)
                return
            if mode == "gt" and not messagebox.askyesno(
                "시뮬레이션 확인",
                "db 의 GT 를 시드로 복사합니다. 이 작업공간의 라벨은 전부 태그 B 로\n"
                "기록되며 사람 주석자 시뮬레이션 용도로만 유효합니다. 계속할까요?",
                parent=top,
            ):
                return
            path = os.path.join(lay.work_dir, safe_name(name))
            if os.path.isdir(path) and not messagebox.askyesno(
                "작업공간", f"{safe_name(name)} 이(가) 이미 있습니다. 이어서 쓸까요?",
                parent=top):
                return

            files, steps = compute()
            if not files:
                messagebox.showerror(
                    "작업공간", "작업목록이 비었습니다.\n\n" + "\n".join(steps),
                    parent=top)
                return
            top.destroy()
            ws = Workspace.create(
                lay, name, self.project.root, kind=v_kind.get(),
                note=v_note.get().strip(), parent=v_parent.get() or None,
                model=({"note": v_model.get().strip()} if v_model.get().strip() else {}),
                seed_mode=mode, pred_dir=pdir, pred_src=v_src.get(),
                hi=hi, lo=lo, shuffle_seed=sd,
                worklist={self.split.name: files},
                classes=list(self.project.names),
                worklist_spec={
                    "split": self.split.name, "pick": v_pick.get(),
                    "n": n, "stride": int(v_stride.get() or 1), "seed": sd,
                    "only_with_pred": bool(v_onlypred.get()),
                    "exclude_other_workspaces": bool(v_nooverlap.get()),
                    "from_list": v_listfile.get().strip() or None,
                    "steps": steps, "result": len(files),
                    "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            self._open_workspace(ws)

        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=3, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="취소", command=top.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="만들기", command=start).pack(side=tk.RIGHT)

    def _open_workspace(self, ws: Workspace) -> None:
        self.save_current()
        self.ws = ws
        self.pred_dir = ws.pred_dir or self.pred_dir
        self.pred_src, self.hi, self.lo = ws.pred_src, ws.hi, ws.lo
        ws.save()
        self._cur_fn = None
        self._worklist_only = bool(ws.worklist)
        self.v_wlonly.set(self._worklist_only)
        self.v_edit.set(True)
        self.toggle_edit()

        # 작업목록이 있는 스플릿으로 옮긴다. 안 그러면 목록 없는 스플릿을 보고
        # 있다가 폴백(전체 표시)에 걸려, 목록 밖 이미지를 라벨하면서도 그 사실을
        # 모르게 된다. 실제로 그런 일이 있었다 -- 진행률만 0 으로 남았다.
        if ws.worklist and self.project:
            cur = self.split.name if self.split else None
            if cur not in ws.worklist:
                for i, sp in enumerate(self.project.splits):
                    if sp.name in ws.worklist and ws.worklist[sp.name]:
                        self.cb_split.current(i)
                        self.split = sp
                        self.index = 0
                        self._cur_fn = None
                        self.status.config(
                            text=f"작업목록이 '{sp.name}' 에 있어 스플릿을 옮겼습니다.")
                        break

        self._apply_filter()
        if self.split:
            first = ws.next_pending(self.split.name)
            if first and first in self.split.image_files:
                self.index = self.split.image_files.index(first)
                self.show_current(sync_list=True)
        self._update_session_label()
        done, total = ws.progress()
        self.status.config(
            text=f"작업공간 '{ws.name}' 열림 · {done}/{total} · "
                 f"저장 위치 work/{os.path.basename(ws.path)}/"
        )

    def on_ws_open(self) -> None:
        lay = self._require_layout()
        if not lay:
            return
        wss = lay.workspaces()
        if not wss:
            messagebox.showinfo("작업공간", "저장된 작업공간이 없습니다.")
            return

        top = tk.Toplevel(self)
        top.title("작업공간 열기")
        top.configure(bg="#1e1e20")
        top.transient(self)
        top.grab_set()
        self._center(top)
        lb = tk.Listbox(top, width=88, height=min(14, len(wss)), bg="#252528",
                        fg="#dcdcdc", highlightthickness=0, borderwidth=0)
        lb.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        for w in wss:
            t = w.totals()
            lb.insert(tk.END, f"  {w.name:<24} {WORKSPACE_KINDS.get(w.kind, w.kind):<16}"
                              f" 라벨 {t['images']:>5}장  "
                              f"{os.path.basename(w.source)}  {w.created}")
        lb.selection_set(0)

        def ok(_=None):
            sel = lb.curselection()
            if not sel:
                return
            w = wss[sel[0]]
            top.destroy()
            # 작업공간이 가리키는 데이터셋이 지금 열린 것과 다르면 그쪽으로 옮긴다.
            if not self.project or os.path.normpath(self.project.root) != \
                    os.path.normpath(w.source):
                if os.path.isdir(w.source):
                    self.load_project(w.source)
                else:
                    messagebox.showerror(
                        "작업공간", f"소스 데이터셋을 찾을 수 없습니다:\n{w.source}")
                    return
            self._open_workspace(w)

        lb.bind("<Double-Button-1>", ok)
        ttk.Button(top, text="열기", command=ok).pack(pady=(0, 12))

    # ---- 보고서 -----------------------------------------------------------
    def on_ws_report(self) -> None:
        if not self.ws:
            messagebox.showinfo("보고서", "열린 작업공간이 없습니다.")
            return
        self.save_current()
        w = self.ws
        siblings = self.layout.workspaces() if self.layout else []
        data = build_report(w, self.project, siblings)
        t = data["totals"]
        ops: EditOps = t["ops"]
        sc = data["score"]

        top = tk.Toplevel(self)
        top.title(f"{w.name} 보고서")
        self._center(top, 1060, 660)
        top.configure(bg="#1e1e20")

        head = ttk.Frame(top, padding=10)
        head.pack(fill=tk.X)
        prov_txt = "  ".join(f"{k}={t['prov'].get(k, 0)}" for k in PROV_TAGS)
        saved = (1 - t["ratio"]) * 100 if t["manual_sec"] else 0.0
        lines = [
            f"{w.name} · {data['kind_desc']} · 소스 {os.path.basename(w.source)}"
            f" · 생성 {w.created}" + (f" · {w.note}" if w.note else ""),
            f"라벨 {t['images']}장 · 박스 {t['boxes']}개 · "
            f"작업목록 {t['done']}/{t['total']} · 출처 {prov_txt}",
            f"편집: 그리기 {ops.drawn} · 삭제 {ops.deleted} · 조정 {ops.adjusted} · "
            f"무편집 {ops.kept}" + (f" · 승격 {t['promoted']}" if t["promoted"] else ""),
            f"비용 {fmt_hms(t['cost_sec'])} / 수동환산 {fmt_hms(t['manual_sec'])}"
            f"  →  {saved:.1f}% 절감 (목표 65%) · 무편집 통과 {t['auto_rate'] * 100:.1f}%",
            f"실측 소요 {fmt_hms(ops.seconds)} · 이미지당 중앙값 {t['median_sec']:.0f}초"
            f" · 기간 {t['first_ts'][:16]} → {t['last_ts'][:16]}",
        ]
        if sc:
            ta = sc["tier"]["auto"]
            lines.append(
                f"GT 채점: P {sc['precision'] * 100:.1f}% / R {sc['recall'] * 100:.1f}%"
                f" / F1 {sc['f1'] * 100:.1f}%  (TP {sc['tp']} FP {sc['fp']} FN {sc['fn']})"
                f" · 자동승인 티어 P {ta['precision'] * 100:.1f}%"
                f" {'✅' if ta['precision'] >= 0.95 else '❌'} 목표 95%")
        else:
            lines.append("GT 채점: 소스에 정답 라벨이 없어 생략 (0부터 구축하는 데이터셋)")
        if data["children"]:
            lines.append("이 작업공간 기반: " + ", ".join(
                f"{c['name']}({c['images']}장)" for c in data["children"]))
        ttk.Label(head, justify="left", text="\n".join(lines)).pack(side=tk.LEFT)

        cols = ("split", "file", "drawn", "deleted", "adjusted", "kept",
                "cost", "sec", "src")
        tvf = ttk.Frame(top)
        tvf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        tree = ttk.Treeview(tvf, columns=cols, show="headings")
        for c, txt, wd in (("split", "스플릿", 90), ("file", "파일", 260),
                           ("drawn", "그리기", 66), ("deleted", "삭제", 60),
                           ("adjusted", "조정", 60), ("kept", "무편집", 66),
                           ("cost", "비용(s)", 72), ("sec", "실측(s)", 72),
                           ("src", "출처", 140)):
            tree.heading(c, text=txt)
            tree.column(c, width=wd, anchor="w")
        vs = ttk.Scrollbar(tvf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rows = []
        for (sp, fn), rec in sorted(w.records.items()):
            o = EditOps.from_dict(rec.get("ops"))
            cnt: Dict[str, int] = {}
            for tag in rec.get("src", []):
                cnt[tag] = cnt.get(tag, 0) + 1
            row = (sp, fn, o.drawn, o.deleted, o.adjusted, o.kept,
                   f"{rec.get('cost_sec', 0):.0f}", f"{o.seconds:.0f}",
                   " ".join(f"{k}:{v}" for k, v in sorted(cnt.items())) or "-")
            rows.append(row)
            tree.insert("", tk.END, values=row)

        def export_csv():
            p = filedialog.asksaveasfilename(
                parent=top, defaultextension=".csv",
                initialfile=f"{safe_name(w.name)}_edits.csv",
                filetypes=[("CSV", "*.csv")])
            if not p:
                return
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                wr.writerow(cols)
                wr.writerows(rows)
                wr.writerow([])
                for k, v in (("cost_sec", t["cost_sec"]), ("manual_sec", t["manual_sec"]),
                             ("ratio", t["ratio"]), ("real_sec", ops.seconds),
                             ("auto_rate", t["auto_rate"])):
                    wr.writerow([k, f"{v:.4f}"])
                for k in PROV_TAGS:
                    wr.writerow([f"prov_{k}", t["prov"].get(k, 0)])
                if sc:
                    for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
                        wr.writerow([k, sc[k]])
            messagebox.showinfo("보고서", f"저장 완료:\n{p}", parent=top)

        def open_html():
            try:
                p = write_report(w, self.project, siblings)
            except OSError as e:
                messagebox.showerror("보고서", f"{e}", parent=top)
                return
            try:
                os.startfile(p)  # noqa: S606  (Windows 기본 브라우저)
            except (AttributeError, OSError):
                messagebox.showinfo("보고서", f"저장됨:\n{p}", parent=top)

        ttk.Button(head, text="CSV 내보내기", command=export_csv).pack(side=tk.RIGHT)
        ttk.Button(head, text="📄 HTML 보고서",
                   command=open_html).pack(side=tk.RIGHT, padx=6)

    # ---- 내보내기 ---------------------------------------------------------
    def on_export(self) -> None:
        if not self.project:
            messagebox.showwarning("내보내기", "먼저 데이터셋을 열어주세요.")
            return
        lay = self._require_layout()
        if not lay:
            return
        wss = lay.workspaces()
        mine = [w for w in wss
                if os.path.normpath(w.source) == os.path.normpath(self.project.root)]
        if not mine:
            messagebox.showinfo("내보내기", "이 데이터셋에 연결된 작업공간이 없습니다.")
            return

        top = tk.Toplevel(self)
        top.title("내보내기")
        top.configure(bg="#1e1e20")
        top.transient(self)
        top.grab_set()
        self._center(top)
        frm = ttk.Frame(top, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="라벨 출처 (여러 개 선택 가능 · 위에 있는 것이 우선)",
                  style="Head.TLabel").pack(anchor="w")
        ttk.Label(frm, text="같은 이미지를 여러 작업공간이 라벨링했다면 위쪽이 이깁니다.",
                  style="Head.TLabel").pack(anchor="w", pady=(0, 4))
        lbf = ttk.Frame(frm)
        lbf.pack(fill=tk.X)
        lb = tk.Listbox(lbf, height=min(8, max(3, len(mine))), selectmode=tk.EXTENDED,
                        bg="#252528", fg="#dcdcdc", highlightthickness=0, borderwidth=0,
                        exportselection=False)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        order = list(mine)

        def refill_lb(keep=0):
            lb.delete(0, tk.END)
            for w in order:
                t = w.totals()
                lb.insert(tk.END, f"  {w.name}   ({t['images']}장 · {w.kind})")
            lb.selection_set(0, tk.END)
            if order:
                lb.see(min(keep, len(order) - 1))

        def move(delta):
            sel = list(lb.curselection())
            if len(sel) != 1:
                return
            i = sel[0]
            j = i + delta
            if 0 <= j < len(order):
                order[i], order[j] = order[j], order[i]
                refill_lb(j)
                lb.selection_clear(0, tk.END)
                lb.selection_set(j)

        btncol = ttk.Frame(lbf)
        btncol.pack(side=tk.LEFT, padx=6)
        ttk.Button(btncol, text="▲", width=3, command=lambda: move(-1)).pack()
        ttk.Button(btncol, text="▼", width=3, command=lambda: move(1)).pack(pady=2)
        refill_lb()

        opt = ttk.Frame(frm)
        opt.pack(fill=tk.X, pady=(12, 0))
        opt.columnconfigure(1, weight=1)

        v_mode = tk.StringVar(value="ratio" if len(self.project.splits) <= 1 else "keep")
        v_tr = tk.StringVar(value="0.8")
        v_va = tk.StringVar(value="0.1")
        v_te = tk.StringVar(value="0.1")
        v_seed = tk.StringVar(value="0")
        v_unlab = tk.BooleanVar(value=False)
        v_name = tk.StringVar(
            value=f"{time.strftime('%Y%m%d_%H%M')}_{safe_name(self.project.name)}")

        ttk.Label(opt, text="스플릿").grid(row=0, column=0, sticky="nw", pady=3)
        mfrm = ttk.Frame(opt)
        mfrm.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(mfrm, text=f"원본 스플릿 유지 "
                                   f"({', '.join(s.name for s in self.project.splits)})",
                        value="keep", variable=v_mode).pack(anchor="w")
        ttk.Radiobutton(mfrm, text="비율로 다시 나누기", value="ratio",
                        variable=v_mode).pack(anchor="w")
        ratio_row = ttk.Frame(mfrm)
        ratio_row.pack(anchor="w", padx=(20, 0), pady=2)
        for lbl, var in (("train", v_tr), ("valid", v_va), ("test", v_te)):
            ttk.Label(ratio_row, text=f"  {lbl}").pack(side=tk.LEFT)
            ttk.Entry(ratio_row, textvariable=var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(ratio_row, text="  시드").pack(side=tk.LEFT)
        ttk.Entry(ratio_row, textvariable=v_seed, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(opt, text="라벨 없는 이미지도 배경(빈 txt)으로 포함",
                        variable=v_unlab).grid(row=1, column=1, sticky="w",
                                               padx=(8, 0), pady=3)
        ttk.Label(opt, text="내보낼 폴더명").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(opt, textvariable=v_name).grid(row=2, column=1, sticky="we",
                                                 padx=(8, 0), pady=3)
        ttk.Label(opt, text="이미지는 실제 복사됩니다.", style="Head.TLabel").grid(
            row=3, column=1, sticky="w", padx=(8, 0))

        info = ttk.Label(frm, text="", style="Head.TLabel", justify="left")
        info.pack(anchor="w", pady=(10, 0))

        def selected_ws():
            sel = lb.curselection()
            return [order[i] for i in sel] if sel else []

        def preview():
            ws_sel = selected_ws()
            if not ws_sel:
                info.config(text="작업공간을 하나 이상 선택하세요.")
                return None
            plan = ExportPlan(
                out_dir="", project=self.project, sources=ws_sel,
                split_mode=v_mode.get(), include_unlabeled=v_unlab.get())
            items = plan.collect()
            per: Dict[str, int] = {}
            for it in items:
                k = it["workspace"] or "(라벨 없음)"
                per[k] = per.get(k, 0) + 1
            info.config(text=f"내보낼 이미지 {len(items)}장   "
                             + "  ".join(f"{k} {v}장" for k, v in per.items()))
            return items

        def run():
            ws_sel = selected_ws()
            if not ws_sel:
                messagebox.showerror("내보내기", "작업공간을 선택하세요.", parent=top)
                return
            try:
                ratios = (float(v_tr.get()), float(v_va.get()), float(v_te.get()))
                sd = int(v_seed.get())
            except ValueError:
                messagebox.showerror("내보내기", "비율/시드를 확인하세요.", parent=top)
                return
            name = safe_name(v_name.get().strip() or "export")
            out = os.path.join(lay.export_dir, name)
            if os.path.isdir(out) and os.listdir(out):
                if not messagebox.askyesno(
                    "내보내기", f"{name} 폴더가 이미 있고 비어 있지 않습니다.\n"
                                f"그 안에 덮어쓸까요?", parent=top):
                    return
            plan = ExportPlan(
                out_dir=out, project=self.project, sources=ws_sel,
                split_mode=v_mode.get(), ratios=ratios, seed=sd,
                include_unlabeled=v_unlab.get(),
                class_names=list(ws_sel[0].classes or self.project.names))
            top.destroy()

            self.pb.pack(side=tk.BOTTOM, fill=tk.X)
            self.pb["value"] = 0
            result: dict = {}
            q: "queue.Queue[Tuple[int, int, str]]" = queue.Queue()

            def worker():
                try:
                    result["manifest"] = plan.run(lambda d, t, m: q.put((d, t, m)))
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    result["error"] = str(e)
                result["done"] = True

            threading.Thread(target=worker, daemon=True).start()

            def poll():
                last = None
                try:
                    while True:
                        last = q.get_nowait()
                except queue.Empty:
                    pass
                if last:
                    d, t, m = last
                    self.pb["maximum"] = max(t, 1)
                    self.pb["value"] = d
                    self.status.config(text=f"내보내는 중… {d}/{t}  {m}")
                if result.get("done"):
                    self.pb.pack_forget()
                    if "error" in result:
                        messagebox.showerror("내보내기 실패", result["error"])
                        return
                    mf = result["manifest"]
                    c = mf["counts"]
                    self.status.config(
                        text=f"내보내기 완료 · train {c['train']} / valid {c['valid']}"
                             f" / test {c['test']} → {out}")
                    if messagebox.askyesno(
                        "내보내기 완료",
                        f"train {c['train']} · valid {c['valid']} · test {c['test']}\n"
                        f"{out}\n\n폴더를 열까요?"):
                        try:
                            os.startfile(out)  # noqa: S606
                        except (AttributeError, OSError):
                            pass
                else:
                    self.after(80, poll)

            self.after(80, poll)

        lb.bind("<<ListboxSelect>>", lambda e: preview())
        for v in (v_mode, v_unlab):
            v.trace_add("write", lambda *a: preview())
        preview()

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(btns, text="취소", command=top.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="내보내기", command=run).pack(side=tk.RIGHT)

    # ---------------------------------------------------------------- 오류 처리
    def report_callback_exception(self, exc, val, tb) -> None:
        """Tk 콜백에서 터진 예외를 삼키지 않는다.

        pythonw 로 띄우면 콘솔이 없어 traceback 이 갈 곳이 없다. 편집 중에 조용히
        아무 일도 안 일어나는 것보다 파일에 남기고 알려주는 편이 낫다.
        """
        text = "".join(traceback.format_exception(exc, val, tb))
        try:
            with open(error_log_path(), "a", encoding="utf-8") as f:
                f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{text}")
        except OSError:
            pass
        sys.stderr.write(text)
        try:
            messagebox.showerror(
                "오류",
                f"{type(val).__name__}: {val}\n\n"
                f"작업 중이던 내용은 그대로 있습니다. 저장(Ctrl+S)을 눌러보세요.\n\n"
                f"자세한 내용: {error_log_path()}")
        except Exception:  # noqa: BLE001  (메시지창마저 실패하면 포기)
            pass

    # ------------------------------------------------------------------ 종료
    def _on_close(self) -> None:
        self.save_current()
        if self._validator:
            self._validator.cancel()
        # 우리가 띄운 워커는 우리가 거둔다. 죽이지 않고 신호만 남기는 이유는
        # 강제 종료하면 돌던 학습 몇 분이 통째로 날아가기 때문이다.
        if self._worker_proc is not None and self.layout:
            w = Workspace._worker()
            if w:
                w.ask_stop(self.layout.root)
        self.loader.shutdown()
        self.destroy()


# ==============================================================================
# ENTRY
# ==============================================================================
def error_log_path() -> str:
    """오류 기록 위치. 스크립트 옆의 docs/ 에 둔다."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "error.log")
    return os.path.join(d, "error.log")


def _install_excepthook() -> None:
    """시작 단계에서 터지는 예외도 파일에 남긴다 (pythonw 는 콘솔이 없다)."""
    def hook(exc, val, tb):
        text = "".join(traceback.format_exception(exc, val, tb))
        try:
            with open(error_log_path(), "a", encoding="utf-8") as f:
                f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작 실패 ====="
                        f"\n{text}")
        except OSError:
            pass
        sys.stderr.write(text)
        try:
            import tkinter.messagebox as mb
            r = tk.Tk()
            r.withdraw()
            mb.showerror("시작 실패", f"{type(val).__name__}: {val}\n\n"
                                      f"자세한 내용: {error_log_path()}")
            r.destroy()
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def main() -> None:
    _install_excepthook()
    root_arg = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else None
    if root_arg is None:
        # 인자가 없으면 스크립트가 놓인 폴더에서 db/work/export 배치를 찾아
        # 데이터셋이 하나뿐이면 그걸 바로 연다. 매번 폴더를 고르게 하지 않는다.
        lay = Layout.discover(os.path.dirname(os.path.abspath(__file__)))
        if lay:
            ds = lay.datasets()
            if len(ds) == 1:
                root_arg = ds[0]
    app = App(root_arg)
    if not _HAS_YAML:
        print("[info] PyYAML 없음 -> 내장 미니 파서 사용 (pip install pyyaml 권장)")
    if not _HAS_PIL:
        print("[info] Pillow 없음 -> raw PPM 경로 사용 (pip install pillow 시 더 빠름)")
    app.mainloop()


if __name__ == "__main__":
    main()
