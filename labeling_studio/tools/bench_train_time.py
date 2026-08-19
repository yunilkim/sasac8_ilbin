# -*- coding: utf-8 -*-
"""학습 한 판의 시간이 어디로 가는가 -- 검증 / 학습계산 / 워커 spawn 으로 분해한다.

연속형 라벨링에서는 모델을 자주 갈아끼우는 것이 목적이라 학습 한 판의 길이가
곧 '사람이 낡은 예측을 보는 시간'이 된다. 그래서 mAP 가 아니라 초를 재야 한다.

--------------------------------------------------------------------------
 [ 2026-08-18 RTX 5070 Ti 실측 ]
--------------------------------------------------------------------------
 1) 학습 에폭은 검증이 지배한다 (12에폭, 정상속도 구간)

      학습분    val=True   val=False   검증 몫
      150장     7.65초/ep   1.50초/ep   6.15초 (80%)
      600장     9.55초/ep   4.03초/ep   5.52초 (58%)

    학습 계산 = 0.66초/에폭 + 장당 5.6ms
    검증     = 약 5.8초/에폭, 학습분과 무관 (valid 1258장 고정)

    -> 150장을 500장으로 늘려도 학습 계산은 에폭당 2초 느는 게 전부다.
       검증이 상수라 총 시간이 거의 안 변한다. 80에폭 완주 환산:
       50장 9.3분 / 150장 9.9분(실측 10.2분) / 500장 12.2분.

 2) 검증 설정은 무엇을 바꿔도 안 빨라진다 (standalone val, 1258장)

      기본 batch=16        23.13초   mAP50 0.8124
      batch=64             23.57초   0.8125
      batch=128           106.76초   0.8126   <- 오히려 4.6배 느려짐
      batch=64 + half      23.47초   0.8123
      batch=64 conf=0.01   23.74초   0.7977   <- 시간 그대로, 점수만 깎임
      batch=64 max_det=100 24.11초   0.8125

    GPU 가 병목이 아니다. 배치도 정밀도도 NMS 도 아니다.

 3) 진짜 범인은 윈도우 워커 spawn -- 워커 하나당 +2초

      workers=0    8.35초 (151 img/s)
      workers=2   10.52초
      workers=4   15.17초
      workers=8   24.45초 ( 51 img/s)

    리눅스 fork 와 달리 윈도우 spawn 은 워커마다 파이썬+torch 를 새로
    import 한다. 검증 8초짜리에 spawn 16초를 얹고 있었다.
    단 학습 루프는 검증 로더를 한 번 만들어 80에폭 재사용하므로 학습
    중에는 해당 없다(그래서 학습 중 검증은 6.15초). train_model.py 의
    최종 검증은 이미 workers=0 을 쓰고 있다.

 4) 효과가 있는 유일한 것 -- 검증셋 크기

      검증 1258장   7.65초/에폭
      검증  400장   2.90초/에폭     (2.6배)

    확인 완료 -- 학습분 150장·seed 0 으로 80에폭 완주시키고 나온 best.pt 를
    valid 전량(1258장)으로 다시 재서 비교했다.

      검증 1258장 · 매 에폭   10.2분   mAP50 0.8132   mAP50-95 0.5213
      검증 1258장 · 5에폭마다  4.2분         0.8124            0.5213
      검증  400장 · 매 에폭    5.1분         0.8124            0.5213
      검증  400장 · 5에폭마다  3.0분         0.8124            0.5213

    3.4배 빨라지고 손해는 mAP50 0.0008 -- 잡음(0.002) 이하다.

    손해의 원인은 둘로 나눌 수 없다. 검증셋만 줄여도, 주기만 늘려도, 둘 다
    해도 전부 0.8124 다. 기준만 epoch 79 를 골랐고 나머지 셋은 epoch 80 을
    골랐기 때문이다(후반 곡선이 평평해 둘이 사실상 같다). 0.0008 은 어느
    설정을 바꿨느냐와 무관하게 best 에폭 선택이 한 칸 옮겨간 값이다.
    -> train_model.py --val-fast-n 400 --val-period 5 가 이것이다.

 5) GPU 디코드는 여기서 의미 없다

    검증 205 img/s 에서 디코드 몫은 4.5%. 워커를 늘려 디코드를 병렬화해도
    안 빨라진다는 것이 증거다(오히려 느려진다). 추론에서 GPU 디코드가
    4배였던 것은 그때 디코드가 94%였기 때문이고 지금은 다른 작업 뒤에
    숨어 있다. 관련: memory/autolabel-perf

--------------------------------------------------------------------------
 실행:
     .venv\\Scripts\\python.exe tools\\bench_train_time.py [--full]

     기본은 검증 스윕만(빠름). --full 이면 학습 분해까지 다시 잰다.
     임시 데이터셋은 %TEMP% 아래 하드링크라 디스크를 안 먹는다.
--------------------------------------------------------------------------
"""
import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
import time

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# 작업 루트는 이 스크립트 위치에서 유도한다(tools/ 의 부모).
# 절대경로를 박아 두면 저장소를 받은 사람 환경에서 바로 깨진다.
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(D, "db", "vest-helmet_crop_dedup")
REF = os.path.join(D, "models", "y11s_n150_dedup")
SCRATCH = os.path.join(tempfile.gettempdir(), "bench_train_time")
NVAL_FULL = 1258


def _link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(n_train, n_val=None):
    """train n장 + val n_val장(None 이면 전량) 임시 데이터셋을 하드링크로 만든다."""
    root = os.path.join(SCRATCH, "t%d_v%s" % (n_train, n_val or "all"))
    yml = os.path.join(root, "data.yaml")
    imgs = os.path.join(root, "train", "images")
    if os.path.isdir(imgs) and len(os.listdir(imgs)) == n_train and os.path.exists(yml):
        return yml  # db 라벨이 읽기전용이라 지우지 않고 재사용한다

    pool = sorted(os.listdir(os.path.join(SRC, "train", "images")))
    random.Random(0).shuffle(pool)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(root, "train", sub), exist_ok=True)
        os.makedirs(os.path.join(root, "val", sub), exist_ok=True)
    for fn in pool[:n_train]:
        stem = os.path.splitext(fn)[0]
        _link(os.path.join(SRC, "train", "images", fn),
              os.path.join(root, "train", "images", fn))
        _link(os.path.join(SRC, "train", "labels", stem + ".txt"),
              os.path.join(root, "train", "labels", stem + ".txt"))

    vsrc = os.path.join(REF, "data", "val")
    vi = sorted(os.listdir(os.path.join(vsrc, "images")))
    if n_val:
        random.Random(0).shuffle(vi)
        vi = vi[:n_val]
    for fn in vi:
        stem = os.path.splitext(fn)[0]
        _link(os.path.join(vsrc, "images", fn), os.path.join(root, "val", "images", fn))
        _link(os.path.join(vsrc, "labels", stem + ".txt"),
              os.path.join(root, "val", "labels", stem + ".txt"))

    with open(yml, "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\n"
                "nc: 2\nnames: ['reflective_jacket', 'safety_helmet']\n" % root)
    return yml


def steady_epoch(run_dir):
    """정상속도 구간의 에폭당 초. 1에폭은 워밍업, 마지막은 최종검증이 붙어 뺀다."""
    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8") as f:
        ts = [float(r["time"]) for r in csv.DictReader(f)]
    d = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    tail = d[-4:-1] or d
    return sum(tail) / len(tail)


def bench_train(n_train, n_val, do_val, epochs=12):
    from ultralytics import YOLO
    yml = build(n_train, n_val)
    out = os.path.join(SCRATCH, "runs")
    name = "t%d_v%s_%d" % (n_train, n_val or "all", int(do_val))
    shutil.rmtree(os.path.join(out, name), ignore_errors=True)
    YOLO(os.path.join(D, "yolo11s.pt")).train(
        data=yml, epochs=epochs, imgsz=640, batch=16, workers=8, device=0,
        amp=True, val=do_val, plots=False, patience=epochs + 1, seed=0,
        project=out, name=name, exist_ok=True, verbose=False)
    per = steady_epoch(os.path.join(out, name))
    print("  학습 %4d장 · 검증 %4s장 · val=%-5s -> %5.2f 초/에폭"
          % (n_train, n_val or NVAL_FULL, do_val, per), flush=True)
    return per


def bench_val(tag, **kw):
    from ultralytics import YOLO
    w = os.path.join(REF, "weights", "best.pt")
    if not os.path.exists(w):
        w = os.path.join(REF, "weights", "last.pt")
    m, yml = YOLO(w), build(150)
    common = dict(data=yml, device=0, verbose=False, plots=False,
                  project=os.path.join(SCRATCH, "vruns"), name="w", exist_ok=True)
    m.val(**common, **kw)  # 워밍업
    t0 = time.time()
    r = m.val(**common, **kw)
    d = time.time() - t0
    print("  %-30s%7.2f초  %6.0f img/s  mAP50 %.4f"
          % (tag, d, NVAL_FULL / d, r.box.map50), flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="학습 분해까지 다시 잰다 (느림)")
    a = ap.parse_args()

    print("[1] 검증 설정 스윕 (valid 1258장) — 무엇을 바꿔도 안 빨라진다")
    bench_val("기본 (batch=16, workers=8)", batch=16, workers=8)
    bench_val("batch=64", batch=64, workers=8)
    bench_val("batch=64 + half", batch=64, workers=8, half=True)
    bench_val("batch=64 + max_det=100", batch=64, workers=8, max_det=100)

    print("\n[2] 워커 spawn 비용 — 윈도우는 워커 하나당 +2초")
    for w in (0, 2, 4, 8):
        bench_val("workers=%d" % w, batch=64, workers=w)

    if not a.full:
        print("\n(학습 분해는 --full. 위 값은 docstring 기록과 대조할 것.)")
        return 0

    print("\n[3] 학습 에폭 분해 (12에폭)")
    r = {}
    for n in (150, 600):
        for v in (True, False):
            r[(n, v)] = bench_train(n, None, v)
    slope = (r[(600, False)] - r[(150, False)]) / 450
    print("\n  학습 장당비용 %.2f ms/장/에폭 · 고정비 %.2f 초/에폭"
          % (slope * 1000, r[(150, False)] - slope * 150))
    print("  검증 고정비  n=150 %.2f초 · n=600 %.2f초"
          % (r[(150, True)] - r[(150, False)], r[(600, True)] - r[(600, False)]))

    print("\n[4] 검증셋 축소 효과")
    bench_train(150, NVAL_FULL, True)
    bench_train(150, 400, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
