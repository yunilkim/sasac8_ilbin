"""라벨 마스크가 얼마나 미덥지 않은가 — 학습된 모델과의 불일치로 잰다.

00039 를 뜯어 보니 문제는 '지그재그'가 아니라 SAM 마스크가 옆 물체(초록 자재)를
같이 먹은 것이었다. 폴리곤은 그 마스크를 충실히 옮겼을 뿐이다(충실도 0.91).

이런 '의미적 과잉 포함'은 기하 신호로 못 잡는다. 대신 표준적인 방법이 있다 --
학습된 모델의 예측과 라벨이 크게 어긋나는 것을 의심 목록에 올린다
(confident learning 계열). 모델이 데이터 전반에서 배운 '조끼다움'과 그 라벨이
어긋나면, 둘 중 하나가 틀렸고 대개는 라벨이다.

y11sseg_n150_dedup 의 학습분 150장은 제외한다 -- 본 이미지에서는 라벨을 그대로
외웠을 것이라 불일치가 과소평가된다.
"""
import json
import os
import random

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 작업 루트는 이 스크립트 위치에서 유도한다(tools/ 의 부모).
# 절대경로를 박아 두면 저장소를 받은 사람 환경에서 바로 깨진다.
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG = os.path.join(D, "db", "vest-helmet_crop_dedup_seg")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(D, "models", "y11sseg_n150_dedup")
NAMES = ["reflective_jacket", "safety_helmet"]
N = 300


def main():
    from ultralytics import YOLO
    meta = json.load(open(os.path.join(MODEL, "model.json"), encoding="utf-8"))
    trained = set(meta["train_image_list"])
    m = YOLO(os.path.join(MODEL, "weights", "best.pt"))

    idir = os.path.join(SEG, "train", "images")
    ldir = os.path.join(SEG, "train", "labels")
    files = [f for f in sorted(os.listdir(idir)) if f not in trained]
    random.Random(5).shuffle(files)
    files = files[:N]
    print(f"학습분 제외 후 {len(files)}장 (모델이 본 적 없는 이미지)")

    rows = []
    for n, fn in enumerate(files, 1):
        img = cv2.imdecode(np.fromfile(os.path.join(idir, fn), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        lab = []
        for l in open(os.path.join(ldir, os.path.splitext(fn)[0] + ".txt"),
                      encoding="utf-8"):
            t = l.split()
            if len(t) >= 7:
                p = (np.array([float(x) for x in t[1:]], np.float64)
                     .reshape(-1, 2) * [w, h]).astype(np.int32)
                mk = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mk, [p], 1)
                lab.append((int(float(t[0])), mk, len(p)))
        if not lab:
            continue
        r = m.predict(img, verbose=False, device=0, conf=0.25, imgsz=640,
                      retina_masks=True)[0]
        pred = []
        if r.masks is not None:
            pm = r.masks.data.cpu().numpy().astype(np.uint8)
            pc = r.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(pm)):
                mk = pm[k]
                if mk.shape != (h, w):
                    mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
                pred.append((int(pc[k]), mk))
        for cls, lm, npts in lab:
            best = 0.0
            for pc_, pmk in pred:
                if pc_ != cls:
                    continue
                inter = int(np.logical_and(lm, pmk).sum())
                uni = int(np.logical_or(lm, pmk).sum())
                best = max(best, inter / max(uni, 1))
            rows.append({"file": fn, "cls": cls, "iou": round(best, 4),
                         "pts": npts, "area": int(lm.sum())})
        if n % 60 == 0:
            print(f"  {n}/{len(files)}")

    v = np.array([r["iou"] for r in rows])
    print(f"\n객체 {len(rows)}개 · 라벨 마스크 vs 모델 예측 IoU")
    for q in (5, 10, 25, 50, 75, 90):
        print(f"  {q:>3}백분위 {np.percentile(v, q):.3f}")
    print(f"  0.5 미만 {np.mean(v < 0.5):.1%} · 0.3 미만 {np.mean(v < 0.3):.1%} "
          f"· 0(예측 없음) {np.mean(v == 0):.1%}")
    for ci, nm in enumerate(NAMES):
        s = np.array([r["iou"] for r in rows if r["cls"] == ci])
        if len(s):
            print(f"  {nm:<20} n={len(s):>5} 중앙값 {np.median(s):.3f} · "
                  f"0.5미만 {np.mean(s < 0.5):.1%}")

    # 점 개수(지그재그 정도)와 불일치가 상관이 있나
    pts = np.array([r["pts"] for r in rows])
    hi = v[pts > 200]
    lo = v[pts <= 200]
    print(f"\n  점 200 초과 (지그재그) n={len(hi)} 중앙값 "
          f"{np.median(hi) if len(hi) else float('nan'):.3f}")
    print(f"  점 200 이하           n={len(lo)} 중앙값 {np.median(lo):.3f}")
    print("  -> 지그재그가 실제로 나쁜 라벨인지 여기서 갈린다")

    rows.sort(key=lambda r: r["iou"])
    json.dump(rows, open(os.path.join(HERE, "label_vs_model.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  가장 어긋난 5개")
    for r in rows[:5]:
        print(f"    IoU {r['iou']:.3f}  {NAMES[r['cls']]:<18} "
              f"점 {r['pts']:>4}  {r['file'][:34]}")


if __name__ == "__main__":
    main()
